# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""HardwareContext — ExecutionContext implementation for real robots via ROS 2.

Implements the same interface as SimContext so identical user code works
on both simulation and real hardware:

    # Simulation
    with robot.sim() as ctx:
        robot.pickup("can")

    # Real hardware
    with robot.hardware() as ctx:
        robot.pickup("can")

Uses standard ROS 2 interfaces:
- FollowJointTrajectory actions for batch trajectory execution
- JointState subscription for state feedback (via JointStateListener)
- GripperCommand actions for grasp/release
- forward_position_controller/commands (Float64MultiArray) for streaming

By default each arm uses **forward_position_controller** after connect so
``step`` / ``step_cartesian`` publish immediately without a controller switch.
The trajectory controller is activated only for ``execute()`` and the context
switches back to forward position when the trajectory finishes.

Optional: pass ``ssot_path`` to load YAML and build config via
:mod:`mj_manipulator_ros.bimanual_ssot` (UR5e bimanual layout only).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rclpy
import yaml
from controller_manager_msgs.srv import ListControllers, SwitchController
from std_msgs.msg import Bool, Float64MultiArray

from mj_manipulator_ros.bimanual_ssot import (
    hardware_config_from_bimanual_ssot,
    named_poses_from_bimanual_ssot,
)
from mj_manipulator_ros.config import HardwareConfig
from mj_manipulator_ros.hardware_arm_controller import HardwareArmController
from mj_manipulator_ros.interfaces import (
    FORWARD_POSITION_CONTROLLER,
    ROBOT_STATUS_TOPIC,
    SCALED_JOINT_TRAJECTORY_CONTROLLER,
    follow_joint_trajectory_action,
    forward_position_commands_topic,
    gripper_command_action,
)
from mj_manipulator_ros.ros_arm_client import ArmTrajectoryClient
from mj_manipulator_ros.ros_gripper_client import GripperClient
from mj_manipulator_ros.ros_state_listener import JointStateListener
from mj_manipulator_ros.trajectory_convert import trajectory_to_msg

if TYPE_CHECKING:
    from mj_manipulator.planning import PlanResult
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


def _wait(future, timeout_sec: float = 30.0):
    """Block until *future* completes (node is spun in background)."""
    deadline = time.monotonic() + timeout_sec
    while not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("ROS future did not complete in time")
        time.sleep(0.01)
    return future.result()


class HardwareContext:
    """ExecutionContext for real robot hardware via ROS 2.

    Args:
        config: Pre-built hardware configuration.
        ssot_path: If set, ``config`` is built from this YAML (bimanual SSoT).
        node_name: ROS 2 node name.
    """

    def __init__(
        self,
        config: HardwareConfig | None = None,
        *,
        ssot_path: str | Path | None = None,
        node_name: str = "hardware_context",
    ):
        if ssot_path is not None:
            with open(Path(ssot_path), encoding="utf-8") as f:
                ssot = yaml.safe_load(f)
            self._ssot: dict[str, Any] | None = ssot
            self._named_poses = named_poses_from_bimanual_ssot(ssot)
            config = hardware_config_from_bimanual_ssot(ssot)
        else:
            self._ssot = None
            self._named_poses = {}
            if config is None:
                raise ValueError("Provide ``config`` or ``ssot_path``")

        self._config = config
        self._node_name = node_name

        self._node = None
        self._spin_thread = None
        self._state_listener = None
        self._arm_clients: dict[str, ArmTrajectoryClient] = {}
        self._gripper_clients: dict[str, GripperClient] = {}
        self._arm_controllers: dict[str, HardwareArmController] = {}
        self._streaming_pubs: dict[str, object] = {}
        self._switch_clients: dict[str, object] = {}
        self._list_clients: dict[str, object] = {}
        self._active_controller: dict[str, str | None] = {}
        self._robot_status = True
        self._running = False

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Home/ready joint vectors from SSoT (empty if config was not from SSoT)."""
        return self._named_poses

    def __enter__(self) -> HardwareContext:
        """Connect to ROS 2 and wait for all action servers."""
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(self._node_name)

        self._state_listener = JointStateListener(self._node)

        self._node.create_subscription(
            Bool,
            ROBOT_STATUS_TOPIC,
            self._status_callback,
            10,
        )

        for arm_config in self._config.arms:
            name = arm_config.name
            fjt = arm_config.follow_joint_trajectory_action or follow_joint_trajectory_action(
                name,
                trajectory_controller=arm_config.joint_trajectory_controller,
            )
            traj_client = ArmTrajectoryClient(self._node, name, action_name=fjt)
            self._arm_clients[name] = traj_client

            gripper_client = None
            if arm_config.has_gripper:
                gact = (
                    arm_config.gripper_command_action
                    or gripper_command_action(name)
                )
                gripper_client = GripperClient(self._node, name, action_name=gact)
                self._gripper_clients[name] = gripper_client

            self._arm_controllers[name] = HardwareArmController(
                arm_config,
                gripper_client,
            )

            if arm_config.forward_position_commands_topic:
                stream_topic = arm_config.forward_position_commands_topic
            elif arm_config.forward_position_controller == FORWARD_POSITION_CONTROLLER:
                stream_topic = forward_position_commands_topic(name)
            else:
                stream_topic = (
                    f"/{name}/{arm_config.forward_position_controller}/commands"
                )
            self._streaming_pubs[name] = self._node.create_publisher(
                Float64MultiArray,
                stream_topic,
                10,
            )

            self._switch_clients[name] = self._node.create_client(
                SwitchController,
                f"/{name}/controller_manager/switch_controller",
            )
            self._list_clients[name] = self._node.create_client(
                ListControllers,
                f"/{name}/controller_manager/list_controllers",
            )

        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            daemon=True,
        )
        self._spin_thread.start()

        time.sleep(2)

        logger.info("Waiting for action servers...")
        for name, client in self._arm_clients.items():
            if not client.wait_for_server(timeout_sec=10.0):
                raise TimeoutError(
                    f"Timed out waiting for {name} trajectory action server",
                )
        for name, client in self._gripper_clients.items():
            if not client.wait_for_server(timeout_sec=10.0):
                raise TimeoutError(
                    f"Timed out waiting for {name} gripper action server",
                )

        for name in self._arm_clients:
            self._active_controller[name] = self._query_active_controller(name)
            logger.info("%s: active controller = %s", name, self._active_controller[name])

        logger.info("Waiting for joint states...")
        deadline = time.time() + 10.0
        while not self._state_listener.has_data and time.time() < deadline:
            time.sleep(0.01)
        if not self._state_listener.has_data:
            raise TimeoutError("Timed out waiting for /joint_states")

        # Default to forward position so streaming never pays a switch on first step.
        for name in self._arm_clients:
            self._ensure_forward_position_controller(name)
            logger.info(
                "%s: default active controller (streaming) = %s",
                name,
                self._active_controller.get(name),
            )

        logger.info("HardwareContext ready")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shut down ROS 2 node and background spinner."""
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        return False

    def _query_active_controller(self, arm_name: str) -> str | None:
        client = self._list_clients.get(arm_name)
        if client is None:
            return None
        if not client.wait_for_service(timeout_sec=2.0):
            logger.warning("ListControllers unavailable for %s", arm_name)
            return None
        try:
            result = _wait(client.call_async(ListControllers.Request()))
        except Exception as e:
            logger.warning("ListControllers failed for %s: %s", arm_name, e)
            return None
        arm_cfg = next((a for a in self._config.arms if a.name == arm_name), None)
        jtc = (
            arm_cfg.joint_trajectory_controller
            if arm_cfg is not None
            else SCALED_JOINT_TRAJECTORY_CONTROLLER
        )
        fpc = (
            arm_cfg.forward_position_controller
            if arm_cfg is not None
            else FORWARD_POSITION_CONTROLLER
        )
        for c in result.controller:
            if c.state == "active" and c.name in (jtc, fpc):
                return c.name
        return None

    def _switch_controller(
        self,
        arm_name: str,
        activate: list[str],
        deactivate: list[str],
    ) -> None:
        client = self._switch_clients[arm_name]
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"SwitchController unavailable for {arm_name}")
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        result = _wait(client.call_async(req))
        if not result.ok:
            raise RuntimeError(
                f"Controller switch failed on {arm_name}: "
                f"activate={activate}, deactivate={deactivate}",
            )
        logger.info("%s: switched controllers  -%s  +%s", arm_name, deactivate, activate)

    def _ensure_trajectory_controller(self, arm_name: str) -> None:
        """Ensure scaled (or configured) joint trajectory controller is active."""
        arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
        jtc = arm_cfg.joint_trajectory_controller
        if self._active_controller.get(arm_name) == jtc:
            return
        current = self._active_controller.get(arm_name)
        deactivate = [current] if current else []
        self._switch_controller(arm_name, activate=[jtc], deactivate=deactivate)
        self._active_controller[arm_name] = jtc

    def _ensure_forward_position_controller(self, arm_name: str) -> None:
        """Activate forward_position_controller (default for streaming)."""
        arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
        fpc = arm_cfg.forward_position_controller
        if self._active_controller.get(arm_name) == fpc:
            return
        current = self._active_controller.get(arm_name)
        deactivate = [current] if current else []
        self._switch_controller(arm_name, activate=[fpc], deactivate=deactivate)
        self._active_controller[arm_name] = fpc

    # -- ExecutionContext protocol -------------------------------------------

    def execute(self, item: Trajectory | PlanResult) -> bool:
        """Execute a trajectory or plan result via FollowJointTrajectory."""
        from mj_manipulator.planning import PlanResult
        from mj_manipulator.trajectory import Trajectory

        if isinstance(item, PlanResult):
            for traj in item.trajectories:
                if not self._execute_trajectory(traj):
                    return False
            return True
        elif isinstance(item, Trajectory):
            return self._execute_trajectory(item)
        else:
            raise TypeError(f"Cannot execute {type(item)}")

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Publish streaming joint commands for one control cycle."""
        if targets:
            for name, q in targets.items():
                pub = self._streaming_pubs.get(name)
                if pub is not None:
                    msg = Float64MultiArray()
                    msg.data = np.asarray(q, dtype=float).tolist()
                    pub.publish(msg)
        time.sleep(self._config.control_dt)

    def step_cartesian(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Publish streaming cartesian-resolved joint command."""
        pub = self._streaming_pubs.get(arm_name)
        if pub is not None:
            msg = Float64MultiArray()
            msg.data = np.asarray(position, dtype=float).tolist()
            pub.publish(msg)
        if velocity is not None:
            logger.debug(
                "step_cartesian: velocity ignored when using forward_position_controller",
            )
        time.sleep(self._config.control_dt)

    def sync(self) -> None:
        """Read latest joint states (no-op — listener updates continuously)."""
        pass

    def is_running(self) -> bool:
        """Check if hardware is connected and no e-stop is active."""
        return self._running and self._robot_status

    def arm(self, name: str) -> HardwareArmController:
        """Get per-arm controller for grasp/release."""
        if name not in self._arm_controllers:
            raise ValueError(f"Unknown arm: {name}")
        return self._arm_controllers[name]

    @property
    def control_dt(self) -> float:
        """Control timestep in seconds."""
        return self._config.control_dt

    # -- State access -------------------------------------------------------

    def get_joint_positions(self, joint_names: list[str]) -> np.ndarray | None:
        """Get latest joint positions from hardware feedback."""
        if self._state_listener is None:
            return None
        return self._state_listener.get_positions(joint_names)

    # -- Internal -----------------------------------------------------------

    def _execute_trajectory(self, traj: Trajectory) -> bool:
        """Send a single trajectory to the appropriate arm controller."""
        entity = traj.entity
        if entity is None:
            raise ValueError("Trajectory has no entity set")

        # Map entity to arm name (e.g. "left_arm" -> "left", "left" -> "left")
        arm_name = entity.replace("_arm", "")
        client = self._arm_clients.get(arm_name)
        if client is None:
            # Try exact entity name
            client = self._arm_clients.get(entity)
        if client is None:
            raise ValueError(f"No trajectory client for entity: {entity}")

        self._ensure_trajectory_controller(arm_name)
        msg = trajectory_to_msg(traj)
        try:
            return client.send_trajectory(msg, timeout_sec=self._config.action_timeout)
        finally:
            # Always return to forward position so streaming never stutters after a plan.
            try:
                self._ensure_forward_position_controller(arm_name)
            except Exception as e:
                logger.warning(
                    "%s: could not switch back to forward_position after trajectory: %s",
                    arm_name,
                    e,
                )

    def _status_callback(self, msg: Bool) -> None:
        self._robot_status = msg.data

    def _spin_loop(self) -> None:
        """Background thread: spin the ROS 2 node for callbacks."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
