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
    FORWARD_VELOCITY_CONTROLLER,
    ROBOT_STATUS_TOPIC,
    SCALED_JOINT_TRAJECTORY_CONTROLLER,
    follow_joint_trajectory_action,
    forward_position_commands_topic,
    forward_velocity_commands_topic,
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

def _mujoco_to_ros_joint_name(name: str) -> str:
    """Best-effort mapping for Geodude-style MuJoCo joint names to bimanual ROS names.

    Examples:
        ``left_ur5e/shoulder_pan_joint`` -> ``left_shoulder_pan_joint``
        ``right_ur5e/wrist_3_joint`` -> ``right_wrist_3_joint``

    If the name already looks like a ROS joint (no ``/``), it is returned unchanged.
    """
    if "/" not in name:
        return name
    # Common Geodude convention: {side}_ur5e/{suffix}
    if "_ur5e/" in name:
        return name.replace("_ur5e/", "_")
    # Fallback: make it a flat ROS-ish token.
    return name.replace("/", "_")


def _map_joint_names(names: list[str], mapping: dict[str, str]) -> list[str]:
    """Map each joint name via dict, defaulting to heuristic passthrough."""
    return [mapping.get(n, _mujoco_to_ros_joint_name(n)) for n in names]


def _reorder_vector(
    *,
    provided_joint_names: list[str],
    provided_values: np.ndarray,
    desired_joint_names: list[str],
    name_map: dict[str, str],
) -> np.ndarray:
    """Reorder a joint vector by name.

    Args:
        provided_joint_names: Joint names corresponding to provided_values. Can be ROS names
            (e.g. ``left_shoulder_pan_joint``) or MuJoCo-style names (e.g. ``left_ur5e/...``).
        provided_values: Joint values aligned with provided_joint_names.
        desired_joint_names: Desired joint name ordering (ROS / controller order).
        name_map: Per-arm mapping from config joint names to ROS names.

    Returns:
        Values reordered into desired_joint_names order.
    """
    prov_ros = _map_joint_names(list(provided_joint_names), name_map)
    desired_ros = _map_joint_names(list(desired_joint_names), name_map)
    v = np.asarray(provided_values, dtype=float).ravel()
    if v.size != len(prov_ros):
        raise ValueError(
            f"Expected {len(prov_ros)} values for joints {prov_ros}, got {v.size}",
        )
    index = {n: i for i, n in enumerate(prov_ros)}
    missing = [n for n in desired_ros if n not in index]
    if missing:
        raise ValueError(f"Missing joints in streaming target: {missing}")
    return np.asarray([v[index[n]] for n in desired_ros], dtype=float)


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
        default_streaming_controller: str | None = None,
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

        if default_streaming_controller is not None:
            config.default_streaming_controller = default_streaming_controller
        self._config = config
        self._node_name = node_name

        self._node = None
        self._spin_thread = None
        self._state_listener = None
        self._arm_clients: dict[str, ArmTrajectoryClient] = {}
        self._gripper_clients: dict[str, GripperClient] = {}
        self._arm_controllers: dict[str, HardwareArmController] = {}
        self._streaming_pubs: dict[str, object] = {}
        self._streaming_vel_pubs: dict[str, object] = {}
        self._switch_clients: dict[str, object] = {}
        self._list_clients: dict[str, object] = {}
        self._active_controller: dict[str, str | None] = {}
        # Per-arm mapping from configured joint names -> ROS /joint_states names.
        # This allows configs to use MuJoCo-style names (e.g. left_ur5e/...) while the
        # joint state stream uses bimanual ros2_control names (left_...).
        self._joint_name_map: dict[str, dict[str, str]] = {}
        self._robot_status = True
        self._running = False
        # Per-arm spacing between consecutive ``step_cartesian`` calls (same arm_name).
        self._step_cartesian_prev_time: dict[str, float] = {}
        self._step_cartesian_last_dt: dict[str, float] = {}
        self._step_cartesian_last_hz: dict[str, float] = {}
        # Throttle stdout FPS prints (one line per arm per second at most).
        self._step_cartesian_fps_print_at: dict[str, float] = {}

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
            # Build per-arm joint-name map (config name -> ROS joint name).
            self._joint_name_map[name] = {
                j: _mujoco_to_ros_joint_name(j) for j in arm_config.joint_names
            }
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

            # Velocity streaming (Float64MultiArray). Useful for teleop without
            # fighting scaled_joint_trajectory_controller which claims position interfaces.
            if getattr(arm_config, "forward_velocity_commands_topic", None):
                vel_topic = arm_config.forward_velocity_commands_topic
            elif (
                getattr(arm_config, "forward_velocity_controller", FORWARD_VELOCITY_CONTROLLER)
                == FORWARD_VELOCITY_CONTROLLER
            ):
                vel_topic = forward_velocity_commands_topic(name)
            else:
                vel_topic = f"/{name}/{arm_config.forward_velocity_controller}/commands"
            self._streaming_vel_pubs[name] = self._node.create_publisher(
                Float64MultiArray,
                vel_topic,
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
        # Wait until at least one joint state arrives (synchronized /joint_states).
        while not self._state_listener.has_data and time.time() < deadline:
            time.sleep(0.01)
        if not self._state_listener.has_data:
            raise TimeoutError("Timed out waiting for /joint_states")
        # Additionally, ensure all configured joints have been observed at least once.
        required_ros_joints: list[str] = []
        for arm_cfg in self._config.arms:
            required_ros_joints.extend(
                _map_joint_names(arm_cfg.joint_names, self._joint_name_map.get(arm_cfg.name, {})),
            )
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._state_listener.get_positions(required_ros_joints) is not None:
                break
            time.sleep(0.01)

        # Default streaming controller: position (typical) or velocity (teleop-friendly).
        for name in self._arm_clients:
            if self._config.default_streaming_controller == FORWARD_VELOCITY_CONTROLLER:
                self._ensure_forward_velocity_controller(name)
            else:
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
        fpc = arm_cfg.forward_position_controller
        if self._active_controller.get(arm_name) == jtc:
            return
        # Trajectory and forward-position controllers both claim the same joint command
        # interfaces; always deactivate the paired controller explicitly. Relying on
        # cached ``_active_controller`` alone can leave FPC active and make JTC
        # activation fail with "already claimed" (seen when switching modes).
        self._switch_controller(arm_name, activate=[jtc], deactivate=[fpc])
        self._active_controller[arm_name] = jtc

    def _ensure_forward_position_controller(self, arm_name: str) -> None:
        """Activate forward_position_controller (default for streaming)."""
        arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
        fpc = arm_cfg.forward_position_controller
        jtc = arm_cfg.joint_trajectory_controller
        if self._active_controller.get(arm_name) == fpc:
            return
        self._switch_controller(arm_name, activate=[fpc], deactivate=[jtc])
        self._active_controller[arm_name] = fpc

    def _ensure_forward_velocity_controller(self, arm_name: str) -> None:
        """Activate forward_velocity_controller for streaming joint velocities.

        This lets scaled JTC remain active (position interfaces) while streaming
        commands on velocity interfaces, avoiding JTC/FPC resource conflicts.
        """
        arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
        fvc = arm_cfg.forward_velocity_controller
        fpc = arm_cfg.forward_position_controller
        if self._active_controller.get(arm_name) == fvc:
            return
        self._switch_controller(arm_name, activate=[fvc], deactivate=[fpc])
        self._active_controller[arm_name] = fvc

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
        """Publish streaming joint commands for one control cycle.

        Backwards compatible behavior:
        - ``targets[arm] = q`` publishes q directly (assumed already in controller joint order).

        Safer behavior:
        - ``targets[arm] = (joint_names, q)`` will reorder q into the controller's expected
          joint order using joint names (accepts MuJoCo-style names too).
        """
        if targets:
            for name, q in targets.items():
                pub = self._streaming_pubs.get(name)
                if pub is not None:
                    msg = Float64MultiArray()
                    if isinstance(q, tuple):
                        provided_names, values = q
                        arm_cfg = next(a for a in self._config.arms if a.name == name)
                        reordered = _reorder_vector(
                            provided_joint_names=list(provided_names),
                            provided_values=np.asarray(values, dtype=float),
                            desired_joint_names=arm_cfg.joint_names,
                            name_map=self._joint_name_map.get(name, {}),
                        )
                        msg.data = reordered.tolist()
                    else:
                        msg.data = np.asarray(q, dtype=float).tolist()
                    pub.publish(msg)
        time.sleep(self._config.control_dt)

    def step_cartesian(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        *,
        joint_names: list[str] | None = None,
    ) -> None:
        """Publish streaming cartesian-resolved joint command.

        ``position`` is published on the arm's forward-position command topic.
        If ``joint_names`` is provided, positions are reordered into controller joint order.
        """
        now = time.time()
        prev = self._step_cartesian_prev_time.get(arm_name)
        if prev is not None:
            dt = now - prev
            self._step_cartesian_last_dt[arm_name] = dt
            if dt > 0.0:
                hz = 1.0 / dt
                self._step_cartesian_last_hz[arm_name] = hz
                last_print = self._step_cartesian_fps_print_at.get(arm_name, 0.0)
                if now - last_print >= 1.0:
                    print(f"step_cartesian[{arm_name}] {hz:.1f} Hz", flush=True)
                    self._step_cartesian_fps_print_at[arm_name] = now
        self._step_cartesian_prev_time[arm_name] = now

        # Only use velocity streaming if the context is explicitly configured for it.
        if (
            velocity is not None
            and arm_name in self._streaming_vel_pubs
            and self._config.default_streaming_controller == FORWARD_VELOCITY_CONTROLLER
        ):
            self._ensure_forward_velocity_controller(arm_name)
            pub = self._streaming_vel_pubs.get(arm_name)
            if pub is not None:
                msg = Float64MultiArray()
                if joint_names is not None:
                    arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
                    reordered = _reorder_vector(
                        provided_joint_names=list(joint_names),
                        provided_values=np.asarray(velocity, dtype=float),
                        desired_joint_names=arm_cfg.joint_names,
                        name_map=self._joint_name_map.get(arm_name, {}),
                    )
                    msg.data = reordered.tolist()
                else:
                    msg.data = np.asarray(velocity, dtype=float).tolist()
                pub.publish(msg)
        else:
            if velocity is not None:
                logger.debug("step_cartesian: velocity ignored (using forward_position_controller)")
            # Ensure the position streaming controller is active. If another node
            # (e.g., UR driver controller logic / MoveIt) activates JTC, publishing
            # to the FPC topic will stop having effect unless we switch back.
            self._ensure_forward_position_controller(arm_name)
            pub = self._streaming_pubs.get(arm_name)
            if pub is not None:
                msg = Float64MultiArray()
                if joint_names is not None:
                    arm_cfg = next(a for a in self._config.arms if a.name == arm_name)
                    reordered = _reorder_vector(
                        provided_joint_names=list(joint_names),
                        provided_values=np.asarray(position, dtype=float),
                        desired_joint_names=arm_cfg.joint_names,
                        name_map=self._joint_name_map.get(arm_name, {}),
                    )
                    msg.data = reordered.tolist()
                else:
                    msg.data = np.asarray(position, dtype=float).tolist()
                pub.publish(msg)
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

    def step_cartesian_dt(self, arm_name: str) -> float | None:
        """Seconds since the previous ``step_cartesian`` call for ``arm_name`` (``None`` on first call)."""
        return self._step_cartesian_last_dt.get(arm_name)

    def step_cartesian_hz(self, arm_name: str) -> float | None:
        """Instantaneous call rate ``1/dt`` from the last pair of calls for ``arm_name``."""
        return self._step_cartesian_last_hz.get(arm_name)

    # -- State access -------------------------------------------------------

    def get_joint_positions(self, joint_names: list[str]) -> np.ndarray | None:
        """Get latest joint positions from hardware feedback."""
        if self._state_listener is None:
            return None
        # Accept either ROS joint names (left_...) or MuJoCo-style names (left_ur5e/...).
        mapped = [_mujoco_to_ros_joint_name(n) for n in joint_names]
        return self._state_listener.get_positions(mapped)

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
        # Ensure ROS joint names match /joint_states and controller expectations.
        # Many planners produce trajectories with MuJoCo joint names; map them.
        msg.joint_names = _map_joint_names(
            list(msg.joint_names),
            self._joint_name_map.get(arm_name, {}),
        )
        try:
            return client.send_trajectory(msg, timeout_sec=self._config.action_timeout)
        finally:
            # Always return to the default streaming controller so streaming never stutters after a plan.
            try:
                if self._config.default_streaming_controller == FORWARD_VELOCITY_CONTROLLER:
                    self._ensure_forward_velocity_controller(arm_name)
                else:
                    self._ensure_forward_position_controller(arm_name)
            except Exception as e:
                logger.warning(
                    "%s: could not switch back to default streaming controller after trajectory: %s",
                    arm_name,
                    e,
                )

    def _status_callback(self, msg: Bool) -> None:
        self._robot_status = msg.data

    def _spin_loop(self) -> None:
        """Background thread: spin the ROS 2 node for callbacks."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
