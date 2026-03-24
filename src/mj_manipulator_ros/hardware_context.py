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
- JointState subscription for state feedback
- GripperCommand actions for grasp/release
- Streaming joint command topics for step()/step_cartesian()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import rclpy.executors
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint

from mj_manipulator_ros.config import HardwareConfig
from mj_manipulator_ros.hardware_arm_controller import HardwareArmController
from mj_manipulator_ros.interfaces import (
    ROBOT_STATUS_TOPIC,
    joint_commands_topic,
)
from mj_manipulator_ros.ros_arm_client import ArmTrajectoryClient
from mj_manipulator_ros.ros_gripper_client import GripperClient
from mj_manipulator_ros.ros_state_listener import JointStateListener
from mj_manipulator_ros.trajectory_convert import trajectory_to_msg

if TYPE_CHECKING:
    from mj_manipulator.planning import PlanResult
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


class HardwareContext:
    """ExecutionContext for real robot hardware via ROS 2.

    Args:
        config: Hardware configuration (arm names, joints, control rate).
        node_name: Name for the ROS 2 node.
    """

    def __init__(
        self,
        config: HardwareConfig,
        node_name: str = "hardware_context",
    ):
        self._config = config
        self._node_name = node_name

        # Initialized in __enter__
        self._node = None
        self._spin_thread = None
        self._state_listener = None
        self._arm_clients: dict[str, ArmTrajectoryClient] = {}
        self._gripper_clients: dict[str, GripperClient] = {}
        self._arm_controllers: dict[str, HardwareArmController] = {}
        self._streaming_pubs: dict[str, object] = {}
        self._robot_status = True
        self._running = False

    def __enter__(self) -> HardwareContext:
        """Connect to ROS 2 and wait for all action servers."""
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(self._node_name)

        # Joint state listener
        self._state_listener = JointStateListener(self._node)

        # Robot status subscriber
        self._node.create_subscription(
            Bool, ROBOT_STATUS_TOPIC, self._status_callback, 10,
        )

        # Per-arm clients
        for arm_config in self._config.arms:
            name = arm_config.name

            # Trajectory action client
            traj_client = ArmTrajectoryClient(self._node, name)
            self._arm_clients[name] = traj_client

            # Gripper action client
            gripper_client = None
            if arm_config.has_gripper:
                gripper_client = GripperClient(self._node, name)
                self._gripper_clients[name] = gripper_client

            # Arm controller
            self._arm_controllers[name] = HardwareArmController(
                arm_config, gripper_client,
            )

            # Streaming publisher
            self._streaming_pubs[name] = self._node.create_publisher(
                JointTrajectoryPoint, joint_commands_topic(name), 10,
            )

        # Spin in background thread
        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True,
        )
        self._spin_thread.start()

        # Wait for action servers
        logger.info("Waiting for action servers...")
        for name, client in self._arm_clients.items():
            if not client.wait_for_server(timeout_sec=10.0):
                raise TimeoutError(
                    f"Timed out waiting for {name} trajectory action server"
                )
        for name, client in self._gripper_clients.items():
            if not client.wait_for_server(timeout_sec=10.0):
                raise TimeoutError(
                    f"Timed out waiting for {name} gripper action server"
                )

        # Wait for joint states
        logger.info("Waiting for joint states...")
        deadline = time.time() + 10.0
        while not self._state_listener.has_data and time.time() < deadline:
            time.sleep(0.01)
        if not self._state_listener.has_data:
            raise TimeoutError("Timed out waiting for /joint_states")

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
                    point = JointTrajectoryPoint()
                    point.positions = q.tolist()
                    pub.publish(point)

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
            point = JointTrajectoryPoint()
            point.positions = np.asarray(position).tolist()
            if velocity is not None:
                point.velocities = np.asarray(velocity).tolist()
            pub.publish(point)

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

        msg = trajectory_to_msg(traj)
        return client.send_trajectory(msg, timeout_sec=self._config.action_timeout)

    def _status_callback(self, msg: Bool) -> None:
        self._robot_status = msg.data

    def _spin_loop(self) -> None:
        """Background thread: spin the ROS 2 node for callbacks."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
