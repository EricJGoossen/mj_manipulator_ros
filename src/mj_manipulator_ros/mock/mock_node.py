"""MuJoCo mock ROS 2 node.

Serves the same ROS 2 interfaces as real hardware (FollowJointTrajectory,
GripperCommand, /joint_states) but drives a MuJoCo simulation instead.
From HardwareContext's perspective, this is indistinguishable from a real robot.

Usage:
    ros2 run mj_manipulator_ros mock_node --ros-args -p model_path:=path/to/model.xml
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import rclpy
import rclpy.node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer

try:
    from rclpy.callback_group import ReentrantCallbackGroup
except ImportError:
    # Older RoboStack builds don't have callback_group as a separate module
    ReentrantCallbackGroup = None
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from mj_manipulator_ros.interfaces import (
    JOINT_STATES_TOPIC,
    ROBOT_STATUS_TOPIC,
    follow_joint_trajectory_action,
    gripper_command_action,
)

if TYPE_CHECKING:
    from mj_manipulator_ros.mock.mujoco_backend import MuJoCoBackend

logger = logging.getLogger(__name__)


class MockRobotNode(rclpy.node.Node):
    """ROS 2 node that serves robot interfaces backed by MuJoCo.

    Args:
        backend: MuJoCoBackend that drives the simulation.
        arm_names: List of arm names to create action servers for.
        publish_rate: Rate in Hz for publishing /joint_states.
    """

    def __init__(
        self,
        backend: MuJoCoBackend,
        arm_names: list[str],
        publish_rate: float = 500.0,
    ):
        super().__init__("mock_robot")
        self._backend = backend
        self._lock = threading.Lock()
        self._cb_group = ReentrantCallbackGroup()

        # Joint state publisher
        self._joint_state_pub = self.create_publisher(
            JointState, JOINT_STATES_TOPIC, 10,
        )
        self._timer = self.create_timer(
            1.0 / publish_rate, self._publish_joint_states,
        )

        # Robot status publisher (always True for mock)
        self._status_pub = self.create_publisher(
            Bool, ROBOT_STATUS_TOPIC, 10,
        )
        self._status_timer = self.create_timer(1.0, self._publish_status)

        # Per-arm action servers
        self._traj_servers: dict[str, ActionServer] = {}
        self._gripper_servers: dict[str, ActionServer] = {}

        for arm_name in arm_names:
            # FollowJointTrajectory
            self._traj_servers[arm_name] = ActionServer(
                self,
                FollowJointTrajectory,
                follow_joint_trajectory_action(arm_name),
                execute_callback=self._make_traj_callback(arm_name),
                callback_group=self._cb_group,
            )

            # GripperCommand
            self._gripper_servers[arm_name] = ActionServer(
                self,
                GripperCommand,
                gripper_command_action(arm_name),
                execute_callback=self._make_gripper_callback(arm_name),
                callback_group=self._cb_group,
            )

        logger.info("MockRobotNode ready with arms: %s", arm_names)

    def _make_traj_callback(self, arm_name: str):
        """Create a trajectory execution callback for a specific arm."""

        def callback(goal_handle):
            trajectory_msg = goal_handle.request.trajectory
            logger.info(
                "Executing trajectory on %s (%d points)",
                arm_name, len(trajectory_msg.points),
            )

            with self._lock:
                ok = self._backend.execute_trajectory(trajectory_msg, arm_name)

            result = FollowJointTrajectory.Result()
            if ok:
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                goal_handle.succeed()
            else:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                goal_handle.abort()

            return result

        return callback

    def _make_gripper_callback(self, arm_name: str):
        """Create a gripper command callback for a specific arm."""

        def callback(goal_handle):
            position = goal_handle.request.command.position
            logger.info(
                "Gripper command on %s: position=%.3f", arm_name, position,
            )

            with self._lock:
                if position > 0.01:
                    # Closing — attempt grasp
                    self._backend.grasp(arm_name, "")
                else:
                    # Opening — release
                    self._backend.release(arm_name)

            result = GripperCommand.Result()
            result.position = position
            result.reached_goal = True
            goal_handle.succeed()
            return result

        return callback

    def _publish_joint_states(self) -> None:
        """Publish current joint state from MuJoCo."""
        with self._lock:
            names, positions, velocities = self._backend.get_joint_state()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions.tolist()
        msg.velocity = velocities.tolist()
        self._joint_state_pub.publish(msg)

    def _publish_status(self) -> None:
        """Publish robot status (always True for mock)."""
        msg = Bool()
        msg.data = self._backend.is_running
        self._status_pub.publish(msg)
