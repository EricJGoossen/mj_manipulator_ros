# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ROS 2 action client for GripperCommand.

Wraps the standard gripper action interface for grasp/release operations.
"""

from __future__ import annotations

import logging

import rclpy.node
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from mj_manipulator_ros.interfaces import (
    follow_joint_trajectory_action_for_controller,
    gripper_command_action,
    wait_for_future,
)

logger = logging.getLogger(__name__)


class GripperClient:
    """Action client for controlling a gripper."""

    def __init__(self, node: rclpy.node.Node, arm_name: str):
        self._node = node
        self._arm_name = arm_name
        self._action_name = gripper_command_action(arm_name)

        self._client = ActionClient(
            node,
            GripperCommand,
            self._action_name,
        )

    def wait_for_server(self, timeout_sec: float = 5.0) -> bool:
        """Wait for the action server to become available."""
        return self._client.wait_for_server(timeout_sec)

    def send_command(
        self,
        position: float,
        max_effort: float = 50.0,
        timeout_sec: float = 10.0,
        synchronous: bool = True,
    ) -> bool:
        """Send a gripper command and wait for completion.

        Args:
            position: Target gripper position (0.0=open, max=closed).
            max_effort: Maximum effort/force in Newtons.
            timeout_sec: Maximum time to wait.

        Returns:
            True if command completed successfully.
        """
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        future = self._client.send_goal_async(goal)
        wait_for_future(future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("Gripper command rejected by %s", self._action_name)
            return False

        result_future = goal_handle.get_result_async()
    
        if not synchronous:
            result_future.add_done_callback(
                lambda f: logger.info(
                    "Gripper command on %s finished (async): %s",
                    self._action_name, "ok" if f.result() is not None else "failed/timed out"))
            return True

        wait_for_future(result_future, timeout_sec=timeout_sec)
        if result_future.result() is None:
            logger.warning("Gripper command timed out on %s", self._action_name)
            return False

        return True


class TrajectoryGripperClient:
    """FollowJointTrajectory-based gripper client.

    For a gripper driven by a plain joint_trajectory_controller (single
    joint) rather than a dedicated GripperCommand action server -- e.g.
    OpenArm's parallel-jaw gripper, as opposed to the Robotiq gripper
    GripperClient above was built for. Implements the same
    wait_for_server()/send_command() interface so HardwareContext can use
    either client interchangeably based on ArmHardwareConfig.gripper_interface.
    """

    def __init__(self, node: rclpy.node.Node, arm_name: str, joint_name: str):
        self._node = node
        self._arm_name = arm_name
        self._joint_name = joint_name
        self._action_name = follow_joint_trajectory_action_for_controller(f"{arm_name}_gripper_controller")

        self._client = ActionClient(
            node,
            FollowJointTrajectory,
            self._action_name,
        )

    def wait_for_server(self, timeout_sec: float = 5.0) -> bool:
        """Wait for the action server to become available."""
        return self._client.wait_for_server(timeout_sec)

    def send_command(
        self,
        position: float,
        max_effort: float = 50.0,
        timeout_sec: float = 10.0,
        synchronous: bool = True,
        move_duration_s: float = 1.0,
    ) -> bool:
        """Send a single-point trajectory moving the gripper joint to `position`.

        Args:
            position: Target gripper joint position.
            max_effort: Accepted for interface parity with GripperClient but
                unused -- FollowJointTrajectory has no effort field. Stall/
                effort handling belongs to the controller's own parameters
                (see allow_stalling in the controller config).
            timeout_sec: Maximum time to wait for the move to complete.
            move_duration_s: Nominal duration commanded for the single
                trajectory point.

        Returns:
            True if the move completed successfully.
        """
        secs = int(move_duration_s)
        nsecs = int((move_duration_s - secs) * 1e9)
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)

        trajectory = JointTrajectory()
        trajectory.joint_names = [self._joint_name]
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        future = self._client.send_goal_async(goal)
        wait_for_future(future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("Gripper trajectory goal rejected by %s", self._action_name)
            return False

        result_future = goal_handle.get_result_async()

        if not synchronous:
            result_future.add_done_callback(
                lambda f: logger.info(
                    "Gripper trajectory on %s finished (async): %s",
                    self._action_name, "ok" if f.result() is not None else "failed/timed out"))
            return True

        wait_for_future(result_future, timeout_sec=timeout_sec)
        result = result_future.result()
        if result is None:
            logger.warning("Gripper trajectory timed out on %s", self._action_name)
            return False

        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            logger.warning(
                "Gripper trajectory failed on %s: error_code=%d",
                self._action_name, result.result.error_code,
            )
            return False

        return True
