# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ROS 2 action client for FollowJointTrajectory.

Wraps the standard ros2_control action interface for sending joint
trajectories to an arm controller.
"""

from __future__ import annotations

import logging

import rclpy.node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory

from mj_manipulator_ros.interfaces import follow_joint_trajectory_action, wait_for_future

logger = logging.getLogger(__name__)


class ArmTrajectoryClient:
    """Action client for sending trajectories to an arm controller."""

    def __init__(self, node: rclpy.node.Node, arm_name: str):
        self._node = node
        self._arm_name = arm_name
        self._action_name = follow_joint_trajectory_action(arm_name)

        self._client = ActionClient(
            node,
            FollowJointTrajectory,
            self._action_name,
        )

    def wait_for_server(self, timeout_sec: float = 5.0) -> bool:
        """Wait for the action server to become available."""
        return self._client.wait_for_server(timeout_sec)

    def send_trajectory(
        self,
        trajectory_msg: JointTrajectory,
        timeout_sec: float = 30.0,
    ) -> bool:
        """Send a trajectory and block until it completes.

        For single-arm callers with no cancellation needs. Synchronized
        multi-arm execution uses send_goal() directly instead --
        see HardwareContext._execute_single.
        """
        logger.info(
            "Sending trajectory to %s (%d points, %.2fs)",
            self._action_name,
            len(trajectory_msg.points),
            trajectory_msg.points[-1].time_from_start.sec + trajectory_msg.points[-1].time_from_start.nanosec * 1e-9
            if trajectory_msg.points else 0.0,
        )

        goal_handle = self.send_goal(trajectory_msg, timeout_sec=5.0)
        if goal_handle is None:
            return False

        result_future = goal_handle.get_result_async()
        wait_for_future(result_future, timeout_sec=timeout_sec)

        result = result_future.result()
        if result is None:
            logger.warning("Trajectory execution timed out on %s", self._action_name)
            return False

        error_code = result.result.error_code
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            logger.warning(
                "Trajectory execution failed on %s: error_code=%d",
                self._action_name,
                error_code,
            )
            return False

        return True

    def send_goal(self, trajectory_msg: JointTrajectory, timeout_sec: float = 5.0):
        """Send a trajectory and return its accepted goal handle without
        waiting for execution to finish.

        Returns None if the server rejects the goal or acceptance itself
        times out. The caller owns waiting on goal_handle.get_result_async()
        and may call goal_handle.cancel_goal_async() to abort mid-execution.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory_msg

        future = self._client.send_goal_async(goal)
        wait_for_future(future, timeout_sec=timeout_sec)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("Trajectory goal rejected by %s", self._action_name)
            return None

        return goal_handle
