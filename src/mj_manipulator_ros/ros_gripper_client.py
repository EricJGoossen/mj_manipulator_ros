# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ROS 2 action client for GripperCommand.

Wraps the standard gripper action interface for grasp/release operations.
"""

from __future__ import annotations

import logging

import rclpy.node
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

from mj_manipulator_ros.interfaces import gripper_command_action

logger = logging.getLogger(__name__)


class GripperClient:
    """Action client for controlling a gripper."""

    def __init__(
        self,
        node: rclpy.node.Node,
        arm_name: str,
        *,
        action_name: str | None = None,
    ):
        self._node = node
        self._arm_name = arm_name
        self._action_name = action_name or gripper_command_action(arm_name)

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
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            logger.warning("Gripper command rejected by %s", self._action_name)
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self._node,
            result_future,
            timeout_sec=timeout_sec,
        )

        result = result_future.result()
        if result is None:
            logger.warning("Gripper command timed out on %s", self._action_name)
            return False

        return True
