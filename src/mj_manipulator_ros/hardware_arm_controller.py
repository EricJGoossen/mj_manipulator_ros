# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ArmController implementation for real hardware via ROS 2.

Implements the ArmController protocol (grasp/release) by sending
GripperCommand actions to the robot's gripper controller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mj_manipulator_ros.config import ArmHardwareConfig
    from mj_manipulator_ros.ros_gripper_client import GripperClient

logger = logging.getLogger(__name__)


class HardwareArmController:
    """ArmController for real hardware — grasp/release via GripperCommand."""

    def __init__(
        self,
        config: ArmHardwareConfig,
        gripper_client: GripperClient | None,
    ):
        self._config = config
        self._gripper = gripper_client

    def grasp(self, object_name: str, synchronous: bool = True) -> str | None:
        """Close gripper to grasp an object.

        Sends a GripperCommand to close position. On real hardware,
        the gripper's force feedback determines grasp success.

        Args:
            object_name: Name of the object to grasp.

        Returns:
            object_name if gripper closed successfully, None otherwise.
        """
        if self._gripper is None:
            logger.warning("No gripper configured for arm %s", self._config.name)
            return None

        logger.info("Grasping %s with %s arm", object_name, self._config.name)
        ok = self._gripper.send_command(
            position=self._config.gripper_closed,
            max_effort=50.0,
            synchronous=synchronous
        )

        if ok:
            logger.info("Grasp succeeded: %s", object_name)
            return object_name

        logger.warning("Grasp failed for %s", object_name)
        return None

    def release(self, object_name: str | None = None, synchronous: bool = True) -> None:
        """Open gripper to release held object(s).

        Args:
            object_name: Ignored on hardware (always fully opens).
        """
        if self._gripper is None:
            return

        logger.info("Releasing from %s arm", self._config.name)
        self._gripper.send_command(
            position=self._config.gripper_open,
            max_effort=50.0,
            synchronous=synchronous,
        )
