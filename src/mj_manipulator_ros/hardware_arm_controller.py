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
        self._last_width: float | None = None

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

    def set_width(self, width: float, synchronous: bool = True) -> bool:
        """Set gripper width as a fraction open (0.0=closed, 1.0=open).

        Sends a GripperCommand to the real gripper, mapping the [0, 1]
        fraction onto this arm's [gripper_open, gripper_closed] raw travel
        range. When synchronous=True (the default), blocks until the
        action server reports the motion complete -- i.e. until the
        physical gripper has actually finished moving, not just until the
        command was accepted.

        Args:
            width: Desired width, clamped to [0.0, 1.0].
            synchronous: If True, block until the gripper finishes moving.
                If False, send the command and return immediately.
        """
        if self._gripper is None:
            logger.warning("No gripper configured for arm %s", self._config.name)
            return False

        width = max(0.0, min(1.0, width))
        open_pos = self._config.gripper_open
        closed_pos = self._config.gripper_closed
        position = open_pos + (1.0 - width) * (closed_pos - open_pos)

        ok = self._gripper.send_command(
            position=position,
            max_effort=50.0,
            synchronous=synchronous,
        )
        if ok:
            self._last_width = width
        return ok

    def get_width(self) -> float:
        """Last commanded gripper width as a fraction open.

        Real hardware has no position feedback wired here yet -- this
        reflects the last width successfully sent via set_width, defaulting
        to 1.0 (open) before any command has been sent.
        """
        return self._last_width if self._last_width is not None else 1.0
