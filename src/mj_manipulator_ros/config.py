# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Configuration for HardwareContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ArmHardwareConfig:
    """Hardware configuration for a single arm."""

    name: str
    joint_names: list[str]
    has_gripper: bool = True
    gripper_open: float = 0.0
    gripper_closed: float = 0.255  # Robotiq 2F-140 max travel (meters)
    # ros2_control controller that owns this arm's FollowJointTrajectory
    # action server. Defaults to the "{name}_controller" convention
    joint_trajectory_controller: str | None = None
    # Action interface the gripper controller actually serves. Grippers
    # driven by a dedicated gripper_action_controller (e.g. a Robotiq
    # gripper) use "gripper_command" (control_msgs/GripperCommand); grippers
    # driven by a plain joint_trajectory_controller (e.g. OpenArm's
    # parallel-jaw gripper) use "follow_joint_trajectory" instead, which
    # requires gripper_joint_name to be set.
    gripper_interface: Literal["gripper_command", "follow_joint_trajectory"] = "gripper_command"
    # Required when gripper_interface == "follow_joint_trajectory" -- the
    # single joint name the gripper controller's trajectory goal targets.
    gripper_joint_name: str | None = None

    def __post_init__(self) -> None:
        if self.joint_trajectory_controller is None:
            self.joint_trajectory_controller = f"{self.name}_controller"
        if self.gripper_interface == "follow_joint_trajectory" and self.gripper_joint_name is None:
            raise ValueError(
                "gripper_joint_name is required when gripper_interface == 'follow_joint_trajectory'"
            )


@dataclass
class HardwareConfig:
    """Configuration for HardwareContext.

    Parameterizes the ROS 2 interface — arm names, joint names, control rate.
    Robot-specific packages provide concrete configs (e.g. Geodude's two UR5e arms).
    """

    arms: list[ArmHardwareConfig] = field(default_factory=list)
    control_dt: float = 0.002  # 500 Hz (UR RTDE default)
    action_timeout: float = 30.0  # Seconds to wait for trajectory execution
    speed_scale: float = 1.0  # Velocity scaling factor (0.0-1.0)
