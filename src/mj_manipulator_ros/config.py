# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Configuration for HardwareContext."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArmHardwareConfig:
    """Hardware configuration for a single arm."""

    name: str
    joint_names: list[str]
    has_gripper: bool = True
    gripper_open: float = 0.0
    gripper_closed: float = 0.255  # Robotiq 2F-140 max travel (meters)


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
