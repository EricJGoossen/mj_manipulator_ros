# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic ROS 2 bridge for mj_manipulator.

Provides HardwareContext (ExecutionContext implementation) that talks to
real robot hardware via standard ROS 2 interfaces (FollowJointTrajectory,
GripperCommand, JointState). Also includes a MuJoCo mock node for testing
without real hardware.

Requires system ROS 2 install (apt install ros-humble-desktop).
"""

try:
    import rclpy as _rclpy  # noqa: F401

    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

if HAS_ROS2:
    from mj_manipulator_ros.hardware_context import HardwareContext

    __all__ = ["HardwareContext", "HAS_ROS2"]
else:
    __all__ = ["HAS_ROS2"]
