"""Shared fixtures and markers for mj_manipulator_ros tests."""

import pytest

try:
    import rclpy  # noqa: F401

    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

requires_ros2 = pytest.mark.skipif(
    not HAS_ROS2, reason="ROS 2 not installed (requires system ros-humble)",
)
