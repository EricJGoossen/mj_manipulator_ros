# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for Trajectory <-> JointTrajectory conversion.

These tests verify the data bridge between mj_manipulator and ROS 2
without requiring a running ROS node. Skipped on macOS (no system ROS 2).
"""

from __future__ import annotations

import numpy as np
import pytest
from mj_manipulator.trajectory import Trajectory

from tests.conftest import HAS_ROS2

if not HAS_ROS2:
    pytest.skip("ROS 2 not installed", allow_module_level=True)

from mj_manipulator_ros.trajectory_convert import msg_to_trajectory, trajectory_to_msg


def _make_trajectory(n_waypoints: int = 10, dof: int = 6) -> Trajectory:
    """Create a simple test trajectory."""
    timestamps = np.linspace(0.0, 1.0, n_waypoints)
    positions = np.random.randn(n_waypoints, dof)
    velocities = np.random.randn(n_waypoints, dof)
    accelerations = np.random.randn(n_waypoints, dof)
    joint_names = [f"joint_{i}" for i in range(dof)]
    return Trajectory(
        timestamps=timestamps,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        entity="left",
        joint_names=joint_names,
    )


class TestTrajectoryToMsg:
    def test_basic_conversion(self):
        traj = _make_trajectory()
        msg = trajectory_to_msg(traj)

        assert len(msg.points) == traj.num_waypoints
        assert msg.joint_names == traj.joint_names

    def test_positions_preserved(self):
        traj = _make_trajectory(n_waypoints=5, dof=3)
        msg = trajectory_to_msg(traj)

        for i in range(traj.num_waypoints):
            np.testing.assert_allclose(
                msg.points[i].positions,
                traj.positions[i],
                atol=1e-10,
            )

    def test_timestamps_preserved(self):
        traj = _make_trajectory(n_waypoints=5)
        msg = trajectory_to_msg(traj)

        for i in range(traj.num_waypoints):
            point = msg.points[i]
            t_msg = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            np.testing.assert_allclose(t_msg, traj.timestamps[i], atol=1e-6)

    def test_no_joint_names_raises(self):
        traj = _make_trajectory()
        traj.joint_names = None
        with pytest.raises(ValueError, match="joint_names"):
            trajectory_to_msg(traj)


class TestMsgToTrajectory:
    def test_roundtrip(self):
        original = _make_trajectory(n_waypoints=8, dof=6)
        msg = trajectory_to_msg(original)
        recovered = msg_to_trajectory(msg, entity="left")

        np.testing.assert_allclose(
            recovered.positions,
            original.positions,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            recovered.velocities,
            original.velocities,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            recovered.timestamps,
            original.timestamps,
            atol=1e-6,
        )
        assert recovered.entity == "left"
        assert recovered.joint_names == original.joint_names

    def test_entity_preserved(self):
        traj = _make_trajectory()
        msg = trajectory_to_msg(traj)
        recovered = msg_to_trajectory(msg, entity="right_arm")
        assert recovered.entity == "right_arm"
