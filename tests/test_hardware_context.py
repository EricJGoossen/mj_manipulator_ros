# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration tests: HardwareContext ↔ mock node (Phase 2 style).

Requires ROS 2 (e.g. ``ros-humble-desktop`` on Ubuntu). Skipped when ``rclpy`` is
unavailable or on macOS (no system ROS in typical setups).

The mock must expose the same action/topic/service names as
:class:`~mj_manipulator_ros.hardware_context.HardwareContext` expects — see
``mj_manipulator_ros.interfaces`` (bimanual-style paths). Tests use an explicit
:class:`~mj_manipulator_ros.config.HardwareConfig` whose joint names match the
kinematic mock model.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid

import numpy as np
import pytest

from tests.conftest import HAS_ROS2

if not HAS_ROS2:
    pytest.skip("ROS 2 not installed", allow_module_level=True)

import mujoco
import rclpy
from rclpy.executors import MultiThreadedExecutor

pytestmark = [
    pytest.mark.skipif(sys.platform == "darwin", reason="ROS 2 hardware tests target Linux"),
]

from mj_manipulator.trajectory import Trajectory

from mj_manipulator_ros.config import ArmHardwareConfig, HardwareConfig
from mj_manipulator_ros.hardware_context import HardwareContext
from mj_manipulator_ros.mock.kinematic_backend import KinematicMockBackend
from mj_manipulator_ros.mock.mock_node import MockRobotNode


def _fixture_xml_path() -> str:
    import os

    here = os.path.dirname(__file__)
    return os.path.join(here, "fixtures", "minimal_mock_arm.xml")


@pytest.fixture
def ros_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    # Do not shutdown here — other tests in the same session may need rclpy.


def test_hardware_context_trajectory_and_streaming(ros_init):
    """Connect, execute a trajectory, then stream via ``step`` / ``step_cartesian``."""
    # Unique arm namespace (avoid another /left/... on the machine). Must not contain the
    # substring "_arm" except as the standard "left_arm" suffix — HardwareContext strips
    # "_arm" when mapping trajectory entity → arm client name.
    arm_name = f"left_{uuid.uuid4().hex[:8]}"

    model = mujoco.MjModel.from_xml_path(_fixture_xml_path())
    data = mujoco.MjData(model)
    joint_names = ["left_j0", "left_j1"]
    backend = KinematicMockBackend(
        model,
        data,
        joint_names_by_arm={arm_name: joint_names},
    )
    backend.start()

    node = MockRobotNode(
        backend,
        [arm_name],
        publish_rate=200.0,
        node_name=f"mock_robot_{uuid.uuid4().hex[:8]}",
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.5)

    cfg = HardwareConfig(
        arms=[
            ArmHardwareConfig(
                name=arm_name,
                joint_names=joint_names,
                has_gripper=False,
            )
        ],
        control_dt=0.02,
        action_timeout=30.0,
    )

    try:
        with HardwareContext(config=cfg, node_name=f"hw_ctx_{uuid.uuid4().hex[:8]}") as ctx:
            q0 = ctx.get_joint_positions(joint_names)
            assert q0 is not None
            assert q0.shape == (2,)

            traj = Trajectory(
                timestamps=np.array([0.0, 0.2, 0.4]),
                positions=np.array(
                    [
                        [0.0, 0.0],
                        [0.5, -0.25],
                        [0.5, -0.25],
                    ]
                ),
                velocities=np.zeros((3, 2)),
                accelerations=np.zeros((3, 2)),
                entity=arm_name,
                joint_names=joint_names,
            )
            assert ctx.execute(traj) is True

            q1 = ctx.get_joint_positions(joint_names)
            assert q1 is not None
            np.testing.assert_allclose(q1, [0.5, -0.25], atol=1e-3)

            ctx.step({arm_name: np.array([0.1, 0.2])})
            time.sleep(0.05)
            q2 = ctx.get_joint_positions(joint_names)
            assert q2 is not None
            np.testing.assert_allclose(q2, [0.1, 0.2], atol=1e-2)

            ctx.step_cartesian(arm_name, np.array([-0.2, 0.3]))
            time.sleep(0.05)
            q3 = ctx.get_joint_positions(joint_names)
            assert q3 is not None
            np.testing.assert_allclose(q3, [-0.2, 0.3], atol=1e-2)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
