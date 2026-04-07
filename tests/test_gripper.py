# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Gripper and streaming integration tests: HardwareContext ↔ mock node.

Uses a unique ``left_<uuid>`` arm name so tests do not collide with another
``/left/...`` stack on the host. On-robot configs use ``name="left"``; behavior
is the same.

Requires ROS 2 (Linux). Skipped when ``rclpy`` is unavailable or on macOS.
"""

from __future__ import annotations

import os
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

from mj_manipulator_ros.config import ArmHardwareConfig, HardwareConfig
from mj_manipulator_ros.hardware_context import HardwareContext
from mj_manipulator_ros.mock.kinematic_backend import KinematicMockBackend
from mj_manipulator_ros.mock.mock_node import MockRobotNode


def _fixture_xml_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "fixtures", "minimal_mock_arm.xml")


@pytest.fixture
def ros_init():
    if not rclpy.ok():
        rclpy.init()
    yield


def test_gripper_grasp_release_round_trip(ros_init):
    """``HardwareArmController.grasp`` / ``release`` via GripperCommand to mock backend."""
    arm_name = f"left_{uuid.uuid4().hex[:8]}"
    joint_names = ["left_j0", "left_j1"]

    model = mujoco.MjModel.from_xml_path(_fixture_xml_path())
    data = mujoco.MjData(model)
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
                has_gripper=True,
            )
        ],
        control_dt=0.02,
        action_timeout=30.0,
    )

    try:
        with HardwareContext(config=cfg, node_name=f"hw_ctx_{uuid.uuid4().hex[:8]}") as ctx:
            # GripperCommand carries position only; object id is returned by HardwareArmController
            # when the action succeeds (same as real hardware).
            assert ctx.arm(arm_name).grasp("can_0") == "can_0"
            assert backend.last_grasp is not None
            assert backend.last_grasp[0] == arm_name
            # Mock passes "" to backend (no object field on standard GripperCommand).
            assert backend.last_grasp[1] == ""

            ctx.arm(arm_name).release()
            assert backend.last_release is not None
            assert backend.last_release[0] == arm_name
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()


def test_streaming_step_and_cartesian(ros_init):
    """``step`` / ``step_cartesian`` publish to forward_position_controller/commands; mock updates state."""
    arm_name = f"left_{uuid.uuid4().hex[:8]}"
    joint_names = ["left_j0", "left_j1"]

    model = mujoco.MjModel.from_xml_path(_fixture_xml_path())
    data = mujoco.MjData(model)
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
            ctx.step({arm_name: np.array([0.15, -0.1])})
            time.sleep(0.06)
            q = ctx.get_joint_positions(joint_names)
            assert q is not None
            np.testing.assert_allclose(q, [0.15, -0.1], atol=1e-2)

            ctx.step_cartesian(arm_name, np.array([0.2, 0.1]))
            time.sleep(0.06)
            q2 = ctx.get_joint_positions(joint_names)
            assert q2 is not None
            np.testing.assert_allclose(q2, [0.2, 0.1], atol=1e-2)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
