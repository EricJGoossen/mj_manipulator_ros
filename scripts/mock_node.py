#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Entry point for the MuJoCo mock robot node.

Usage:
    pixi run python scripts/mock_node.py --model path/to/scene.xml --arms left right
"""

from __future__ import annotations

import argparse
import logging

import mujoco
import rclpy

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo mock robot node")
    parser.add_argument("--model", required=True, help="Path to MuJoCo XML model")
    parser.add_argument("--arms", nargs="+", required=True, help="Arm names")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--rate", type=float, default=500.0, help="Joint state pub rate")
    args = parser.parse_args()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # Import here to avoid importing mj_manipulator at module level

    # Build arms from model (simplified — real usage would pass proper configs)
    arms = {}
    for arm_name in args.arms:
        # This is a minimal example — real setup would use proper joint configs
        print(
            f"Note: arm '{arm_name}' setup requires proper ArmConfig. "
            f"Use geodude_hardware for Geodude-specific configuration."
        )

    from mj_manipulator_ros.mock.mock_node import MockRobotNode
    from mj_manipulator_ros.mock.mujoco_backend import MuJoCoBackend

    backend = MuJoCoBackend(
        model,
        data,
        arms,
        physics=True,
        show_viewer=args.viewer,
    )
    backend.start()

    rclpy.init()
    try:
        node = MockRobotNode(backend, args.arms, publish_rate=args.rate)
        rclpy.spin(node)
    finally:
        backend.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
