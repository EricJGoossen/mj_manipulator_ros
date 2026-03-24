#!/usr/bin/env python3
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
    from mj_manipulator import Arm
    from mj_manipulator.config import ArmConfig, KinematicLimits
    from mj_manipulator.arms.ur5e import UR5E_VELOCITY_LIMITS, UR5E_ACCELERATION_LIMITS

    # Build arms from model (simplified — real usage would pass proper configs)
    arms = {}
    for arm_name in args.arms:
        # This is a minimal example — real setup would use proper joint configs
        print(f"Note: arm '{arm_name}' setup requires proper ArmConfig. "
              f"Use geodude_hardware for Geodude-specific configuration.")

    from mj_manipulator_ros.mock.mujoco_backend import MuJoCoBackend
    from mj_manipulator_ros.mock.mock_node import MockRobotNode

    backend = MuJoCoBackend(
        model, data, arms,
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
