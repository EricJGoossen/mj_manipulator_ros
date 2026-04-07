# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Bimanual UR5e + Robotiq SSoT parsing (Geodude / ``bimanual`` layout).

This is **robot-specific**: joint naming and YAML keys match the Personal
Robotics bimanual stack, not a generic mj_manipulator robot. :class:`HardwareContext`
can consume an SSoT path and delegates building :class:`HardwareConfig` here.
"""

from __future__ import annotations

import logging
from typing import Any

from mj_manipulator_ros.config import ArmHardwareConfig, HardwareConfig
from mj_manipulator_ros.interfaces import (
    FORWARD_POSITION_CONTROLLER,
    SCALED_JOINT_TRAJECTORY_CONTROLLER,
    follow_joint_trajectory_action,
    forward_position_commands_topic,
    gripper_command_action,
)

logger = logging.getLogger(__name__)

# UR5e serial-chain order; ROS joint names are ``{side}_{suffix}``.
# TODO: there needs to be a mapping from the mujoco joint names to the ros joint names
UR5E_BIMANUAL_JOINT_SUFFIXES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def named_poses_from_bimanual_ssot(
    ssot: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    """``home`` / ``ready`` joint vectors from ``arms.*.home_joint_positions``."""
    poses: dict[str, dict[str, list[float]]] = {}
    for side, arm_cfg in ssot.get("arms", {}).items():
        if not arm_cfg.get("enabled", False):
            continue
        home_joints = arm_cfg.get("home_joint_positions")
        if not home_joints:
            continue
        ordered: list[float] = []
        for suffix in UR5E_BIMANUAL_JOINT_SUFFIXES:
            ros_name = f"{side}_{suffix}"
            if ros_name not in home_joints:
                logger.warning("Missing %s in SSoT home_joint_positions", ros_name)
                ordered.append(0.0)
            else:
                ordered.append(float(home_joints[ros_name]))
        poses.setdefault("home", {})[side] = ordered
        poses.setdefault("ready", {})[side] = ordered
    return poses


def hardware_config_from_bimanual_ssot(ssot: dict[str, Any]) -> HardwareConfig:
    """Build :class:`HardwareConfig` from a loaded bimanual SSoT dict."""
    arms: list[ArmHardwareConfig] = []
    hands = ssot.get("hands", {})

    for side, arm_cfg in ssot.get("arms", {}).items():
        if not arm_cfg.get("enabled", False):
            continue
        jtc = arm_cfg.get(
            "initial_joint_controller",
            SCALED_JOINT_TRAJECTORY_CONTROLLER,
        )
        joint_names = [f"{side}_{suf}" for suf in UR5E_BIMANUAL_JOINT_SUFFIXES]
        fjt = follow_joint_trajectory_action(side, trajectory_controller=jtc)
        hand = hands.get(side, {})
        has_gripper = bool(hand.get("enabled", False))
        gripper_closed = float(hand.get("max_gripper_position", 0.255))
        grip_act = gripper_command_action(side)
        stream_topic = forward_position_commands_topic(side)

        arms.append(
            ArmHardwareConfig(
                name=side,
                joint_names=joint_names,
                has_gripper=has_gripper,
                gripper_open=0.0,
                gripper_closed=gripper_closed,
                joint_trajectory_controller=jtc,
                follow_joint_trajectory_action=fjt,
                gripper_command_action=grip_act if has_gripper else None,
                forward_position_commands_topic=stream_topic,
                forward_position_controller=FORWARD_POSITION_CONTROLLER,
            )
        )

    if not arms:
        raise ValueError("SSoT has no enabled arms")

    return HardwareConfig(arms=arms)
