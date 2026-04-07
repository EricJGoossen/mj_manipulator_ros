# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ROS 2 interface names and conventions.

Matches a typical **bimanual UR5e + ros2_control** stack (per-arm namespaces
``/left``, ``/right``, tool chain ``/left_hand``, ``/right_hand``), e.g.:

- ``/left/scaled_joint_trajectory_controller/...``
- ``/left/forward_position_controller/commands``
- ``/left_hand/robotiq_gripper_controller/...``

All interfaces use standard ROS 2 message types — no custom messages.
"""

# ---------------------------------------------------------------------------
# Global topics (aggregated robot state)
# ---------------------------------------------------------------------------

JOINT_STATES_TOPIC = "/joint_states"
ROBOT_STATUS_TOPIC = "/robot_status"

# ---------------------------------------------------------------------------
# Per-arm UR controller names (ros2_control)
# ---------------------------------------------------------------------------

SCALED_JOINT_TRAJECTORY_CONTROLLER = "scaled_joint_trajectory_controller"
JOINT_TRAJECTORY_CONTROLLER = "joint_trajectory_controller"
FORWARD_POSITION_CONTROLLER = "forward_position_controller"
FORWARD_VELOCITY_CONTROLLER = "forward_velocity_controller"

# ---------------------------------------------------------------------------
# Per-hand (tool) gripper controller
# ---------------------------------------------------------------------------

ROBOTIQ_GRIPPER_CONTROLLER = "robotiq_gripper_controller"


def follow_joint_trajectory_action(
    arm_name: str,
    *,
    trajectory_controller: str = SCALED_JOINT_TRAJECTORY_CONTROLLER,
) -> str:
    """FollowJointTrajectory action for an arm (ros2_control).

    Example: ``/left/scaled_joint_trajectory_controller/follow_joint_trajectory``
    """
    return f"/{arm_name}/{trajectory_controller}/follow_joint_trajectory"


def gripper_command_action(arm_name: str) -> str:
    """GripperCommand action for an arm's Robotiq gripper chain.

    Example: ``/left_hand/robotiq_gripper_controller/gripper_cmd``
    """
    return f"/{arm_name}_hand/{ROBOTIQ_GRIPPER_CONTROLLER}/gripper_cmd"


def forward_position_commands_topic(arm_name: str) -> str:
    """Streaming joint **position** commands (Float64MultiArray).

    Example: ``/left/forward_position_controller/commands``
    """
    return f"/{arm_name}/{FORWARD_POSITION_CONTROLLER}/commands"


def forward_velocity_commands_topic(arm_name: str) -> str:
    """Streaming joint **velocity** commands (Float64MultiArray).

    Example: ``/left/forward_velocity_controller/commands``
    """
    return f"/{arm_name}/{FORWARD_VELOCITY_CONTROLLER}/commands"


def arm_joint_states_topic(arm_name: str) -> str:
    """Per-arm joint states (in addition to aggregated ``/joint_states``).

    Example: ``/left/joint_states``
    """
    return f"/{arm_name}/joint_states"


def hand_joint_states_topic(arm_name: str) -> str:
    """Per-hand (gripper) joint states.

    Example: ``/left_hand/joint_states``
    """
    return f"/{arm_name}_hand/joint_states"


def joint_commands_topic(arm_name: str) -> str:
    """Deprecated alias: use :func:`forward_position_commands_topic`.

    Historically pointed at a generic ``joint_commands`` topic; the bimanual
    stack exposes position streaming on ``forward_position_controller/commands``.
    """
    return forward_position_commands_topic(arm_name)
