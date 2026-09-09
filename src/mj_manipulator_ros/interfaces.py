# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import threading

"""ROS 2 interface names and conventions.

All interfaces use standard ROS 2 message types — no custom messages.
Topic and action names are parameterized by arm name for multi-arm support.
"""

def wait_for_future(future, timeout_sec: float):
    """Block until 'future' completes, without spinning ourselfs"""
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())

    if not event.wait(timeout_sec):
        raise TimeoutError(f"Timed out after {timeout_sec}s waiting for future")
    
    return future.result()


def follow_joint_trajectory_action(arm_name: str) -> str:
    """FollowJointTrajectory action server name for an arm."""
    return follow_joint_trajectory_action_for_controller(f"{arm_name}_controller")


def follow_joint_trajectory_action_for_controller(controller_name: str) -> str:
    """FollowJointTrajectory action server name for a given controller."""
    return f"/{controller_name}/follow_joint_trajectory"


def gripper_command_action(arm_name: str) -> str:
    """GripperCommand action server name for an arm's gripper."""
    return f"/{arm_name}_gripper_controller/gripper_cmd"


def joint_commands_topic(arm_name: str) -> str:
    """Streaming joint command topic for an arm."""
    return f"/{arm_name}_controller/joint_commands"


JOINT_STATES_TOPIC = "/joint_states"
ROBOT_STATUS_TOPIC = "/robot_status"
