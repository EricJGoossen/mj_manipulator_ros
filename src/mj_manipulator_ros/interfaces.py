"""ROS 2 interface names and conventions.

All interfaces use standard ROS 2 message types — no custom messages.
Topic and action names are parameterized by arm name for multi-arm support.
"""


def follow_joint_trajectory_action(arm_name: str) -> str:
    """FollowJointTrajectory action server name for an arm."""
    return f"/{arm_name}_controller/follow_joint_trajectory"


def gripper_command_action(arm_name: str) -> str:
    """GripperCommand action server name for an arm's gripper."""
    return f"/{arm_name}_gripper_controller/gripper_cmd"


def joint_commands_topic(arm_name: str) -> str:
    """Streaming joint command topic for an arm."""
    return f"/{arm_name}_controller/joint_commands"


JOINT_STATES_TOPIC = "/joint_states"
ROBOT_STATUS_TOPIC = "/robot_status"
