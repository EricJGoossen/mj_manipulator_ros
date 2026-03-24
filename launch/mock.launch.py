"""Launch file for the MuJoCo mock robot node.

Usage:
    ros2 launch mj_manipulator_ros mock.launch.py model_path:=path/to/model.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_path", description="Path to MuJoCo XML model"),
        DeclareLaunchArgument("arms", default_value="left right", description="Arm names"),
        DeclareLaunchArgument("viewer", default_value="false", description="Show viewer"),
        DeclareLaunchArgument("rate", default_value="500.0", description="Joint state rate"),

        Node(
            package="mj_manipulator_ros",
            executable="mock_node",
            name="mock_robot",
            parameters=[{
                "model_path": LaunchConfiguration("model_path"),
                "arms": LaunchConfiguration("arms"),
                "viewer": LaunchConfiguration("viewer"),
                "rate": LaunchConfiguration("rate"),
            }],
            output="screen",
        ),
    ])
