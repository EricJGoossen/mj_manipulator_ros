# mj_manipulator_ros

ROS 2 bridge for [`mj_manipulator`](https://github.com/personalrobotics/mj_manipulator): `HardwareContext` (same `ExecutionContext` protocol as `SimContext`), trajectory conversion, and a MuJoCo-backed mock node for integration tests.

## ROS 2 interface layout (bimanual / ros2_control)

`HardwareContext` uses the same naming convention as a typical **dual UR5e + Robotiq** `ros2_control` stack (per-arm namespaces `/left`, `/right`, tools under `/left_hand`, `/right_hand`). Defaults are defined in `mj_manipulator_ros.interfaces` and `ArmHardwareConfig`.

| Role | Topic / service / action (pattern) |
|------|-------------------------------------|
| Joint feedback (aggregated) | `/joint_states` (`sensor_msgs/JointState`) |
| Robot status | `/robot_status` (`std_msgs/Bool`) |
| Trajectory execution | `/{arm}/scaled_joint_trajectory_controller/follow_joint_trajectory` (`control_msgs/FollowJointTrajectory`) |
| Streaming joint positions | `/{arm}/forward_position_controller/commands` (`std_msgs/Float64MultiArray`) |
| Gripper | `/{arm}_hand/robotiq_gripper_controller/gripper_cmd` (`control_msgs/GripperCommand`) |
| Controller switch (for JTC vs forward position) | `/{arm}/controller_manager/switch_controller`, `/{arm}/controller_manager/list_controllers` |

`HardwareContext` keeps **forward position** active for streaming (`step` / `step_cartesian`), switches to the trajectory controller only for `execute()`, then switches back.

### Contract with the mock and tests

- **Joint names** in `HardwareConfig` / `ArmHardwareConfig` must match the names published on `/joint_states`.
- **Action and topic names** must match what the mock (or real stack) advertises — either use the defaults from `interfaces.py` or set explicit `follow_joint_trajectory_action`, `gripper_command_action`, and `forward_position_commands_topic` on each arm.
- Integration tests use a **unique** `/{arm}/` prefix (e.g. `left_<random>`) so they do not collide with another running `/left/...` stack on the same ROS domain.

## Tests

```bash
uv run pytest mj_manipulator_ros/tests/ -v
```

`test_hardware_context.py` requires ROS 2 (e.g. Ubuntu + `ros-humble-desktop`) and is skipped when `rclpy` is missing or on macOS.
