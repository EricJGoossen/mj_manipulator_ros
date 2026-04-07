# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo mock ROS 2 node.

Serves the same ROS 2 interfaces as real hardware — FollowJointTrajectory (bimanual
``scaled_joint_trajectory_controller`` path), GripperCommand, ``/joint_states``,
``controller_manager`` list/switch services, and ``forward_position_controller/commands``
streaming when the backend implements
:meth:`~mj_manipulator_ros.mock.kinematic_backend.KinematicMockBackend.apply_forward_position`.
From HardwareContext's perspective, this is indistinguishable from a real robot when
names match :mod:`mj_manipulator_ros.interfaces`.

Usage:
    ros2 run mj_manipulator_ros mock_node --ros-args -p model_path:=path/to/model.xml
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import rclpy.node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer

try:
    from rclpy.callback_group import ReentrantCallbackGroup
except ImportError:
    # Older RoboStack builds don't have callback_group as a separate module
    ReentrantCallbackGroup = None
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from mj_manipulator_ros.interfaces import (
    FORWARD_POSITION_CONTROLLER,
    JOINT_STATES_TOPIC,
    ROBOT_STATUS_TOPIC,
    SCALED_JOINT_TRAJECTORY_CONTROLLER,
    follow_joint_trajectory_action,
    forward_position_commands_topic,
    gripper_command_action,
)

if TYPE_CHECKING:
    from mj_manipulator_ros.mock.kinematic_backend import KinematicMockBackend
    from mj_manipulator_ros.mock.mujoco_backend import MuJoCoBackend

logger = logging.getLogger(__name__)


class MockRobotNode(rclpy.node.Node):
    """ROS 2 node that serves robot interfaces backed by MuJoCo.

    Args:
        backend: MuJoCoBackend that drives the simulation.
        arm_names: List of arm names to create action servers for.
        publish_rate: Rate in Hz for publishing /joint_states.
    """

    def __init__(
        self,
        backend: MuJoCoBackend | KinematicMockBackend,
        arm_names: list[str],
        publish_rate: float = 500.0,
        *,
        node_name: str = "mock_robot",
    ):
        super().__init__(node_name)
        self._backend = backend
        self._lock = threading.Lock()
        self._cb_group = ReentrantCallbackGroup() if ReentrantCallbackGroup is not None else None

        # Joint state publisher
        self._joint_state_pub = self.create_publisher(
            JointState,
            JOINT_STATES_TOPIC,
            10,
        )
        self._timer = self.create_timer(
            1.0 / publish_rate,
            self._publish_joint_states,
        )

        # Robot status publisher (always True for mock)
        self._status_pub = self.create_publisher(
            Bool,
            ROBOT_STATUS_TOPIC,
            10,
        )
        self._status_timer = self.create_timer(1.0, self._publish_status)

        # Per-arm action servers
        self._traj_servers: dict[str, ActionServer] = {}
        self._gripper_servers: dict[str, ActionServer] = {}
        # Mirrors ros2_control so HardwareContext can switch JTC <-> forward position.
        self._active_controller: dict[str, str] = {
            n: FORWARD_POSITION_CONTROLLER for n in arm_names
        }
        self._fpc_subs: list = []

        for arm_name in arm_names:
            # FollowJointTrajectory
            self._traj_servers[arm_name] = ActionServer(
                self,
                FollowJointTrajectory,
                follow_joint_trajectory_action(arm_name),
                execute_callback=self._make_traj_callback(arm_name),
                callback_group=self._cb_group,
            )

            # GripperCommand
            self._gripper_servers[arm_name] = ActionServer(
                self,
                GripperCommand,
                gripper_command_action(arm_name),
                execute_callback=self._make_gripper_callback(arm_name),
                callback_group=self._cb_group,
            )

            self.create_service(
                ListControllers,
                f"/{arm_name}/controller_manager/list_controllers",
                self._make_list_controllers_cb(arm_name),
            )
            self.create_service(
                SwitchController,
                f"/{arm_name}/controller_manager/switch_controller",
                self._make_switch_controller_cb(arm_name),
            )

            if hasattr(backend, "apply_forward_position"):
                topic = forward_position_commands_topic(arm_name)
                self._fpc_subs.append(
                    self.create_subscription(
                        Float64MultiArray,
                        topic,
                        self._make_fpc_callback(arm_name),
                        10,
                    ),
                )
                logger.info("Subscribed to streaming commands: %s", topic)

        logger.info("MockRobotNode ready with arms: %s", arm_names)

    def _make_traj_callback(self, arm_name: str):
        """Create a trajectory execution callback for a specific arm."""

        def callback(goal_handle):
            trajectory_msg = goal_handle.request.trajectory
            logger.info(
                "Executing trajectory on %s (%d points)",
                arm_name,
                len(trajectory_msg.points),
            )

            with self._lock:
                ok = self._backend.execute_trajectory(trajectory_msg, arm_name)

            result = FollowJointTrajectory.Result()
            if ok:
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                goal_handle.succeed()
            else:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                goal_handle.abort()

            return result

        return callback

    def _make_gripper_callback(self, arm_name: str):
        """Create a gripper command callback for a specific arm."""

        def callback(goal_handle):
            position = goal_handle.request.command.position
            logger.info(
                "Gripper command on %s: position=%.3f",
                arm_name,
                position,
            )

            with self._lock:
                if position > 0.01:
                    # Closing — attempt grasp
                    self._backend.grasp(arm_name, "")
                else:
                    # Opening — release
                    self._backend.release(arm_name)

            result = GripperCommand.Result()
            result.position = position
            result.reached_goal = True
            goal_handle.succeed()
            return result

        return callback

    def _make_list_controllers_cb(self, arm_name: str):
        def cb(_request: ListControllers.Request, response: ListControllers.Response):
            jtc = ControllerState()
            jtc.name = SCALED_JOINT_TRAJECTORY_CONTROLLER
            jtc.state = (
                "active"
                if self._active_controller.get(arm_name) == SCALED_JOINT_TRAJECTORY_CONTROLLER
                else "inactive"
            )
            jtc.type = "joint_trajectory_controller/JointTrajectoryController"

            fpc = ControllerState()
            fpc.name = FORWARD_POSITION_CONTROLLER
            fpc.state = (
                "active"
                if self._active_controller.get(arm_name) == FORWARD_POSITION_CONTROLLER
                else "inactive"
            )
            fpc.type = "position_controllers/ForwardCommandController"

            response.controller = [jtc, fpc]
            return response

        return cb

    def _make_switch_controller_cb(self, arm_name: str):
        def cb(request: SwitchController.Request, response: SwitchController.Response):
            activate = list(request.activate_controllers)
            deactivate = list(request.deactivate_controllers)
            logger.info(
                "Mock switch_controller %s: +%s -%s",
                arm_name,
                activate,
                deactivate,
            )
            if activate:
                # HardwareContext activates one trajectory or one forward controller at a time.
                name = activate[0]
                if name in (SCALED_JOINT_TRAJECTORY_CONTROLLER, FORWARD_POSITION_CONTROLLER):
                    self._active_controller[arm_name] = name
            response.ok = True
            return response

        return cb

    def _make_fpc_callback(self, arm_name: str):
        def cb(msg: Float64MultiArray):
            arr = np.asarray(msg.data, dtype=float)
            with self._lock:
                self._backend.apply_forward_position(arm_name, arr)

        return cb

    def _publish_joint_states(self) -> None:
        """Publish current joint state from MuJoCo."""
        with self._lock:
            names, positions, velocities = self._backend.get_joint_state()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions.tolist()
        msg.velocity = velocities.tolist()
        self._joint_state_pub.publish(msg)

    def _publish_status(self) -> None:
        """Publish robot status (always True for mock)."""
        msg = Bool()
        msg.data = self._backend.is_running
        self._status_pub.publish(msg)
