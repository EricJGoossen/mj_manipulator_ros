# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Lightweight MuJoCo backend for the mock node without full :class:`~mj_manipulator.arm.Arm` / SimContext.

Drives joint positions directly from ROS trajectories or ``Float64MultiArray`` streaming
commands. Use this in tests and minimal demos when you only need joint-name-aligned
interfaces matching :mod:`mj_manipulator_ros.interfaces`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator_ros.trajectory_convert import msg_to_trajectory

if TYPE_CHECKING:
    from trajectory_msgs.msg import JointTrajectory

logger = logging.getLogger(__name__)


class KinematicMockBackend:
    """MuJoCo kinematic playback for mock ROS 2 interfaces.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified in place).
        joint_names_by_arm: Maps each arm name to the list of MuJoCo joint names
            controlled by that arm (order matches trajectory / streaming vectors).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names_by_arm: dict[str, list[str]],
    ):
        self._model = model
        self._data = data
        self._joint_names_by_arm = joint_names_by_arm
        #: Last (arm_name, object_name) passed to :meth:`grasp` (for tests / debugging).
        self.last_grasp: tuple[str, str] | None = None
        #: Last (arm_name, object_name) passed to :meth:`release`.
        self.last_release: tuple[str, str | None] | None = None
        self._name_to_qposadr: dict[str, int] = {}
        self._name_to_qveladr: dict[str, int] = {}
        for arm_joints in joint_names_by_arm.values():
            for jn in arm_joints:
                if jn in self._name_to_qposadr:
                    continue
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise ValueError(f"Joint {jn!r} not found in model")
                self._name_to_qposadr[jn] = int(model.jnt_qposadr[jid])
                self._name_to_qveladr[jn] = int(model.jnt_dofadr[jid])

    @property
    def joint_names_by_arm(self) -> dict[str, list[str]]:
        return self._joint_names_by_arm

    def start(self) -> None:
        mujoco.mj_forward(self._model, self._data)

    def stop(self) -> None:
        pass

    def execute_trajectory(self, msg: JointTrajectory, arm_name: str) -> bool:
        """Apply each waypoint by setting ``qpos`` (kinematic tracking)."""
        traj = msg_to_trajectory(msg, entity=arm_name)
        joint_names = list(traj.joint_names)
        for i in range(traj.num_waypoints):
            q = traj.positions[i]
            for j, name in enumerate(joint_names):
                adr = self._name_to_qposadr.get(name)
                if adr is None:
                    logger.error("Unknown joint %s in trajectory", name)
                    return False
                self._data.qpos[adr] = float(q[j])
            mujoco.mj_forward(self._model, self._data)
        return True

    def apply_forward_position(self, arm_name: str, positions: np.ndarray) -> None:
        """Apply one streaming position command for ``arm_name``."""
        names = self._joint_names_by_arm.get(arm_name)
        if not names:
            logger.warning("Unknown arm %s for forward position", arm_name)
            return
        q = np.asarray(positions, dtype=float).ravel()
        if q.size != len(names):
            logger.warning(
                "Expected %d positions for %s, got %d",
                len(names),
                arm_name,
                q.size,
            )
            return
        for i, jn in enumerate(names):
            self._data.qpos[self._name_to_qposadr[jn]] = float(q[i])
        mujoco.mj_forward(self._model, self._data)

    def grasp(self, arm_name: str, object_name: str) -> str | None:
        self.last_grasp = (arm_name, object_name)
        return None

    def release(self, arm_name: str, object_name: str | None = None) -> None:
        self.last_release = (arm_name, object_name)

    def get_joint_state(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """All arm joints: positions and velocities (zeros for velocities)."""
        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []
        for arm_joints in self._joint_names_by_arm.values():
            for jn in arm_joints:
                names.append(jn)
                positions.append(float(self._data.qpos[self._name_to_qposadr[jn]]))
                velocities.append(float(self._data.qvel[self._name_to_qveladr[jn]]))
        return names, np.array(positions), np.array(velocities)

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        if targets:
            for arm_name, q in targets.items():
                self.apply_forward_position(arm_name, q)

    @property
    def is_running(self) -> bool:
        return True
