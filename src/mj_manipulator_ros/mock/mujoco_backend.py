"""MuJoCo backend for the mock node.

Wraps a SimContext to provide the imperative interface the mock ROS 2
node needs: execute trajectories, close/open grippers, read joint state.
Keeps ROS boilerplate in mock_node.py and MuJoCo logic here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mj_manipulator.sim_context import SimContext
from mj_manipulator.trajectory import Trajectory

from mj_manipulator_ros.trajectory_convert import msg_to_trajectory

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from trajectory_msgs.msg import JointTrajectory

logger = logging.getLogger(__name__)


class MuJoCoBackend:
    """Drives MuJoCo simulation in response to ROS 2 requests.

    Args:
        model_path: Path to the MuJoCo XML model.
        arms: Dict mapping arm names to Arm instances.
        physics: Whether to use physics simulation (default True).
        show_viewer: Whether to open a MuJoCo viewer window.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[str, Arm],
        *,
        physics: bool = True,
        show_viewer: bool = False,
    ):
        self._model = model
        self._data = data
        self._arms = arms
        self._physics = physics
        self._show_viewer = show_viewer

        self._ctx: SimContext | None = None

    def start(self) -> None:
        """Enter the SimContext."""
        self._ctx = SimContext(
            self._model, self._data, self._arms,
            physics=self._physics,
            headless=not self._show_viewer,
        )
        self._ctx.__enter__()

    def stop(self) -> None:
        """Exit the SimContext."""
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    def execute_trajectory(self, msg: JointTrajectory, entity: str) -> bool:
        """Execute a JointTrajectory message in MuJoCo.

        Converts the ROS message to an mj_manipulator Trajectory and
        executes it through the SimContext.

        Args:
            msg: JointTrajectory from the action goal.
            entity: Entity name for routing (arm name).

        Returns:
            True if execution completed successfully.
        """
        if self._ctx is None:
            logger.error("Backend not started")
            return False

        traj = msg_to_trajectory(msg, entity=entity)
        return self._ctx.execute(traj)

    def grasp(self, arm_name: str, object_name: str) -> str | None:
        """Close gripper on the specified arm."""
        if self._ctx is None:
            return None
        return self._ctx.arm(arm_name).grasp(object_name)

    def release(self, arm_name: str, object_name: str | None = None) -> None:
        """Open gripper on the specified arm."""
        if self._ctx is None:
            return
        self._ctx.arm(arm_name).release(object_name)

    def get_joint_state(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Read current joint positions and velocities from MuJoCo.

        Returns:
            Tuple of (joint_names, positions, velocities).
        """
        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []

        for arm in self._arms.values():
            for idx in arm.joint_qpos_indices:
                jnt_id = np.searchsorted(
                    self._model.jnt_qposadr, idx, side="right",
                ) - 1
                name = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, int(jnt_id),
                )
                names.append(name or f"joint_{jnt_id}")
                positions.append(float(self._data.qpos[idx]))

            for idx in arm.joint_qvel_indices:
                velocities.append(float(self._data.qvel[idx]))

        return names, np.array(positions), np.array(velocities)

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Advance one control cycle (for streaming control)."""
        if self._ctx is not None:
            self._ctx.step(targets)

    @property
    def is_running(self) -> bool:
        """Check if the SimContext is still active."""
        if self._ctx is None:
            return False
        return self._ctx.is_running()
