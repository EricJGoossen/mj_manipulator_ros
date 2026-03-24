"""Convert between mj_manipulator Trajectory and ROS 2 JointTrajectory.

This is the data bridge between the planning stack (mj_manipulator) and
ROS 2 hardware drivers. All conversions are pure data transformations
with no ROS node dependencies, so they're fully unit-testable.
"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Duration
from mj_manipulator.trajectory import Trajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def trajectory_to_msg(traj: Trajectory) -> JointTrajectory:
    """Convert mj_manipulator Trajectory to ROS 2 JointTrajectory.

    Args:
        traj: Dense trajectory from mj_manipulator (waypoints at control_dt).

    Returns:
        JointTrajectory message ready to send as a FollowJointTrajectory goal.

    Raises:
        ValueError: If trajectory has no joint_names set.
    """
    if traj.joint_names is None:
        raise ValueError(
            "Trajectory must have joint_names set for ROS 2 conversion. "
            "Set entity and joint_names when creating the trajectory."
        )

    msg = JointTrajectory()
    msg.joint_names = list(traj.joint_names)

    for i in range(traj.num_waypoints):
        point = JointTrajectoryPoint()
        point.positions = traj.positions[i].tolist()
        point.velocities = traj.velocities[i].tolist()
        point.accelerations = traj.accelerations[i].tolist()

        secs = int(traj.timestamps[i])
        nsecs = int((traj.timestamps[i] - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)

        msg.points.append(point)

    return msg


def msg_to_trajectory(
    msg: JointTrajectory,
    entity: str | None = None,
) -> Trajectory:
    """Convert ROS 2 JointTrajectory to mj_manipulator Trajectory.

    Args:
        msg: JointTrajectory message (e.g. from a recorded trajectory).
        entity: Entity name to set on the trajectory.

    Returns:
        mj_manipulator Trajectory.
    """
    n = len(msg.points)
    dof = len(msg.joint_names)

    timestamps = np.zeros(n)
    positions = np.zeros((n, dof))
    velocities = np.zeros((n, dof))
    accelerations = np.zeros((n, dof))

    for i, point in enumerate(msg.points):
        t = point.time_from_start
        timestamps[i] = t.sec + t.nanosec * 1e-9
        positions[i] = point.positions
        velocities[i] = point.velocities if point.velocities else 0.0
        accelerations[i] = point.accelerations if point.accelerations else 0.0

    return Trajectory(
        timestamps=timestamps,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        entity=entity,
        joint_names=list(msg.joint_names),
    )
