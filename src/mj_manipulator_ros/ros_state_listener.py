"""ROS 2 joint state subscriber.

Caches the latest joint positions/velocities from /joint_states,
keyed by joint name. Thread-safe for use from the HardwareContext
main thread while rclpy spins in a background thread.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy.node
from sensor_msgs.msg import JointState

from mj_manipulator_ros.interfaces import JOINT_STATES_TOPIC


class JointStateListener:
    """Subscribe to /joint_states and cache the latest values."""

    def __init__(self, node: rclpy.node.Node):
        self._node = node
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        self._efforts: dict[str, float] = {}

        self._sub = node.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._callback, 10,
        )

    def _callback(self, msg: JointState) -> None:
        with self._lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self._positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self._velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self._efforts[name] = msg.effort[i]

    def get_positions(self, joint_names: list[str]) -> np.ndarray | None:
        """Get latest positions for the given joints.

        Returns None if any joint hasn't been received yet.
        """
        with self._lock:
            try:
                return np.array([self._positions[n] for n in joint_names])
            except KeyError:
                return None

    def get_velocities(self, joint_names: list[str]) -> np.ndarray | None:
        """Get latest velocities for the given joints."""
        with self._lock:
            try:
                return np.array([self._velocities[n] for n in joint_names])
            except KeyError:
                return None

    @property
    def has_data(self) -> bool:
        """True if at least one joint state message has been received."""
        with self._lock:
            return len(self._positions) > 0
