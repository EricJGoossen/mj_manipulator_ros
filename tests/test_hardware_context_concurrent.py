# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for HardwareContext's concurrent multi-arm trajectory execution.

Covers HardwareContext._execute_trajectories() / _execute_single():

- A single trajectory takes the fast path (no threading, no shared start
  stamp) straight through _execute_single().
- Two or more trajectories are synchronized to ONE shared start stamp
  (computed once from the node clock) and sent from separate threads.
- Every trajectory in a batch must share identical timestamps -- checked
  up front, before anything is sent to any action server.
- If one arm's goal fails, rejects, times out, or the caller's abort_fn
  fires, a shared cancel_event tells every other still-running arm in the
  batch to cancel its own goal too, instead of running to completion
  alone.
- An exception raised while sending/polling one arm's goal propagates out
  of _execute_trajectories(), and still-running siblings get cancelled.

No real ROS 2 node or action server is used -- HardwareContext is built
directly (bypassing __enter__, which would try to connect to one) with a
fake node and fake per-arm trajectory clients standing in for
ArmTrajectoryClient. This keeps the tests fast and deterministic while
exercising the real _execute_trajectories/_execute_single logic.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from mj_manipulator.planning import PlanGroupResult
from mj_manipulator.trajectory import Trajectory

from tests.conftest import HAS_ROS2

if not HAS_ROS2:
    pytest.skip("ROS 2 not installed", allow_module_level=True)

import rclpy.duration
import rclpy.time
from control_msgs.action import FollowJointTrajectory

from mj_manipulator_ros.config import HardwareConfig
from mj_manipulator_ros.hardware_context import HardwareContext

SUCCESS = FollowJointTrajectory.Result.SUCCESSFUL
ABORTED = getattr(FollowJointTrajectory.Result, "PATH_TOLERANCE_VIOLATED", SUCCESS + 1)


# ---------------------------------------------------------------------------
# Fakes -- stand-ins for the ROS 2 node/action-client layer
# ---------------------------------------------------------------------------


class FakeClock:
    def now(self):
        return rclpy.time.Time()


class FakeNode:
    """Just enough of a rclpy.node.Node for _execute_trajectories()'s
    self._node.get_clock().now() call."""

    def get_clock(self):
        return FakeClock()


class FakeResultFuture:
    """Stand-in for goal_handle.get_result_async()'s Future.

    Starts NOT done unless constructed with an immediate result --
    lets a test hold a goal "in flight" until it explicitly resolves it,
    to exercise the cancel-on-sibling-failure polling loop.
    """

    def __init__(self):
        self._done = threading.Event()
        self._value = None

    def done(self) -> bool:
        return self._done.is_set()

    def result(self):
        return self._value

    def resolve(self, error_code: int) -> None:
        self._value = SimpleNamespace(result=SimpleNamespace(error_code=error_code))
        self._done.set()


class FakeGoalHandle:
    def __init__(self, *, error_code: int | None = SUCCESS):
        """error_code=None leaves the goal unresolved until resolve() is
        called explicitly (simulates a goal that's still executing)."""
        self.cancel_calls = 0
        self._result_future = FakeResultFuture()
        if error_code is not None:
            self._result_future.resolve(error_code)

    def get_result_async(self) -> FakeResultFuture:
        return self._result_future

    def cancel_goal_async(self) -> None:
        self.cancel_calls += 1

    def resolve(self, error_code: int = SUCCESS) -> None:
        self._result_future.resolve(error_code)


class FakeClient:
    """Stand-in for ArmTrajectoryClient, used only via send_goal()."""

    def __init__(self, goal_handle=None, *, raises: Exception | None = None):
        self._goal_handle = goal_handle
        self._raises = raises
        self.sent_msgs: list = []

    def send_goal(self, msg, timeout_sec: float = 5.0):
        self.sent_msgs.append(msg)
        if self._raises is not None:
            raise self._raises
        return self._goal_handle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(clients: dict[str, FakeClient], *, action_timeout: float = 30.0) -> HardwareContext:
    """Build a HardwareContext without going through __enter__ (which would
    try to connect to a real ROS 2 node and wait for action servers)."""
    config = HardwareConfig(arms=[], action_timeout=action_timeout)
    ctx = HardwareContext(config)
    ctx._node = FakeNode()
    ctx._arm_clients = clients
    return ctx


def _traj(start: float, end: float, *, entity: str, n: int = 5, timestamps=None) -> Trajectory:
    if timestamps is not None:
        n = len(timestamps)
    positions = np.linspace(start, end, n).reshape(-1, 1)
    return Trajectory(
        timestamps=timestamps if timestamps is not None else np.linspace(0.0, 1.0, n),
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        joint_names=["joint1"],
        entity=entity,
    )


# ---------------------------------------------------------------------------
# Basic dispatch / single vs. multi path
# ---------------------------------------------------------------------------


class TestExecuteTrajectoriesBasics:
    def test_empty_list_returns_true_without_touching_any_client(self):
        client = FakeClient(raises=AssertionError("should never be called"))
        ctx = _make_context({"left": client})
        assert ctx._execute_trajectories([]) is True
        assert client.sent_msgs == []

    def test_abort_fn_already_true_short_circuits_before_sending(self):
        client = FakeClient(raises=AssertionError("should never be called"))
        ctx = _make_context({"left": client})
        traj = _traj(0.0, 0.3, entity="left")
        assert ctx._execute_trajectories([traj], abort_fn=lambda: True) is False
        assert client.sent_msgs == []

    def test_single_trajectory_takes_fast_path_no_shared_stamp(self):
        """A lone trajectory skips the synchronized-start machinery
        entirely -- no start_stamp is attached to its message."""
        client = FakeClient(FakeGoalHandle())
        ctx = _make_context({"left": client})
        traj = _traj(0.0, 0.3, entity="left")
        assert ctx._execute_trajectories([traj]) is True
        assert len(client.sent_msgs) == 1
        stamp = client.sent_msgs[0].header.stamp
        assert stamp.sec == 0 and stamp.nanosec == 0

    def test_unknown_entity_raises(self):
        ctx = _make_context({})
        traj = _traj(0.0, 0.3, entity="left")
        with pytest.raises(ValueError, match="No trajectory client"):
            ctx._execute_trajectories([traj])

    def test_no_entity_raises(self):
        ctx = _make_context({})
        traj = _traj(0.0, 0.3, entity=None)
        with pytest.raises(ValueError, match="no entity"):
            ctx._execute_trajectories([traj])


class TestExecuteTrajectoriesSynchronizedStart:
    def test_two_trajectories_share_one_start_stamp(self):
        left_client = FakeClient(FakeGoalHandle())
        right_client = FakeClient(FakeGoalHandle())
        ctx = _make_context({"left": left_client, "right": right_client})

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")
        assert ctx._execute_trajectories([traj_left, traj_right]) is True

        left_stamp = left_client.sent_msgs[0].header.stamp
        right_stamp = right_client.sent_msgs[0].header.stamp
        assert (left_stamp.sec, left_stamp.nanosec) == (right_stamp.sec, right_stamp.nanosec)
        # A synchronized start is scheduled in the future, not "now".
        assert left_stamp.sec > 0 or left_stamp.nanosec > 0

    def test_mismatched_timestamps_raises_before_sending_anything(self):
        left_client = FakeClient(raises=AssertionError("should never be called"))
        right_client = FakeClient(raises=AssertionError("should never be called"))
        ctx = _make_context({"left": left_client, "right": right_client})

        traj_left = _traj(0.0, 0.3, entity="left", timestamps=np.array([0.0, 0.5, 1.0]))
        traj_right = _traj(0.0, -0.3, entity="right", timestamps=np.array([0.0, 0.3, 1.0]))

        with pytest.raises(ValueError, match="timestamps"):
            ctx._execute_trajectories([traj_left, traj_right])

        assert left_client.sent_msgs == []
        assert right_client.sent_msgs == []


# ---------------------------------------------------------------------------
# Sibling-failure cancellation
# ---------------------------------------------------------------------------


class TestSiblingFailureCancellation:
    def test_one_arm_failing_cancels_still_running_sibling(self):
        """"right" fails outright; "left" is still polling its goal
        (never resolved) when that happens -- it must be cancelled
        rather than left to run to completion on its own."""
        left_handle = FakeGoalHandle(error_code=None)  # never resolves on its own
        left_client = FakeClient(left_handle)
        right_client = FakeClient(FakeGoalHandle(error_code=ABORTED))
        ctx = _make_context({"left": left_client, "right": right_client})

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")

        assert ctx._execute_trajectories([traj_left, traj_right]) is False
        assert left_handle.cancel_calls >= 1

    def test_goal_rejection_cancels_sibling(self):
        """send_goal() returning None (server rejected the goal) fails
        that arm immediately and must still cancel a sibling that was
        accepted and is polling."""
        left_handle = FakeGoalHandle(error_code=None)
        left_client = FakeClient(left_handle)
        right_client = FakeClient(None)  # rejected
        ctx = _make_context({"left": left_client, "right": right_client})

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")

        assert ctx._execute_trajectories([traj_left, traj_right]) is False
        assert left_handle.cancel_calls >= 1

    def test_caller_abort_fn_cancels_every_running_arm(self):
        left_handle = FakeGoalHandle(error_code=None)
        right_handle = FakeGoalHandle(error_code=None)
        ctx = _make_context(
            {"left": FakeClient(left_handle), "right": FakeClient(right_handle)}
        )

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")

        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()

        assert ctx._execute_trajectories([traj_left, traj_right], abort_fn=stop.is_set) is False
        assert left_handle.cancel_calls >= 1
        assert right_handle.cancel_calls >= 1

    def test_timeout_cancels_and_fails_the_arm(self):
        handle = FakeGoalHandle(error_code=None)  # never resolves
        ctx = _make_context({"left": FakeClient(handle)}, action_timeout=0.01)

        # Single-trajectory path still honors action_timeout.
        traj = _traj(0.0, 0.3, entity="left")
        assert ctx._execute_trajectories([traj]) is False
        assert handle.cancel_calls >= 1

    def test_skips_send_if_sibling_already_failed_before_this_arm_sent(self):
        """If the cancel_event is already set by the time this arm's
        thread gets to send its goal, it must not send at all."""
        right_client = FakeClient(FakeGoalHandle(error_code=ABORTED))
        left_client = FakeClient(raises=AssertionError("should not be reached"))
        ctx = _make_context({"left": left_client, "right": right_client})

        # Force "left"'s thread to run after "right" has already failed
        # by making its send_goal block until we release it.
        release = threading.Event()

        def blocking_send(msg, timeout_sec=5.0):
            release.wait(timeout=2.0)
            left_client.sent_msgs.append(msg)
            return FakeGoalHandle(error_code=SUCCESS)

        left_client.send_goal = blocking_send

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")

        result_holder: dict[str, bool] = {}

        def run() -> None:
            result_holder["result"] = ctx._execute_trajectories([traj_left, traj_right])

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # Give "right"'s thread time to fail and set cancel_event, then
        # release "left" to proceed with the (now-pointless) send.
        time.sleep(0.1)
        release.set()
        t.join(timeout=5.0)

        assert not t.is_alive()
        assert result_holder["result"] is False


class TestExecuteTrajectoriesErrorPropagation:
    def test_exception_in_one_thread_propagates_and_cancels_sibling(self):
        boom = RuntimeError("boom")
        left_handle = FakeGoalHandle(error_code=None)
        left_client = FakeClient(left_handle)
        right_client = FakeClient(raises=boom)
        ctx = _make_context({"left": left_client, "right": right_client})

        traj_left = _traj(0.0, 0.3, entity="left")
        traj_right = _traj(0.0, -0.3, entity="right")

        with pytest.raises(RuntimeError, match="boom"):
            ctx._execute_trajectories([traj_left, traj_right])

        assert left_handle.cancel_calls >= 1


# ---------------------------------------------------------------------------
# execute() dispatch
# ---------------------------------------------------------------------------


class TestExecuteDispatch:
    def test_dispatches_single_trajectory(self):
        client = FakeClient(FakeGoalHandle())
        ctx = _make_context({"left": client})
        traj = _traj(0.0, 0.3, entity="left")
        assert ctx.execute(traj) is True
        assert len(client.sent_msgs) == 1

    def test_dispatches_plan_group_result_as_synchronized_batch(self):
        left_client = FakeClient(FakeGoalHandle())
        right_client = FakeClient(FakeGoalHandle())
        ctx = _make_context({"left": left_client, "right": right_client})

        group = PlanGroupResult.from_trajectories(
            {
                "left": _traj(0.0, 0.3, entity="left"),
                "right": _traj(0.0, -0.3, entity="right"),
            }
        )
        assert ctx.execute(group) is True
        left_stamp = left_client.sent_msgs[0].header.stamp
        right_stamp = right_client.sent_msgs[0].header.stamp
        assert (left_stamp.sec, left_stamp.nanosec) == (right_stamp.sec, right_stamp.nanosec)

    def test_unsupported_type_raises_type_error(self):
        ctx = _make_context({})
        with pytest.raises(TypeError):
            ctx.execute(object())
