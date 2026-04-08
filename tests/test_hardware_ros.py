"""Test ``mj_manipulator_ros.HardwareContext`` against a running ROS stack.

Connects to the robot via ROS 2, opens a MuJoCo viewer that tracks
the real joint states, and optionally runs motion commands.

Prerequisites:
    1. Robot stack running:  ros2 launch bimanual robot.launch.py
    2. MoveIt running:       ros2 launch bimanual_moveit moveit.launch.py

Usage:
    source /opt/ros/humble/setup.bash
    source ~/catkin_ws/install/setup.bash
    uv run mjpython mj_manipulator_ros/tests/test_hardware_ros.py
"""

import os
import sys
import threading
import time
from pathlib import Path

import yaml

try:
    import rclpy
except ImportError:
    print(
        "ERROR: rclpy not found. Source your ROS 2 workspace first:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source ~/catkin_ws/install/setup.bash"
    )
    sys.exit(1)

import mujoco
import mujoco.viewer
import numpy as np

from mj_manipulator_ros.hardware_context import HardwareContext

_HERE = Path(__file__).resolve()


def _ssot_path() -> Path:
    """Resolve SSoT.yaml path (prefer ROS_WORKSPACE like bimanual launch)."""
    ws = os.getenv("ROS_WORKSPACE")
    if ws:
        p = Path(ws) / "src" / "bimanual" / "config" / "SSoT.yaml"
        if p.exists():
            return p
    # Fallbacks for local layouts
    candidates = [
        # Historical layout (if bimanual is vendored inside robot-code)
        _HERE.parents[2] / "bimanual" / "config" / "SSoT.yaml",
        # Common catkin layout (bimanual is sibling of robot-code under catkin_ws/src)
        _HERE.parents[3] / "bimanual" / "config" / "SSoT.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


SSOT_PATH = _ssot_path()


def _parse_origin(origin_value: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Parse SSoT origin field: '\"x y z roll pitch yaw\"' (radians)."""
    s = str(origin_value).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    parts = [p for p in s.split() if p]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 floats in origin, got {len(parts)}: {origin_value!r}")
    x, y, z, r, p, yaw = (float(v) for v in parts)
    return (x, -y, z), (r, yaw, p) # weird convention for mujoco... did we miss something in the vention base mount orientation?


_UR5E_SUFFIXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _ros_to_mujoco_joint_name(ros_name: str) -> str | None:
    """Best-effort mapping for bimanual ROS joint names to Geodude MuJoCo joint names."""
    if ros_name == "vention_left":
        return "left_arm_linear_vention"
    if ros_name == "vention_right":
        return "right_arm_linear_vention"
    for suffix in _UR5E_SUFFIXES:
        if ros_name.endswith(suffix):
            side = ros_name[: -(len(suffix) + 1)]
            return f"{side}_ur5e/{suffix}"
    return None


def _build_mujoco_model_from_ssot(ssot_path: Path) -> mujoco.MjModel:
    """Assemble a MuJoCo model whose mount pose matches SSoT arms.*.origin."""
    try:
        from geodude_assets.assembly import attach_arms_to_vention
    except Exception as e:
        raise RuntimeError(
            "geodude_assets assembly dependencies missing. Run:\n"
            "  cd geodude_assets && uv sync --extra assembly\n"
            "and rerun."
        ) from e

    with open(ssot_path, "r", encoding="utf-8") as f:
        ssot = yaml.safe_load(f)

    arms = ssot.get("arms", {})
    origin = (arms.get("right") or {}).get("origin") or (arms.get("left") or {}).get("origin")
    if origin is None:
        raise KeyError("SSoT missing arms.{left,right}.origin")
    mount_pos, mount_rpy = _parse_origin(origin)

    hands = ssot.get("hands", {})
    left_gripper = "2f140" if (hands.get("left") or {}).get("enabled", False) else None
    right_gripper = "2f140" if (hands.get("right") or {}).get("enabled", False) else None

    print(f"SSoT origin: {origin!r} -> mount_pos={mount_pos} mount_rpy={mount_rpy}")
    return attach_arms_to_vention(
        save_file=False,
        dir="",
        filename="",
        left_gripper_type=left_gripper,
        right_gripper_type=right_gripper,
        mount_pos=mount_pos,
        mount_rpy=mount_rpy,
    )


def _go_home(ctx: HardwareContext, *, duration: float = 4.0) -> bool:
    """Send a simple 2-point trajectory to each enabled arm using named poses from SSoT."""
    if "ready" not in ctx.named_poses:
        print("No 'ready' pose available in ctx.named_poses")
        return False
    ok = True
    arm_cfg_by_name = {a.name: a for a in ctx._config.arms}
    for arm_name, q in ctx.named_poses["ready"].items():
        q = np.asarray(q, dtype=float)
        ts = np.array([0.0, duration], dtype=float)
        pos = np.vstack([q, q])
        zeros = np.zeros_like(pos)
        arm_cfg = arm_cfg_by_name.get(arm_name)
        if arm_cfg is None:
            print(f"Skipping pose for unknown arm {arm_name!r}")
            ok = False
            continue
        from mj_manipulator.trajectory import Trajectory

        traj = Trajectory(
            timestamps=ts,
            positions=pos,
            velocities=zeros,
            accelerations=zeros,
            entity=arm_name,
            joint_names=arm_cfg.joint_names,
        )
        ok = ctx.execute(traj) and ok
    return ok


def main():
    rclpy.init()
    model = _build_mujoco_model_from_ssot(SSOT_PATH)

    print(f"SSoT path: {SSOT_PATH}")

    print("\nConnecting to robot...")
    with HardwareContext(ssot_path=SSOT_PATH, node_name="hardware_context_test") as ctx:
        print("Connected!")
        print(f"Named poses from SSoT: {list(ctx.named_poses.keys())}")

        print("\nOpening viewer — close the window to exit.\n")

        # Visualization data (independent of anything else).
        viz_data = mujoco.MjData(model)

        def sync_viz():
            """Copy latest ROS joint positions into viz_data and forward-kinematics."""
            # Read all joints from synchronized /joint_states
            ros_names = ["vention_left", "vention_right"]
            for side in ("right", "left"):
                ros_names.extend([f"{side}_{suf}" for suf in _UR5E_SUFFIXES])
            q = ctx.get_joint_positions(ros_names)
            if q is None:
                return
            for ros_name, val in zip(ros_names, q, strict=False):
                mj_name = _ros_to_mujoco_joint_name(ros_name)
                if mj_name is None:
                    continue
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
                if jid >= 0:
                    viz_data.qpos[model.jnt_qposadr[jid]] = float(val)

            mujoco.mj_forward(model, viz_data)

        sync_viz()

        with mujoco.viewer.launch_passive(model, viz_data) as viewer:
            viewer.sync()

            def run_commands():
                """Run motion commands while the viewer stays open."""
                try:
                    print("Sending go_home (ready pose) via HardwareContext...")
                    success = _go_home(ctx)
                    print(f"go_home: {'OK' if success else 'FAILED'}")
                    # # Smoke print joint positions from ROS side vs MuJoCo viz_data.
                    # ros_names = ["vention_left", "vention_right"]
                    # for side in ("right", "left"):
                    #     ros_names.extend([f"{side}_{suf}" for suf in _UR5E_SUFFIXES])
                    # ros_q = ctx.get_joint_positions(ros_names)
                    # print(f"ROS /joint_states names: {ros_names}")
                    # print(f"ROS /joint_states positions: {None if ros_q is None else np.round(ros_q, 4).tolist()}")

                    # mj_vals = []
                    # for n in ros_names:
                    #     mj_name = _ros_to_mujoco_joint_name(n)
                    #     if mj_name is None:
                    #         mj_vals.append(None)
                    #         continue
                    #     jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
                    #     mj_vals.append(float(viz_data.qpos[model.jnt_qposadr[jid]]) if jid >= 0 else None)
                    # print(f"MuJoCo (viz) positions: {mj_vals}")
                except Exception as e:
                    print(f"ERROR in run_commands: {e}")
                    import traceback
                    traceback.print_exc()

            cmd_thread = threading.Thread(target=run_commands, daemon=True)
            cmd_thread.start()

            while viewer.is_running():
                sync_viz()
                viewer.sync()
                time.sleep(1.0 / 30)

            cmd_thread.join(timeout=30.0)

    rclpy.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
