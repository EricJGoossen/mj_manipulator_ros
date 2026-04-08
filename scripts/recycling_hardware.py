# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Run the Geodude recycling demo using mj_manipulator_ros.HardwareContext.

This keeps the demo logic (pickup/place/go_home) identical to simulation, but swaps the
ExecutionContext to ROS 2 hardware I/O.

Prereqs:
  - ROS stack running (sim or real): controllers + /joint_states
  - ROS_WORKSPACE set (same as bimanual launch)
  - geodude_assets assembly deps installed:
      cd geodude_assets && uv sync --extra assembly

Run:
  export ROS_WORKSPACE=~/catkin_ws
  cd robot-code
  uv run python mj_manipulator_ros/scripts/recycling_hardware.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from pathlib import Path

import mujoco
import yaml
from geodude.config import GeodudConfig
from geodude.demo_loader import (
    _spawn_manipulable_objects,
    get_demo_functions,
    inject_robot,
    load_demo,
    resolve_scene,
)

from mj_manipulator_ros.hardware_context import HardwareContext


def _start_viser_mirror(
    *,
    robot,
    ctx,
    hz: float = 30.0,
    teleop_hz: float = 30.0,
    open_browser: bool = False,
):
    """Launch a Viser viewer and mirror ROS joint states into MuJoCo."""
    try:
        from mj_viser import MujocoViewer
    except Exception as e:
        raise RuntimeError(
            "Viser viewer not available. Install mj_viser (and its deps) in this env."
        ) from e

    viewer = MujocoViewer(
        robot.model,
        robot.data,
        label="Geodude [hardware]",
        show_sim_controls=False,
        show_visibility=False,
    )

    # Build UI layout (match geodude.console: Stop + tabs + Teleop).
    gui = viewer._server.gui
    stop_btn = gui.add_button("Stop", color="red")

    @stop_btn.on_click
    def _on_stop(_event):
        robot.request_abort()

    tabs = gui.add_tab_group()
    panels_to_register = []

    with tabs.add_tab("Teleop"):
        # IMPORTANT: use a conservative teleop config for real hardware.
        # Default mj_manipulator teleop allows ~1.5 rad/s which can trip UR overspeed.
        from mj_manipulator.teleop import SafetyMode, TeleopConfig, TeleopController
        from mj_viser.teleop_panel import TeleopPanel

        thz = max(float(teleop_hz), 1.0)
        # Keep a modest joint-speed cap (~1.5 rad/s) regardless of teleop rate.
        max_step = min(0.05, 1.5 / thz)
        for side in ("right", "left"):
            arm = robot._resolve_arm(side)
            controller = TeleopController(
                arm,
                ctx,
                config=TeleopConfig(
                    # twist_dt must match TeleopPanel._teleop_loop rate (control_hz).
                    max_joint_step=max_step,
                    twist_dt=1.0 / thz,
                    safety_mode=SafetyMode.REJECT,
                ),
            )

            teleop_panel = TeleopPanel(
                arm=arm,
                controller=controller,
                model=robot.model,
                data=robot.data,
                gripper_body_prefix=f"{side}_ur5e/gripper/",
                arm_label=f"{side.title()} Arm",
                abort_fn=robot.is_abort_requested,
                clear_abort_fn=robot.clear_abort,
                control_hz=thz,
            )
            teleop_panel.setup(gui, viewer)
            panels_to_register.append(teleop_panel)

    viewer.launch_passive(open_browser=open_browser)
    # We set up panels manually inside the tab context above, so register them for
    # on_sync *after* launch to avoid MujocoViewer calling setup() a second time.
    viewer._panels.extend(panels_to_register)

    stop = threading.Event()

    def _loop() -> None:
        dt = 1.0 / max(hz, 1e-3)
        # Mirror arm joints + vention rails.
        left_arm_names = robot.config.joint_names(robot.config.left_arm)
        right_arm_names = robot.config.joint_names(robot.config.right_arm)

        # /joint_states uses vention_{left,right}, while the MuJoCo model uses
        # {left,right}_arm_linear_vention.
        vention_ros_names = ["vention_left", "vention_right"]
        vention_mj_names = ["left_arm_linear_vention", "right_arm_linear_vention"]

        vention_qpos_adrs: list[int] = []
        for mj_name in vention_mj_names:
            j_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if j_id < 0:
                vention_qpos_adrs.append(-1)
            else:
                vention_qpos_adrs.append(int(robot.model.jnt_qposadr[j_id]))

        while not stop.is_set():
            if not viewer.is_running():
                break
            try:
                # One synchronized read from /joint_states
                names = [*vention_ros_names, *left_arm_names, *right_arm_names]
                q = ctx.get_joint_positions(names)
                if q is None:
                    time.sleep(dt)
                    continue

                # Prevent races with TeleopPanel's internal control loop thread.
                from mj_viser.teleop_panel import TeleopPanel

                with TeleopPanel._sim_lock:  # noqa: SLF001
                    # Vention rails
                    for i, adr in enumerate(vention_qpos_adrs):
                        if adr >= 0:
                            robot.data.qpos[adr] = float(q[i])

                    # Always mirror arm joints from /joint_states. Teleop publishes to ROS only;
                    # it does not advance MuJoCo `robot.data.qpos`. If q_current is stale,
                    # TeleopController's per-step clamp (max_joint_step) stalls after a few steps.
                    base = len(vention_ros_names)
                    for i, idx in enumerate(robot.left.joint_qpos_indices):
                        robot.data.qpos[idx] = float(q[base + i])
                    base += len(robot.left.joint_qpos_indices)
                    for i, idx in enumerate(robot.right.joint_qpos_indices):
                        robot.data.qpos[idx] = float(q[base + i])

                    mujoco.mj_forward(robot.model, robot.data)
                    viewer.sync()
            except Exception:
                # Don't kill the session if ROS momentarily hiccups.
                pass
            time.sleep(dt)

    thread = threading.Thread(target=_loop, name="viser_hw_mirror", daemon=True)
    thread.start()

    def _shutdown():
        stop.set()
        thread.join(timeout=1.0)

    return viewer, _shutdown


def _ssot_path() -> Path:
    ws = os.getenv("ROS_WORKSPACE")
    if not ws:
        raise RuntimeError("ROS_WORKSPACE is not set (expected same as bimanual launch).")
    return Path(ws) / "src" / "bimanual" / "config" / "SSoT.yaml"


def _parse_origin(origin_value: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Parse SSoT origin field: '\"x y z roll pitch yaw\"' (radians)."""
    s = str(origin_value).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    parts = [p for p in s.split() if p]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 floats in origin, got {len(parts)}: {origin_value!r}")
    x, y, z, r, p, yaw = (float(v) for v in parts)
    return (x, -y, z), (r, yaw, p) # DONT CHANGE THIS ORDERING PLEAASEEEEE


def _export_mjcf_matching_ssot(ssot_path: Path) -> Path:
    """Export an assembled geodude MJCF with mount pose from SSoT arms.*.origin."""
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

    tmpdir = Path(tempfile.mkdtemp(prefix="geodude_hw_mjcf_"))
    out = tmpdir / "geodude_hw.xml"
    print(f"Exporting MuJoCo model to {out}")
    print(f"  mount_pos={mount_pos}  mount_rpy={mount_rpy}")
    attach_arms_to_vention(
        save_file=True,
        dir=str(tmpdir),
        filename=out.name,
        left_gripper_type=left_gripper,
        right_gripper_type=right_gripper,
        mount_pos=mount_pos,
        mount_rpy=mount_rpy,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Geodude recycling demo (hardware context)")
    parser.add_argument("--viser", action="store_true", help="Launch Viser viewer (http://localhost:8080)")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically (Viser)")
    parser.add_argument("--mirror-hz", type=float, default=30.0, help="Viser mirror rate (Hz)")
    parser.add_argument(
        "--teleop-hz",
        type=float,
        default=500.0,
        help="Teleop loop rate: TeleopController.step / ctx.step_cartesian (Hz). E.g. 500",
    )
    args = parser.parse_args()

    ssot = _ssot_path()
    mjcf_path = _export_mjcf_matching_ssot(ssot)

    # Load demo scene (same as CLI: --demo recycling)
    objects, fixtures, demo_module = resolve_scene("recycling", None)
    spawn_count = demo_module.scene.get("spawn_count") if demo_module and hasattr(demo_module, "scene") else None

    # Build a Geodude robot using the exported MJCF (mount pose parity).
    config = GeodudConfig.default()
    config.model_path = mjcf_path
    from geodude.robot import Geodude

    robot = Geodude(config=config, objects=objects)
    robot.setup_scene(fixtures=fixtures if fixtures else None)
    fixture_types = set(fixtures.keys()) if fixtures else set()
    _spawn_manipulable_objects(robot, objects, fixture_types, spawn_count=spawn_count)

    if demo_module is None:
        demo_module = load_demo("recycling")
    inject_robot(demo_module, robot)

    # Interactive console (like `geodude --demo recycling`, but with HardwareContext).
    with HardwareContext(
        ssot_path=ssot,
        node_name="recycling_hardware",
        default_streaming_controller="forward_position_controller",
    ) as ctx:
        # Match streaming sleep in HardwareContext to teleop loop (step/step_cartesian).
        ctx._config.control_dt = 1.0 / max(args.teleop_hz, 1.0)
        from IPython.terminal.embed import InteractiveShellEmbed
        from IPython.terminal.prompts import Prompts, Token

        # Geodude primitives use robot._active_context to choose sim vs hardware.
        robot._active_context = ctx  # type: ignore[assignment]

        viser_shutdown = None
        if args.viser:
            _, viser_shutdown = _start_viser_mirror(
                robot=robot,
                ctx=ctx,
                hz=max(args.mirror_hz, args.teleop_hz),
                teleop_hz=args.teleop_hz,
                open_browser=args.open_browser,
            )
            print("Viser viewer: http://localhost:8080")

        def commands() -> None:
            print("""
Hardware Recycling Demo (interactive)
====================================

Core:
  robot.pickup()
  robot.place("recycle_bin")
  robot.go_home()

Demo:
  sort_all()

Notes:
  - This console executes through mj_manipulator_ros.HardwareContext (ROS 2).
  - The MuJoCo model here is for planning/state bookkeeping; real motion is via ROS.
  - If started with --viser, joint states are mirrored live into a Viser viewer.
""")

        user_ns: dict = {
            "robot": robot,
            "ctx": ctx,
            "commands": commands,
        }
        for name_fn, func in get_demo_functions(demo_module).items():
            user_ns[name_fn] = func

        banner = (
            "\n"
            + "=" * 60
            + "\n"
            + "  Geodude [hardware] | demo: recycling\n"
            + "=" * 60
            + "\n\n"
            + "  commands()   — quick reference\n"
            + "  robot.<tab>  — tab completion\n"
        )
        if demo_module.__doc__:
            banner += f"\n  {demo_module.__doc__.strip()}\n"

        shell = InteractiveShellEmbed(
            header=banner,
            user_ns=user_ns,
            colors="neutral",
        )

        class GeodudePrompts(Prompts):
            def in_prompt_tokens(self, cli=None):
                return [
                    (Token.Prompt, f"Geodude [hardware] [{self.shell.execution_count}]: "),
                ]

            def out_prompt_tokens(self, cli=None):
                return [
                    (Token.OutPrompt, f"Out[{self.shell.execution_count}]: "),
                ]

        shell.prompts = GeodudePrompts(shell)

        try:
            shell()
        finally:
            if viser_shutdown is not None:
                viser_shutdown()
            robot._active_context = None  # type: ignore[assignment]

    print("Done.")


if __name__ == "__main__":
    main()

