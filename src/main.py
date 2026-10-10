r"""Run from PowerShell: .\.venv\Scripts\python.exe src\main.py"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sim import Simulation


def smoke_check(output):
    sim = Simulation()
    for _ in range(500):
        sim.step()
    start = sim.data.qpos[:3].copy()
    sim.drive(0.4, 0)
    for _ in range(1000):
        sim.step()
    displacement = sim.data.qpos[:3] - start
    assert np.isfinite(sim.data.qpos).all(), "Non-finite robot state"
    assert displacement[0] > 0.3, f"Forward motion failed: {displacement}"
    sim.drive(0, 1)
    yaw_start = sim.data.qpos[3:7].copy()
    for _ in range(500):
        sim.step()
    assert abs(np.dot(yaw_start, sim.data.qpos[3:7])) < 0.99, "Turn failed"
    sim.reset()
    with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
        renderer.update_scene(sim.data, camera="front")
        pixels = renderer.render()
        from PIL import Image
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
        red = (pixels[:, :, 0] > 100) & (pixels[:, :, 0] > pixels[:, :, 1] * 1.5) & (pixels[:, :, 0] > pixels[:, :, 2] * 1.5)
        assert red.sum() > 100, "Red target not visible"
    print(f"PASS: forward displacement={displacement[0]:.3f} m; turn; front camera -> {output}")


def interactive():
    sim = Simulation()
    command = {"linear": 0.0, "angular": 0.0, "until": 0.0, "reset": False}

    # Letter keys are taken by viewer display toggles (W wireframe, S shadow, ...),
    # so drive with GLFW arrow-key codes. Backspace is also the viewer's own reset.
    moves = {265: (0.4, 0), 264: (-0.4, 0), 263: (0, 1), 262: (0, -1), 32: (0, 0)}

    def on_key(key):
        if key in moves:
            linear, angular = moves[key]
            command.update(linear=linear, angular=angular, until=time.monotonic() + 0.5)
        elif key == 259:
            command["reset"] = True

    print("Viewerにフォーカス: ↑/↓ 前進/後退、←/→ 左右旋回、Space 停止、Backspace リセット")
    print("1回のキー入力で0.5秒動作。続けて押すと延長。ウィンドウを閉じると終了。")
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = [0.5, 0, 0]
        viewer.cam.distance = 8
        viewer.cam.elevation = -45
        while viewer.is_running():
            tick = time.monotonic()
            with viewer.lock():
                if command["reset"]:
                    sim.reset()
                    command.update(reset=False, until=0)
                if tick < command["until"]:
                    sim.drive(command["linear"], command["angular"])
                else:
                    sim.drive()
                for _ in range(10):
                    sim.step()
            viewer.sync()
            time.sleep(max(0, 0.02 - (time.monotonic() - tick)))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="GUIなしで移動・旋回・描画を検証")
    parser.add_argument("--output", type=Path, default=Path("outputs/front.png"))
    parser.add_argument("--target", choices=["red", "green", "blue"], help="色認識で目標に自動接近")
    parser.add_argument("--headless", action="store_true", help="自動接近を画面なしで実行")
    parser.add_argument("--check-autonomy", action="store_true", help="3色の探索・接近を検証")
    parser.add_argument("--instruction", help="日本語の指示を計画して実行")
    parser.add_argument("--planner", choices=["llm", "mock"], default="llm")
    parser.add_argument("--plan-only", action="store_true", help="計画を表示して動作前に終了")
    parser.add_argument("--record", type=Path, metavar="MP4", help="指示・計画・動作をMP4に保存（--instruction と使用）")
    args = parser.parse_args()
    if args.record is not None:
        if not args.instruction or args.plan_only or args.check or args.check_autonomy or args.target:
            parser.error("--record は --instruction と使用し、--plan-only や検証モードと併用できません")
        if args.record.suffix.lower() != ".mp4" or args.record.exists():
            parser.error("録画先には存在しない .mp4 ファイルを指定してください")
    if args.instruction is not None:
        import json
        from planner import make_plan
        from executor import execute_plan
        try:
            plan = make_plan(args.instruction, args.planner)
            print(f"Planner: {args.planner}" + (" (AI未使用の確認モード)" if args.planner == "mock" else ""))
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            if not args.plan_only:
                result = execute_plan(plan, headless=args.headless, record=args.record,
                                      instruction=args.instruction, provider=args.planner)
                if result["status"] != "completed":
                    raise SystemExit(1)
        except (ValueError, RuntimeError) as error:
            parser.exit(1, f"{error}\n")
    elif args.check_autonomy:
        from autonomy import check_autonomy
        check_autonomy(Path("outputs"))
    elif args.target:
        from autonomy import approach
        result = approach(args.target, headless=args.headless)
        if result["status"] != "reached":
            raise SystemExit(1)
    elif args.check:
        smoke_check(args.output)
    else:
        interactive()
