"""Search and approach using only RGB and depth from the forward camera."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from perception import detect_color
from sim import Simulation


def approach(color, *, headless=False, timeout=45.0, output=Path("outputs"), sim=None, search_only=False, recorder=None):
    sim = sim if sim is not None else Simulation()
    sim.drive()
    for index in range(250):
        if recorder is not None and index % 50 == 0:
            recorder.capture("姿勢安定")
        sim.step()
    records = []
    status = "timeout"
    reached_frames = 0
    start_time = sim.data.time
    viewer = None
    paused = False

    def on_key(key):
        nonlocal paused
        if key == 32:
            paused = not paused

    output.mkdir(parents=True, exist_ok=True)
    try:
        if not headless:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key)
            viewer.cam.lookat[:] = [0.5, 0, 0]
            viewer.cam.distance = 8
            viewer.cam.elevation = -45
            print("Space: 一時停止／再開。ウィンドウを閉じると中止。")
        with mujoco.Renderer(sim.model, height=240, width=320) as renderer:
            while sim.data.time - start_time < timeout:
                tick = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    status = "cancelled"
                    break
                if paused:
                    sim.drive()
                    viewer.sync()
                    time.sleep(0.1)
                    continue
                renderer.update_scene(sim.data, camera="front")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                detection = detect_color(rgb, depth, color)
                if not records:
                    Image.fromarray(rgb).save(output / f"{color}_start.png")
                if detection is None:
                    reached_frames = 0
                    linear, angular, state = 0.0, 0.65, "search"
                else:
                    if search_only:
                        status = "found"
                        sim.drive()
                        records.append({"time": float(sim.data.time - start_time), "state": "found",
                                        "detection": asdict(detection)})
                        if recorder is not None:
                            recorder.capture("目標発見")
                        break
                    error = detection.center_x
                    angular = float(np.clip(-error * 1.8, -0.8, 0.8))
                    linear = min(0.35, max(0.0, (detection.distance - 0.6) * 0.6))
                    if abs(error) > 0.3:
                        linear = 0.0
                    state = "approach"
                    if detection.distance <= 0.65 and abs(error) < 0.12:
                        linear, angular = 0.0, 0.0
                        reached_frames += 1
                    else:
                        reached_frames = 0
                    if reached_frames >= 5:
                        status = "reached"
                sim.drive(linear, angular)
                records.append({"time": float(sim.data.time - start_time), "state": state,
                                "detection": asdict(detection) if detection else None})
                if recorder is not None:
                    recorder.capture("探索中" if state == "search" else "接近中")
                if status == "reached":
                    break
                for _ in range(50):
                    sim.step()
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0.0, 0.1 - (time.monotonic() - tick)))
            sim.drive()
            # Allow the physical wheels to brake before reporting the final state.
            for index in range(250):
                if recorder is not None and index % 50 == 0:
                    recorder.capture("ブレーキ")
                sim.step()
            renderer.update_scene(sim.data, camera="front")
            Image.fromarray(renderer.render()).save(output / f"{color}_finish.png")
        result = {"color": color, "status": status, "elapsed": float(sim.data.time - start_time),
                  "final_position": sim.data.qpos[:3].tolist(), "frames": records}
        result["operation"] = "search" if search_only else "approach"
        (output / f"{color}_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{color}: {status}, {result['elapsed']:.1f} s")
        return result
    finally:
        sim.drive()
        if viewer is not None:
            viewer.close()


def check_autonomy(output):
    # Ground truth is used only for verification, never by the controller.
    positions = {"red": (2, 0), "green": (1, 2), "blue": (1, -2)}
    for color, target in positions.items():
        result = approach(color, headless=True, output=output)
        assert result["status"] == "reached", result["status"]
        distance = np.linalg.norm(np.array(result["final_position"][:2]) - target)
        assert 0.65 < distance < 1.25, f"{color}: unsafe or distant stop ({distance:.3f} m)"
        if color != "red":
            assert any(frame["state"] == "search" for frame in result["frames"]), "Search not exercised"
        print(f"PASS: {color}, target center distance={distance:.3f} m")
    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    assert detect_color(blank, np.ones((20, 20)), "red") is None
    print("PASS: no target detected in blank image")
