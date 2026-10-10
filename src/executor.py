import json
from pathlib import Path

from autonomy import approach
from planner import validate_plan
from sim import Simulation


def execute_plan(plan, *, headless=False, output=Path("outputs/instruction"), record=None, instruction="", provider="mock"):
    plan = validate_plan(plan)
    sim = Simulation()
    results = []
    status = "completed"
    output.mkdir(parents=True, exist_ok=True)
    recorder = None
    try:
        if record is not None:
            from recording import VideoRecorder
            recorder = VideoRecorder(sim, record, instruction, plan, provider)
            recorder.card("指示と計画", seconds=5)
        for index, action in enumerate(plan["actions"]):
            print(f"{index + 1}: {action}")
            if recorder is not None:
                recorder.set_step(index, f"実行中：{index + 1}/{len(plan['actions'])}")
            if action["action"] == "stop":
                sim.drive()
                results.append({"action": action, "status": "stopped"})
                break
            result = approach(action["color"], sim=sim, headless=headless,
                              search_only=action["action"] == "search",
                              output=output / f"{index + 1:02d}_{action['action']}", recorder=recorder)
            results.append({"action": action, "status": result["status"]})
            expected = "found" if action["action"] == "search" else "reached"
            if result["status"] != expected:
                status = result["status"]
                break
    except BaseException:
        status = "error"
        raise
    finally:
        sim.drive()
        summary = {"status": status, "plan": plan, "results": results}
        if recorder is not None:
            try:
                recorder.card("完了" if status == "completed" else f"中止：{status}", seconds=5)
                summary["video"] = {"path": str(record), "seconds": recorder.duration, "fps": recorder.fps}
            finally:
                recorder.close()
            print(f"MP4: {record} ({recorder.duration:.1f} s)")
        (output / "execution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
