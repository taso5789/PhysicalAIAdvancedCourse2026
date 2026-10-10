import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planner import llm_plan, make_plan, validate_plan
from executor import execute_plan


class PlanningTests(unittest.TestCase):
    def test_invalid_plans_are_rejected(self):
        for plan in [None, {"actions": []}, {"actions": [{"action": "approach", "color": "yellow"}]},
                     {"actions": [{"action": "stop"}, {"action": "stop"}]},
                     {"actions": [{"action": "approach", "color": "red"}]},
                     {"actions": [{"action": "stop", "code": "anything"}]}]:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_mock_preserves_instruction_order(self):
        plan = make_plan("青い球を見てから赤い箱へ行って", "mock")
        self.assertEqual([a["color"] for a in plan["actions"] if a["action"] == "approach"], ["blue", "red"])

    def test_llm_tool_response_validation_without_network(self):
        response = SimpleNamespace(stop_reason="tool_use", content=[
            SimpleNamespace(type="tool_use", name="submit_plan", input={"actions": [{"action": "stop"}]})])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-placeholder", "ANTHROPIC_MODEL": ""}), \
                patch("dotenv.load_dotenv"), patch("anthropic.Anthropic") as client:
            api = client.return_value.__enter__.return_value
            api.beta.messages.create.return_value = response
            self.assertEqual(llm_plan("停止"), {"actions": [{"action": "stop"}]})
            request = api.beta.messages.create.call_args.kwargs
            self.assertEqual(request["model"], "claude-opus-5-5")
            # Current models return 400 for forced tool_choice ("tool"/"any").
            self.assertEqual(request["tool_choice"]["type"], "auto")
            self.assertTrue(request["tools"][0]["strict"])
            response.stop_reason = "max_tokens"
            with self.assertRaises(ValueError):
                llm_plan("停止")

    def test_executor_keeps_same_simulation_and_aborts_on_failure(self):
        plan = make_plan("赤のあと緑", "mock")
        sim = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch("executor.Simulation", return_value=sim), \
                patch("executor.approach", side_effect=[{"status": "found"}, {"status": "timeout"}]) as run:
            result = execute_plan(plan, headless=True, output=Path(directory))
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.kwargs["sim"] is sim for call in run.call_args_list))
            sim.drive.assert_called()


if __name__ == "__main__":
    unittest.main()
