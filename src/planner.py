"""Provider boundary: one LLM call creates a validated, bounded action plan."""
import os
import re
from pathlib import Path

COLORS = ["red", "green", "blue"]
ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"action": {"type": "string", "enum": ["search", "approach", "stop"]},
                   "color": {"type": "string", "enum": COLORS}},
    "required": ["action"],
}
DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept server-side refusal fallbacks in their "default" routing form.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
# Strict tool schemas reject array length constraints; validate_plan enforces 1-8 actions.
PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Always call this once to submit the robot plan for the user's instruction. "
                   "It takes an ordered action plan, not direct motor commands. "
                   "search rotates until the requested color is visible; approach searches if needed and "
                   "drives to that color. stop ends the plan. Only red, green, and blue targets exist. "
                   "Do not invent support for grasping, obstacle avoidance, or other targets.",
    "strict": True,
    "input_schema": {"type": "object", "additionalProperties": False,
                     "properties": {"actions": {"type": "array", "items": ACTION_SCHEMA}},
                     "required": ["actions"]},
}


def validate_plan(value):
    if not isinstance(value, dict) or set(value) != {"actions"}:
        raise ValueError("計画には actions のみ指定できます")
    actions = value["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        raise ValueError("動作数は1〜8個です")
    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            raise ValueError("動作はオブジェクトで指定します")
        action = item.get("action")
        if action == "stop":
            if set(item) != {"action"} or index != len(actions) - 1:
                raise ValueError("stop は最後に、色を付けず指定します")
        elif action in ("search", "approach"):
            if set(item) != {"action", "color"} or item["color"] not in COLORS:
                raise ValueError("search/approach には red/green/blue を指定します")
        else:
            raise ValueError("未対応の動作です")
    if actions[-1]["action"] != "stop":
        raise ValueError("計画は stop で終了してください")
    return {"actions": [dict(item) for item in actions]}


def mock_plan(instruction):
    # Rule-based stand-in for testing without the API (not AI).
    if instruction == "停止":
        return validate_plan({"actions": [{"action": "stop"}]})
    matches = re.findall("赤|緑|青", instruction)
    if not matches:
        raise ValueError("mock モードでは色（赤・緑・青）を明示してください")
    colors = {"赤": "red", "緑": "green", "青": "blue"}
    actions = []
    for match in matches:
        actions.extend([{"action": "search", "color": colors[match]},
                        {"action": "approach", "color": colors[match]}])
    actions.append({"action": "stop"})
    return validate_plan({"actions": actions})


def llm_plan(instruction):
    from dotenv import load_dotenv
    import anthropic

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL
    if not key:
        raise ValueError(".env に ANTHROPIC_API_KEY を設定してください")
    # Current models reject forced tool_choice, so use auto and require the call in the prompt.
    request = dict(
        model=model, max_tokens=16000, output_config={"effort": "low"},
        system="日本語の指示を実行可能なロボット計画に変換し、必ず submit_plan を1回だけ呼び出してください。"
               "各目標に search と approach を順に使い、最後は stop。最大8動作。"
               "赤い箱=red、緑の箱=green、青い球=blue。"
               "未対応の指示や曖昧な指示は stop のみとし、動かさないでください。",
        tools=[PLAN_TOOL], tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=[{"role": "user", "content": instruction}],
    )
    if model in FALLBACK_MODELS:
        request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    with anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=2) as client:
        try:
            response = client.beta.messages.create(**request)
        except anthropic.APIError as error:
            # Avoid dumping request headers, credentials or request bodies.
            raise RuntimeError(f"LLM API 呼び出し失敗（{type(error).__name__}）") from None
    if response.stop_reason != "tool_use":
        raise ValueError("LLM の計画が未完了または拒否されました")
    calls = [block for block in response.content if block.type == "tool_use"]
    if len(calls) != 1 or calls[0].name != "submit_plan":
        raise ValueError("LLM が所定の計画を返しませんでした")
    return validate_plan(calls[0].input)


def make_plan(instruction, provider):
    if not instruction.strip() or len(instruction) > 2000:
        raise ValueError("指示は1〜2000文字にしてください")
    if provider == "mock":
        return mock_plan(instruction)
    if provider == "llm":
        return llm_plan(instruction)
    raise ValueError("未対応の planner です")
