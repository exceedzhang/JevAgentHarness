"""27B validity judge (``DECISION_DATA_V2`` §13.12) as a reusable module.

Ported from ``research/scripts/boundary_judge.py``. For a boundary case it asks a 27B judge a PURE
VALIDITY question -- is the action the agent actually chose a valid / reasonable / non-contradictory
action in this situation? -- NOT "is it gold". Output: VALID / INVALID / UNCERTAIN + one-line reason.

Layer 3 of the eval design: the LLM-free rule-based ``asym_valid`` (``scoring.py``) is the primary,
always-on metric; this judge is a slower periodic spot-check that bounds it (v8: 62.5% rule-based vs
56.4% strict label-match vs 68.1% judge "reasonable-action" rate). Only the pure logic is exercised by
the unit tests; ``judge_case`` / ``call_judge`` hit the network (vllm :8000 by default).
"""
from __future__ import annotations

import json
import re
import urllib.request

REPLY = "reply_without_tool"
DEFAULT_API = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


def clip_state(state: str, head: int = 700, tail: int = 7000) -> str:
    """Keep the task head and the latest observation; elide the middle of a long state."""
    s = str(state).strip()
    if len(s) <= head + tail + 20:
        return s
    return s[:head] + "\n...[middle elided]...\n" + s[-tail:]


def describe_choice(pred: str) -> str:
    if pred == REPLY:
        return "a FINAL TEXT REPLY to the user (it chose to respond without calling any tool)"
    return f"calling the tool/action: {pred}"


def build_prompt(case: dict) -> str:
    """``case`` needs ``state`` (situation text), ``opts`` (option names) and ``pred`` (chosen name)."""
    lines = [
        "You are an expert auditor of an AI agent that picks exactly one action per decision point from a fixed list.",
        "Below is the exact situation: the task, the steps already taken with their results, the available actions, and the action the agent actually chose.",
        "YOUR ONLY JOB: judge whether the CHOSEN action is a VALID, REASONABLE action in this situation.",
        "  - VALID = a competent agent could plausibly and defensibly take it here; it is consistent with the task and the evidence gathered so far, and is not self-contradictory. It does NOT have to be the single best action; several actions may be valid at once.",
        "  - INVALID = clearly a mistake given the context: targets a wrong or nonexistent file, uses an irrelevant tool, repeats an action that already failed, or (for a final reply) the agent claims the task is done / gives an answer when the required information or work is plainly missing.",
        "  - UNCERTAIN = the context does not give enough signal to call it.",
        "Do NOT judge which action is better, and do NOT check whether it matches any reference answer. Judge only the validity of the chosen action itself.",
        "",
        "SITUATION:",
        clip_state(case["state"]),
        "",
        "AVAILABLE ACTIONS:",
    ]
    for i, o in enumerate(case["opts"]):
        lines.append(f"  {chr(65 + (i % 26))}{'' if i < 26 else i // 26}. {o}")
    lines.append("")
    lines.append(f"The agent CHOSE: {describe_choice(case['pred'])}")
    lines.append("")
    lines.append("Answer: first line exactly one of VALID / INVALID / UNCERTAIN, then a single short reason.")
    return "\n".join(lines)


def verdict(txt: str) -> str:
    """Parse the leading VALID / INVALID / UNCERTAIN word (case-insensitive)."""
    m = re.search(r"\b(VALID|INVALID|UNCERTAIN)\b", txt, re.I)
    return m.group(1).upper() if m else "PARSE_ERR"


def call_judge(prompt: str, api: str = DEFAULT_API, model: str = DEFAULT_MODEL,
               key: str = "sk-1234", timeout: int = 180) -> str:
    """POST the prompt to the judge (OpenAI-compatible chat completions) and return its text."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(api, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        m = json.loads(resp.read())["choices"][0]["message"]
    return (m.get("content") or "") or (m.get("reasoning_content") or "")


def judge_case(case: dict, **kw) -> dict:
    """Judge one case -> ``{"verdict": VALID|INVALID|UNCERTAIN|PARSE_ERR|ERR, "reason": str}``."""
    try:
        ans = call_judge(build_prompt(case), **kw)
        return {"verdict": verdict(ans), "reason": ans.strip()[:300]}
    except Exception as e:  # network / parse error: surface, don't crash the whole eval
        return {"verdict": "ERR", "reason": f"ERR {e}"}


def summarize(results: list[dict]) -> dict:
    """Aggregate the VALID fraction overall and by direction (``reply_to_tool`` / ``tool_to_reply``)."""
    from collections import defaultdict

    def agg(sel: list[dict]) -> dict:
        n = len(sel)
        if not n:
            return {"n": 0}
        c: defaultdict[str, int] = defaultdict(int)
        for r in sel:
            c[r["verdict"]] += 1
        return {"n": n, **{k: round(v / n, 3) for k, v in c.items()}, "raw": dict(c)}

    return {
        "all": agg(results),
        "reply_to_tool": agg([r for r in results if r.get("direction") == "reply_to_tool"]),
        "tool_to_reply": agg([r for r in results if r.get("direction") == "tool_to_reply"]),
    }