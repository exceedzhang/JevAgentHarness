"""The state a routing decision is made on: the task and the recent steps.

Ported from ``jevagentharness.router``. The question text, the layout and the truncation limits are the exact
format the decision models were trained on: do not change them, or the model's calibration drifts with it.
Messages are plain OpenAI chat-completions dicts (LiteLLM's ``structured_messages`` shape).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .backend import Question, StepView

QUESTION = Question(
    id="next_step",
    text=("You are watching an AI agent work on the task below, one tool call at a time. Read its latest action and "
          "the result. Decide how hard the agent's NEXT step is. 'routine': the next step follows directly from what "
          "the agent already knows, e.g. read or search more, run a command or test it planned, make an edit it "
          "already worked out. 'hard': the next step needs careful reasoning, e.g. something failed or behaved "
          "unexpectedly, a plan or a fix must be worked out, the agent seems stuck or repeats itself, or it must "
          "judge whether the task is really done."),
    options=["routine", "hard"],
    descriptions=["The next step is routine: a faster, cheaper model will do it as well as a strong one",
                  "The next step needs careful reasoning: debugging, planning, recovering from an error, or "
                  "verifying the result"],
)
HARD = "hard"

TASK_CHARS = 3000
HISTORY_STEPS = 6
ACTION_CHARS = 400
OUTPUT_CHARS = 300
LAST_OUTPUT_CHARS = 2500


def _trunc(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    if n < 11: # too small for the head + " ... " + tail layout (the tail needs one char)
        return s[:n]
    return s[: n // 2] + " ... " + s[-(n - n // 2 - 5):]


def _tail(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "... " + s[-n:]


def _msg_text(m: Mapping[str, Any]) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(c)


@dataclass
class Step:
    action: str
    output: str


def episode(messages: list[Mapping[str, Any]]) -> tuple[str, list[Step]]:
    """(task, steps): the first user message WITH text, then one step per assistant turn with the results
    after it (the standalone proxy's episode)."""
    task = ""
    for m in messages:
        if m.get("role") == "user":
            t = _msg_text(m)
            if t:
                task = t
                break
    steps: list[Step] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if tcs:
            action = "; ".join(f"{(c.get('function') or {}).get('name', '?')}"
                               f"({(c.get('function') or {}).get('arguments', '')})" for c in tcs)
        else:
            action = f"message: {_msg_text(m)}"
        outs: list[str] = []
        for n in messages[i + 1:]:
            r = n.get("role")
            if r == "assistant":
                break
            if r in ("tool", "user"):  # harnesses that send observations as user messages
                t = _msg_text(n)
                if t:
                    outs.append(t)
        steps.append(Step(action=action, output="\n".join(outs)))
    return task, steps


def step_index(messages: list[Mapping[str, Any]]) -> int:
    """The step number of this request: the number of assistant turns already in the history."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def render_state(messages: list[Mapping[str, Any]]) -> tuple[str, StepView]:
    task, steps = episode(messages)
    lines = [f"TASK: {_trunc(task, TASK_CHARS)}", "", f"STEPS SO FAR: {len(steps)}"]
    recent = steps[-HISTORY_STEPS:]
    first = len(steps) - len(recent) + 1
    for j, s in enumerate(recent):
        last = j == len(recent) - 1
        lines.append(f"[{first + j}] ACTION: {_trunc(s.action, ACTION_CHARS)}")
        lines.append(f"    RESULT: {_tail(s.output, LAST_OUTPUT_CHARS) if last else _trunc(s.output, OUTPUT_CHARS)}")
    view = StepView(last_action=steps[-1].action if steps else "", last_output=steps[-1].output if steps else "",
                    prev_action=steps[-2].action if len(steps) > 1 else "")
    return "\n".join(lines), view


def session_key(messages: list[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None) -> str:
    """The identity of the agent run: an explicit id from the request, else a hash of the system prompt and the
    task (the standalone proxy's fallback, so the same conversation gets the same session in both)."""
    md = metadata or {}
    explicit = md.get("jah_session") or md.get("session_id") or md.get("litellm_session_id")
    if explicit:
        return str(explicit)
    task, _ = episode(messages)
    system = "\n\n".join(t for m in messages if m.get("role") == "system" and (t := _msg_text(m)))
    h = hashlib.sha256((system[:4000] + "\x00" + task[:8000]).encode("utf-8", "replace")).hexdigest()
    return h[:16]
