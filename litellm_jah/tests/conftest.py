"""Offline test helpers: a scripted fake decision backend (no keys, no network)."""

from __future__ import annotations

import asyncio

import pytest
from litellm_jah.backend import Decision, DecisionBackend, Question


def decision(p_hard: float) -> Decision:
    return Decision(p={"routine": round(1.0 - p_hard, 4), "hard": round(p_hard, 4)}, model="fake")


class FakeBackend(DecisionBackend):
    """``script(call_no, state, q) -> Decision`` (or raises); counts its calls."""
    name = "fake"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def decide(self, state: str, q: Question, view=None) -> Decision:
        self.calls += 1
        r = self.script(self.calls, state, q)
        if asyncio.iscoroutine(r):
            r = await r
        return r


def p_over(options: list[str], chosen: str, conf: float) -> dict[str, float]:
    """A probability mass over ``options``: ``conf`` on ``chosen``, the rest spread evenly."""
    rest = (1.0 - conf) / max(1, len(options) - 1)
    return {o: round(conf if o == chosen else rest, 6) for o in options}


class Scripted(DecisionBackend):
    """Answers each question by its id: ``{id: fn(q) -> Decision}`` (a missing id is a test bug)."""

    name = "scripted"

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[str] = []

    async def decide(self, state: str, q: Question, view=None) -> Decision:
        self.calls.append(q.id)
        fn = self.answers.get(q.id)
        if fn is None:
            raise AssertionError(f"no scripted answer for question {q.id!r}")
        d = fn(q)
        if asyncio.iscoroutine(d):
            d = await d
        return d


class Ctx:
    """Duck-type of LiteLLM's routing context."""

    def __init__(self, messages, metadata=None, candidate_models=None):
        self.raw_messages = list(messages)
        self.structured_messages = list(messages)
        self.metadata = dict(metadata or {})
        self.candidate_models = list(candidate_models or [])
        self.signals = {}


def msgs(steps: int, task: str = "Fix the bug", system: str = "You are a coding agent.") -> list[dict]:
    """A transcript of ``steps`` completed steps: system, task, then steps of (assistant tool call, tool result)."""
    out = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for i in range(steps):
        out.append({"role": "assistant", "content": None,
                    "tool_calls": [{"id": f"t{i}", "type": "function",
                                    "function": {"name": "bash", "arguments": f'{{"command": "ls -la {i}"}}'}}]})
        out.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"result {i}: all good"})
    return out


@pytest.fixture
def run():
    def _run(coro):
        return asyncio.run(coro)
    return _run
