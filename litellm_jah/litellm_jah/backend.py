"""Decision backends: the small, fast "System 1" that answers one typed multiple-choice question per agent step.

Ported from ``jevagentharness.decision`` (MIT). Backends:

``jev`` any server that speaks the Jev System-One wire format (``POST {base}/v1/systemone``): TypeSafe's hosted
Jev, Jev through OpenRouter, or a self-hosted decision model. The answer is a probability per option.
``rules`` no model at all: transparent heuristics over the agent's last tool result (errors, failing tests,
repeated actions). Useful as a baseline and when no decision-model key is available.
``none`` never answers (the router then uses its ``on_error`` tier).

Failures raise; the router (:mod:`litellm_jah.engine`) turns them into its fail-safe tier, so a
decision-model outage never stops the agent.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from aiohttp import ClientSession, ClientTimeout

# 499: OpenRouter's "internal_error" for a request the provider could not complete (about 1 decision call in
# 5,000 on Terminal-Bench); a retry usually succeeds, where the fail-safe would send the step to the strong tier
RETRY_STATUS = {408, 429, 499, 500, 502, 503, 504, 529}

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "model": "typesafe/jev-1.13"},
    "typesafe": {"base_url": "https://api.typesafe.ai", "key_env": "TYPESAFE_API_KEY", "model": "jev-latest"},
    "local": {"base_url": "http://localhost:8090", "key_env": "", "model": "agentdecision-4b"},
}
# Jev bills input tokens only; OpenRouter reports the billed amount itself (usage.cost), so this rate is a fallback
JEV_PRICE_PER_M_INPUT = 0.10


@dataclass
class Question:
    id: str
    text: str
    options: list[str]
    descriptions: list[str] = field(default_factory=list)

    def wire(self) -> dict[str, Any]:
        descs = self.descriptions or [None] * len(self.options)
        return {"type": "choice", "instructions": self.text,
                "criteria": {o: (d or None) for o, d in zip(self.options, descs)}}


@dataclass
class Decision:
    p: dict[str, float]
    model: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0
    input_tokens: int = 0
    why: str = ""
    calibrated: bool = False  # p lives in the model card's calibrated space (JevBackend.calibration applied)

    @property
    def choice(self) -> str:
        return max(self.p, key=lambda k: self.p[k]) if self.p else ""


class DecisionBackend:
    name = "base"

    async def decide(self, state: str, q: Question) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}


CALIBRATION_KEYS = frozenset({f"choice:{b}" for b in ("2", "3-5", "6-16", "17-52")} | {"noul:2"})


def _bucket(n: int) -> str:
    """The model card's bucket for a question with n options (its README's ``bucket()``)."""
    return "2" if n == 2 else "3-5" if n <= 5 else "6-16" if n <= 16 else "17-52"


def _check_calibration(cal: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate ``decision.calibration``: the model card's calibration.json (a temperature per
    ``<kind>:<bucket>``, ``global`` the fallback) written inline. Our servers answer raw T=1 probabilities,
    so the plugin applies the temperatures itself (``_parse``). Fail-closed: an invalid block raises at
    startup rather than being silently ignored, because silently skipping calibration would re-introduce
    the misread taus it is there to prevent."""
    if cal is None:
        return None
    if not isinstance(cal, Mapping):
        raise ValueError("decision.calibration must be a mapping {global: T, buckets: {key: T}}")
    try:
        g = float(cal["global"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("decision.calibration.global must be the global temperature") from None
    if not math.isfinite(g) or g <= 0:
        raise ValueError("decision.calibration.global must be a finite temperature greater than 0")
    buckets: dict[str, float] = {}
    for k, v in (cal.get("buckets") or {}).items():
        if k not in CALIBRATION_KEYS:
            raise ValueError(f"unknown calibration bucket {k!r}; allowed: {sorted(CALIBRATION_KEYS)}")
        try:
            t = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"calibration bucket {k} must be a finite number") from None
        if not math.isfinite(t) or t <= 0:
            raise ValueError(f"calibration bucket {k} must be a finite temperature greater than 0")
        buckets[k] = t
    return {"global": g, "buckets": buckets}


class JevBackend(DecisionBackend):
    name = "jev"

    def __init__(self, provider: str = "openrouter", model: str | None = None, base_url: str | None = None,
                 key_env: str | None = None, timeout_s: float = 8.0, retries: int = 2,
                 price_per_m: float | None = None, calibration: Mapping[str, Any] | None = None) -> None:
        price = JEV_PRICE_PER_M_INPUT if price_per_m is None else float(price_per_m)
        if not math.isfinite(price) or price < 0:
            raise ValueError("price_per_m must be a non-negative finite number")
        self.price_per_m = price
        self.calibration = _check_calibration(calibration)
        pre = PROVIDERS.get(provider, PROVIDERS["openrouter"])
        self.provider = provider
        self.model = model or pre["model"]
        self.base_url = (base_url or pre["base_url"]).rstrip("/")
        self.key_env = key_env if key_env is not None else pre["key_env"]
        self.timeout_s = timeout_s
        self.retries = retries
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # a session bound to a closed loop must not be reused
    def _key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    async def _client(self) -> ClientSession:
        loop = asyncio.get_running_loop()
        if self._session is None or self._session.closed or self._loop is not loop:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = ClientSession(timeout=ClientTimeout(total=self.timeout_s), trust_env=True)
            self._loop = loop
        return self._session

    async def decide(self, state: str, q: Question) -> Decision:
        body = {"state": state, "model": self.model, "questions": {q.id: q.wire()}}
        headers = {"Content-Type": "application/json"}
        if self._key():
            headers["Authorization"] = f"Bearer {self._key()}"
        delay = 0.4
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                cl = await self._client()
                async with cl.post(f"{self.base_url}/v1/systemone", data=json.dumps(body), headers=headers) as r:
                    text = await r.text()
                    if r.status == 200:
                        data = json.loads(text)
                        return self._parse(data, q, time.monotonic() - t0)
                    last = f"HTTP {r.status}: {text[:200]}"
                    if r.status not in RETRY_STATUS:
                        break
            except (asyncio.TimeoutError, OSError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"decision model failed: {last}")

    def _parse(self, data: Mapping[str, Any], q: Question, latency: float) -> Decision:
        a = (data.get("answers") or {}).get(q.id) or {}
        probs = a.get("probabilities") or ({a["choice"]: 1.0} if "choice" in a else {})
        p = {o: float(probs.get(o, 0.0)) for o in q.options}
        s = sum(p.values())
        if s <= 0:
            raise RuntimeError(f"decision model returned no distribution: {str(a)[:200]}")
        p = {k: v / s for k, v in p.items()}
        calibrated = False
        if self.calibration is not None:  # the card's temperature, as p' ∝ p^(1/T): equivalent to softmax(logits/T)
            t = self.calibration["buckets"].get(f"choice:{_bucket(len(p))}", self.calibration["global"])
            e = 1.0 / t
            q = {k: v ** e for k, v in p.items()}
            s = sum(q.values())  # p summed to 1 and e > 0, so s > 0
            p = {k: v / s for k, v in q.items()}
            calibrated = True
        u = data.get("usage") or {}
        tokens = int(u.get("input_tokens") or 0)
        cost = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else tokens * self.price_per_m / 1e6
        return Decision(p=p, model=str(data.get("model") or self.model), cost_usd=cost, latency_s=latency,
                        input_tokens=tokens, calibrated=calibrated)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "provider": self.provider, "model": self.model, "base_url": self.base_url,
                "calibrated": self.calibration is not None}


# ------------------------------------------------------------------------------------------------ rules (no model)
ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)", r"\b[A-Z][A-Za-z]*Error\b", r"\berror:", r"\bERROR\b", r"\bFAILED\b",
    r"\bfailed\b", r"command not found", r"No such file or directory", r"Permission denied", r"Segmentation fault",
    r"exit(?:ed)? (?:with )?(?:code|status) [1-9]", r"returncode[\"']?\s*[:=]\s*[1-9]", r"\bpanic(?:ked)?\b",
    r"\bfatal\b", r"Syntax ?Error", r"assert(?:ion)? ?(?:error|failed)", r"\b\d+ failed\b",
]
_ERR = re.compile("|".join(ERROR_PATTERNS))


@dataclass
class StepView:
    """What the rules backend looks at: the last action and its result, the action before it, and (for the action
    question) whether the proposed call matched a risky pattern of the action pre-filter."""
    last_action: str = ""
    last_output: str = ""
    prev_action: str = ""
    risky: bool = False


# questions of the other decision points (the standalone proxy's points.py): they need a model, not a rule
_NOT_RULES = {"next_tool", "task_done", "already_done", "fast_step_ok"}


class RulesBackend(DecisionBackend):
    """No model: ``hard`` (routing) and ``yes`` (reflection) when the last tool result shows an error or failing
    tests, or the agent repeats its last action. Other questions (tool choice, the finish check) need a decision
    model: the rules backend refuses them, so those decision points fall back to their LLM-only behaviour."""
    name = "rules"
    ANSWERS = ("next_step", "needs_reflection", "action_risk")

    def __init__(self, hard_option: str = "hard", easy_option: str = "routine") -> None:
        self.hard, self.easy = hard_option, easy_option

    def judge(self, v: StepView) -> tuple[float, str]:
        if v.last_action and v.last_action == v.prev_action:
            return 0.9, "repeated_action"
        if _ERR.search(v.last_output[-6000:] if v.last_output else ""):
            return 0.9, "error_signal"
        return 0.1, "no_signal"

    async def decide(self, state: str, q: Question, view: StepView | None = None) -> Decision:
        v = view or StepView()
        if q.id == "action_risk":
            p = 0.9 if v.risky else 0.1
            return Decision(p={"risky": p, "safe": 1.0 - p}, model="rules",
                            why="risky_pattern" if v.risky else "no_pattern")
        if q.id == "needs_reflection":
            p, why = self.judge(v)
            return Decision(p={"yes": p, "no": 1.0 - p}, model="rules", why=why)
        if q.id in _NOT_RULES:
            raise RuntimeError(f"the rules backend cannot answer the {q.id!r} question; use a decision model")
        p_hard, why = self.judge(v)
        return Decision(p={self.hard: p_hard, self.easy: 1.0 - p_hard}, model="rules", why=why)


class NoneBackend(DecisionBackend):
    name = "none"

    async def decide(self, state: str, q: Question) -> Decision:
        raise RuntimeError("no decision backend configured")


def make_backend(spec: Mapping[str, Any] | None) -> DecisionBackend:
    """Build the decision backend from the ``backend`` block; both a missing block and ``{}`` default to the
    rules backend (no model needed)."""
    spec = dict(spec or {})
    kind = spec.pop("type", "rules")
    if kind == "jev":
        allowed = {"provider", "model", "base_url", "key_env", "timeout_s", "retries",
                   "price_per_m", "calibration"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"unknown decision backend options {sorted(unknown)}; allowed: {sorted(allowed)}")
        return JevBackend(**spec)
    if kind == "rules":
        allowed = {"hard_option", "easy_option"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"unknown decision backend options {sorted(unknown)}; allowed: {sorted(allowed)}")
        return RulesBackend(**spec)
    if kind == "none":
        if spec:
            raise ValueError(f"the none backend takes no options (got {sorted(spec)})")
        return NoneBackend()
    raise ValueError(f"unknown decision backend {kind!r} (jev | rules | none)")
