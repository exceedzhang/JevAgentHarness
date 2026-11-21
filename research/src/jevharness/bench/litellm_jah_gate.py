"""Research-side adapter: run the REAL ``litellm_jah`` ``JahPostCall`` action guard (note + redo) on an
about-to-run irreversible action, using the 27B worker as BOTH the action-risk decision backend and the
strong redo model.

This is the in-loop stand-in for the standalone proxy: instead of LiteLLM routing every worker call through
the plugin, the harness calls the hook's ``_action_check`` directly on the one action it is about to execute.
The decision question (``action_risk``: safe/risky) is answered by the 27B over an OpenAI-compatible chat
completion (thinking disabled for speed); the strong redo (re-calling the worker with the force note) is done
by the harness itself, so the worker genuinely re-decides rather than being overridden.

No ``litellm`` install is required: ``litellm_jah`` imports through its try/except stand-in, and the decision
call uses ``aiohttp`` directly (available in the research venv). The guard is fail-open: any decision-model
failure or unexpected answer returns ``None`` (the action proceeds untouched), exactly as ``litellm_jah`` falls
back to the LLM-only behaviour on a decision-model error.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import aiohttp

from litellm_jah.backend import Decision, DecisionBackend
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter
from litellm_jah.postcall import JahPostCall
from litellm_jah.points import _count_stonewalls as _lj_count_stonewalls
from litellm_jah.state import render_state

_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple, JahPostCall] = {}  # (base_url, model, tau) -> shared JahPostCall

# The litellm_jah action question is written for general (coding) agents, where a dangerous call is `rm -rf`
# or `git push --force`. In this customer-support bench the irreversible actions are business operations
# (ship_order / refund / cancel_order) that a general model reads as "ordinary", so it under-flags them. The
# domain framing below keeps the 27B the decision model but anchors the litellm_jah question's own "irreversible
# business operation = risky" intent to the customer-support domain: such an action is risky exactly when the
# customer has not confirmed the value it needs (deflected / "I already told you"), and safe once confirmed.
_DOMAIN = (
    "This is a customer-support task. The tools ship_order, refund and cancel_order are IRREVERSIBLE business "
    "operations: they change a real customer order (ship goods, issue a refund, cancel an order) and cannot be "
    "undone without contacting the customer. Such an action is 'risky' when the specific value it depends on "
    "(the shipping speed, the refund method, or which order to act on) has not been explicitly confirmed by the "
    "customer (the customer deflected, said they already told you, or did not provide it). It is 'safe' only "
    "when the customer explicitly confirmed the needed value."
)


class _OpenAICompatActionBackend(DecisionBackend):
    """The 27B as the action-risk decision backend (the ``safe`` / ``risky`` question)."""

    name = "openai_compat_action"

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def decide(self, state: str, q: Any) -> Decision:
        prompt = (q.text + "\n\n" + _DOMAIN + "\n\n" + state +
                  "\n\nRespond with exactly one word: " + " or ".join(f'"{o}"' for o in q.options) + ".")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
            # Qwen3 reasoning model: disable thinking so the answer is direct (0 reasoning tokens) and fast
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last = ""
        for _ in range(3):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as s:
                    async with s.post(f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
                        text = await r.text()
                        if r.status == 200:
                            data = json.loads(text)
                            ans = (data["choices"][0]["message"].get("content") or "").strip().lower()
                            if not ans:  # a reasoning overrun / empty answer: retry, then fail-open below
                                continue
                            risky = "risky" in ans
                            p = {"risky": 0.9, "safe": 0.1} if risky else {"safe": 0.9, "risky": 0.1}
                            return Decision(p=p, model=self.model, cost_usd=0.0)
                        last = f"HTTP {r.status}: {text[:200]}"
            except Exception as e:  # noqa: BLE001 - a flaky call is retried, then fail-open
                last = f"{type(e).__name__}: {e}"
        # fail-open: a decision-model failure falls back to "safe" (the LLM-only behaviour), as litellm_jah does
        return Decision(p={"safe": 0.9, "risky": 0.1}, model=self.model, cost_usd=0.0)

    async def close(self) -> None:
        return None


def _make_postcall(base_url: str, api_key: str, model: str, tau: float) -> JahPostCall:
    backend = _OpenAICompatActionBackend(base_url, api_key, model)
    cfg = JAHCfg(mode="cascade", strong=model, decision={"type": "none"},
                 decisions={"action": {"mode": "guard", "prefilter": "all", "tau": tau}})
    engine = JAHRouter(cfg, backend=backend)
    return JahPostCall(engine=engine, recall=None)


def get_postcall(base_url: str, api_key: str, model: str, tau: float = 0.3) -> JahPostCall:
    """The shared litellm_jah hook for the 27B (cached per base_url/model/tau)."""
    key = (base_url.rstrip("/"), model, tau)
    with _LOCK:
        pc = _ENGINE_CACHE.get(key)
        if pc is None:
            pc = _make_postcall(base_url, api_key, model, tau)
            _ENGINE_CACHE[key] = pc
        return pc


def lj_count_stonewalls(messages: list[dict[str, Any]]) -> int:
    """How many customer stonewalls are in the message history (the litellm_jah count; pure, no model call).
    Used as a cheap pre-filter so the 27B action-risk call is only paid when the customer has actually
    stonewalled; it is the same count the guard recomputes as its authoritative source."""
    try:
        return int(_lj_count_stonewalls(list(messages)))
    except Exception:  # noqa: BLE001
        return 0


def llm_conn(llm: Any) -> tuple[str, str, str] | None:
    """(base_url, api_key, model) of an OpenAI-compatible worker LLM client, or None if not one. Reads the
    connection off the client's underlying OpenAI client so the decision call targets the same 27B as the
    worker (the strong redo model of the litellm_jah guard)."""
    model = getattr(llm, "model", None)
    if not model:
        return None
    clients = getattr(llm, "clients", None)
    c = clients[0] if clients else getattr(llm, "client", None)
    if c is None:
        return None
    base_url = getattr(c, "base_url", None)
    api_key = getattr(c, "api_key", None)
    if not base_url or api_key is None:
        return None
    return str(base_url), str(api_key), str(model)


def _make_resp(name: str, args: Any) -> Any:
    """A minimal LiteLLM-style response object: ``choices[0].message.tool_calls = [one call]``."""
    a = args if isinstance(args, dict) else {}
    tc = {"type": "function", "function": {"name": name, "arguments": json.dumps(a)}}
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc], content=None))])


def lj_action_note(name: str, args: Any, messages: list[dict[str, Any]], base_url: str, api_key: str,
                   model: str, tau: float = 0.3) -> str | None:
    """Run the real litellm_jah action guard on an about-to-run irreversible action.

    Returns the force-field note (the NOTE the harness should append before re-deciding) when the guard would
    redo the step -- the 27B flagged the action risky AND >=2 customer stonewalls are in the message history --
    and ``None`` otherwise (the action proceeds untouched). The harness performs the strong redo itself.
    """
    pc = get_postcall(base_url, api_key, model, tau)
    resp = _make_resp(name, args)
    state, view = render_state(messages)

    async def _go() -> Any:
        return await pc._action_check(resp, "strong", state, view, messages)

    try:
        verdict = asyncio.run(_go())
    except Exception:  # noqa: BLE001 - the guard must never break the episode
        return None
    if verdict is not None and getattr(verdict, "redo", False) and getattr(verdict, "reason", None) == "action_guard":
        note = getattr(verdict, "note", None)
        # only the force-field note (the specific missing-field ask) fires the NOTE+REDO in the harness
        if note and "Pre-action force-field" in note:
            return note
    return None
