"""The JAH routing decision point as a per-request engine: session state + decision model + policy.

Port of the routing half of ``jevagentharness`` (``router.Router`` and the routing call in
``server.Proxy.complete``): the same question, the same state rendering, the same policy and the same fail-safe
(a decision-model failure never stops the agent: the step goes to the ``on_error`` tier, strong by default).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .backend import DecisionBackend, RulesBackend, make_backend
from .config import JAHCfg
from .context import current_step
from .journal import Journal, head
from .points import ActionFilter, Asker
from .policy import FAST, STRONG, choose_tier, commit, forced_tier
from .sessions import SessionStore, make_store
from .state import HARD, QUESTION, render_state, session_key, step_index


@dataclass
class RouteResult:
    session: str
    step: int
    tier: str  # "strong" | "fast"
    reason: str
    p_hard: float | None = None
    decision: dict[str, Any] | None = None  # {model, cost_usd, latency_s, input_tokens, why, state_chars} | {error}


class JAHRouter:
    """One routing decision per request: which tier writes this agent step."""

    def __init__(self, cfg: JAHCfg | None = None, backend: DecisionBackend | None = None,
                 store: SessionStore | None = None, journal: Journal | None = None) -> None:
        self.cfg = cfg or JAHCfg()
        self.backend = backend if backend is not None else make_backend(self.cfg.decision)
        self.store = store or make_store(self.cfg)
        self.journal = journal or (Journal(self.cfg.log_dir) if self.cfg.log_dir else None)
        self.asker = Asker(self.backend)
        self.points = self.cfg.points
        self._filters: dict[Any, ActionFilter] = {}

    def action_filter(self, cfg: Any) -> ActionFilter:
        flt = self._filters.get(cfg)
        if flt is None:
            flt = self._filters[cfg] = ActionFilter(cfg)
        return flt

    async def close(self) -> None:
        await self.backend.close()
        await self.store.close()

    async def decide(self, messages: list[dict[str, Any]],
                     metadata: Mapping[str, Any] | None = None) -> RouteResult:
        """The route for this request: {session, step, tier, reason, p_hard, decision}."""
        key = session_key(messages, metadata)
        sess = await self.store.get(key)
        step = step_index(messages)
        current_step.set((key, step)) # the step logger reads it back: the logging events carry no metadata
        if step in sess.decided: # a retry of the same step reuses its decision
            r = sess.decided[step]
            return RouteResult(session=key, step=step, tier=r["tier"],
                               reason="repeat:" + str(r["reason"]).removeprefix("repeat:"),
                               p_hard=r.get("p_hard"), decision=r.get("decision"))
        inflight = sess.deciding.get(step)
        if inflight is not None: # the router may ask for the same step twice at once: one decision, one row
            try:
                return await inflight
            except asyncio.CancelledError:
                if not inflight.cancelled():
                    raise # we were cancelled, not the shared decision
                # the decider was cancelled: its future is cancelled, but that is not OUR cancellation -
                # fall through and decide anew instead of dying with a spurious CancelledError
                # Narrow race: if our own cancellation and the decider's cancel land in the same tick,
                # inflight.cancelled() is also True and we swallow our own cancellation here; the cost is one
# (the single-flight dedupe is per-process; across workers a step already decided is reused via
#  sess.decided, which the Redis store shares)
                # extra decision call for the step, never a hang.
        fut: asyncio.Future[RouteResult] = asyncio.get_running_loop().create_future()
        sess.deciding[step] = fut
        try:
            result = await self._decide_new(sess, key, step, messages)
        except BaseException as e:
            sess.deciding.pop(step, None)
            if not fut.done():
                if isinstance(e, asyncio.CancelledError):
                    fut.cancel() # waiters see a cancelled future, not an exception raised in their frame
                else:
                    fut.set_exception(e)
            raise
        sess.deciding.pop(step, None)
        fut.set_result(result)
        return result

    async def _decide_new(self, sess: Any, key: str, step: int,
                          messages: list[dict[str, Any]]) -> RouteResult:
        c = self.cfg
        rmode = self.points.routing.mode  # the resolved routing point: top-level mode + decisions overrides
        decision: dict[str, Any] | None = None
        p_hard: float | None = None
        asked = False  # the routing question was put to the decision model (a decision-call log row)
        forced = forced_tier(sess, step, c)
        if forced:
            tier, reason = STRONG, forced
            if forced == "stick":
                sess.stick_left -= 1
        elif rmode == "off":
            tier, reason = STRONG, "routing_off"
        elif rmode == "guard":
            tier, reason = FAST, "guard" # the fast tier writes the step; the routing guard checks it after the call
        else:
            asked = True
            state, view = render_state(messages)
            t0 = time.monotonic()
            try:
                if isinstance(self.backend, RulesBackend):
                    coro = self.backend.decide(state, QUESTION, view)
                else:
                    coro = self.backend.decide(state, QUESTION)
                # one timeout for every backend: a custom rules subclass must not hang a request either
                d = await asyncio.wait_for(coro, timeout=c.decision_timeout_s)
                p_hard = float(d.p.get(HARD, 0.0))
                decision = {"model": d.model, "cost_usd": d.cost_usd, "latency_s": round(time.monotonic() - t0, 4),
                            "input_tokens": d.input_tokens, "why": d.why or None, "state_chars": len(state)}
                if rmode == "shadow":
                    tier, reason = STRONG, "shadow"
                else:
                    tier, reason, escalated = choose_tier(p_hard, c)
                    if escalated:
                        sess.stick_left = c.stick_strong
            except Exception as e:  # noqa: BLE001 - fail-safe: a decision-model failure never stops the agent
                tier, reason = (STRONG, "shadow") if rmode == "shadow" else (c.on_error, "decision_error")
                decision = {"error": f"{type(e).__name__}: {str(e)[:200]}",
                            "latency_s": round(time.monotonic() - t0, 4)}
        commit(sess, tier)
        sess.decided[step] = {"tier": tier, "reason": reason, "p_hard": p_hard, "decision": decision}
        for old in [s for s in sess.decided if s < step - 4]:
            sess.decided.pop(old, None)
        if asked and self.journal is not None:
            try:
                await self.journal.decision(self._routing_row(key, step, decision, p_hard, reason))
            except Exception: # noqa: BLE001 - logging must never break routing
                pass
        if os.environ.get("JAH_DEBUG"):
            print(f"JAH decide: key={key} step={step} tier={tier} reason={reason} p_hard={p_hard}",
                  file=sys.stderr, flush=True)
        return RouteResult(session=key, step=step, tier=tier, reason=reason, p_hard=p_hard, decision=decision)

    def _routing_row(self, key: str, step: int, decision: dict[str, Any] | None, p_hard: float | None,
                     reason: str) -> dict[str, Any]:
        """One row of decisions.jsonl for the routing decision point, in the standalone proxy's
        ``Ask.record()`` shape (with the head ``jah report`` / ``jah fit`` expect)."""
        d = decision or {}
        p: dict[str, float] = {}
        if p_hard is not None:
            p = {"routine": round(1.0 - p_hard, 4), HARD: round(p_hard, 4)}
        top = sorted(p.items(), key=lambda kv: -kv[1])[:10]
        choice = max(p, key=lambda k: p[k]) if p else None
        return {
            **head(key, self.cfg.model_name, step),
            "point": "routing", "mode": self.points.routing.mode, "question": QUESTION.id,
            "n_options": len(QUESTION.options), "options": list(QUESTION.options),
            "p": {k: round(v, 4) for k, v in top}, "choice": choice,
            "confidence": None if not p else round(max(p.values()), 4), "tau": self.cfg.tau,
            "act_option": HARD, "p_act": round(p_hard, 4) if p else None,
            "acted": reason in ("hard", "routine"), "effect": reason, "label": None, "label_tier": None,
            "agree": None, "fit_ok": False, "latency_s": round(float(d.get("latency_s") or 0.0), 4),
            "cost_usd": round(float(d.get("cost_usd") or 0.0), 8), "model": str(d["model"]) if d.get("model") else None,
            "rounds": 1, "error": d.get("error"),
        }
