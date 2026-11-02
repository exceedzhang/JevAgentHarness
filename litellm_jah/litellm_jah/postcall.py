"""JAH's post-call decision points as a LiteLLM post-call deployment hook: the finish check, the stop
guard/shadow, the tool-select guard, the routing guard, and the action guard.

Mirrors the standalone proxy's ``_checks`` / ``_run_and_check``: the checks that apply to the reply run; any
that want a redo re-run the step on the strong tier with their harness notes (one redo, then one more look
when a redo proposes a risky action, like the standalone proxy), and their answers are labelled by what the
redo did so `jah fit` can fit their thresholds.

The re-run is a direct ``litellm.acompletion`` on ``cfg.strong`` (the finish check's model), like LiteLLM's
own skills hook; ``in_check`` marks it so this hook and the step logger ignore the internal call. Fail-safe
(deliberate difference from the standalone proxy, where a failed strong re-run fails the step): the original
call already succeeded at the deployment level, so if a re-run fails the original reply is kept. The checks
apply to non-streaming chat completions routed by this plugin.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .context import in_check
from .engine import JAHRouter
from .journal import head
from .points import (
    ACTION,
    CHARS_PROPOSAL,
    MAX_ACTION_ASKS,
    REDO_ORDER,
    ROUTE_GUARD,
    STOP,
    STOP_REJECT_NOTE,
    Ask,
    action_note,
    call_text,
    ends_turn,
    llm_choice,
    pass_through,
    proposal_text,
    recheck_kind,
    resp_tool_calls,
    submit_re,
    tc_name,
    tool_guard_note,
    with_notes,
)
from .state import render_state, session_key, step_index

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError: # litellm not installed: a stand-in so the module imports and unit-tests without it
    class CustomLogger: # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass


@dataclass
class _Verdict:
    """A post-call check's outcome: redo the step on the strong tier (with a note), and the decision calls
    made."""

    redo: bool
    reason: str
    note: str | None = None
    asks: list[Ask] = field(default_factory=list)


def _expand_env(value: str | None) -> str | None:
    """``os.environ/NAME`` reads the variable at call time (the litellm config convention);
    anything else is used as-is."""
    if not value:
        return None
    if value.startswith("os.environ/"):
        return os.environ.get(value.removeprefix("os.environ/")) or None
    return value


def _num(obj: Any, *names: str) -> int:
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def _call_entry(resp: Any, latency_s: float) -> dict[str, Any]:
    """One re-run call, for the step row's ``calls`` list (the first call's latency comes from the logger)."""
    return {"tier": "strong", "model": getattr(resp, "model", None),
            "in": _num(getattr(resp, "usage", None), "prompt_tokens"),
            "out": _num(getattr(resp, "usage", None), "completion_tokens"),
            "latency_s": latency_s}


class JahPostCall(CustomLogger):
    """LiteLLM post-call hook: the JAH checks that apply to a reply, and their strong-tier re-dos."""

    def __init__(self, engine: JAHRouter | None = None, recall: Any = None) -> None:
        super().__init__()
        if engine is None:
            from .classifier import jah_classifier
            engine = jah_classifier.router
        self.engine = engine
        self._recall = recall if recall is not None else self._recall_default

    async def async_post_call_success_deployment_hook(self, request_data: dict, response: Any,
                                                      call_type: Any) -> Any:
        try:
            return await self._check(request_data or {}, response, call_type)
        except Exception: # noqa: BLE001 - the checks must never break a completed request
            return None

    async def _check(self, request_data: dict, response: Any, call_type: Any) -> Any:
        if os.environ.get("JAH_DEBUG"):
            import sys
            import time
            print(f"JAH postcall: t={time.monotonic() % 1000:.1f} call_type={call_type} "
                  f"stream={request_data.get('stream')} model={request_data.get('model')}",
                  file=sys.stderr, flush=True)
        if in_check.get(False):
            return None
        if getattr(call_type, "value", call_type) not in ("completion", "acompletion"):
            return None # chat completions only (the proxy takes the async path)
        if request_data.get("stream"):
            return None # the reply is not in hand here
        messages = request_data.get("messages") or []
        if not messages:
            return None
        group = (request_data.get("metadata") or {}).get("model_group")
        if group and str(group) != self.engine.cfg.model_name:
            return None  # a model group this plugin does not serve ("model" is the resolved deployment name)
        tools = request_data.get("tools") or []
        key = session_key(messages, request_data.get("metadata") or {})
        step = step_index(messages)
        sess = await self.engine.store.get(key)
        rec = sess.decided.get(step)
        if rec is None:
            return None # not a step this plugin routed
        dc = self.engine.points
        state, view = render_state(messages)
        first_tier = rec["tier"]
        if os.environ.get("JAH_DEBUG"):
            import sys
            print(f"JAH postcall: step={step} tier={first_tier} points={dc.modes()}", file=sys.stderr)
        final = bool(tools) and ends_turn(response, submit_re(dc.stop))
        jobs: list = []
        if final and dc.stop.on:
            jobs.append(self._stop_check(response, dc.stop, first_tier, state, view))
        if "tool_select" in sess.pending and dc.tool_select.mode == "guard":
            jobs.append(self._tool_guard(sess.pending["tool_select"], response, first_tier))
        # the routing guard (the fast tier wrote the step): skipped when the finish check re-does this final
        # answer anyway
        redone = final and dc.stop.mode in ("strong", "trigger", "shadow")
        if dc.routing.mode == "guard" and rec["reason"] == "guard" and first_tier != "strong" and not redone:
            jobs.append(self._route_guard(response, dc.routing, first_tier, state, view))
        if dc.action.on and resp_tool_calls(response):
            jobs.append(self._action_check(response, first_tier, state, view))
        # Like the standalone proxy: run the checks concurrently and isolate a
        # failing one, so one bad check cannot starve the others or settle().
        verdicts: list[_Verdict] = []
        for v in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(v, _Verdict):
                verdicts.append(v)
            elif isinstance(v, BaseException) and os.environ.get("JAH_DEBUG"):
                import sys
                print(f"JAH postcall: check failed: {type(v).__name__}: {str(v)[:200]}", file=sys.stderr,
                      flush=True)
        result = response
        redo = [v for v in verdicts if v.redo]
        v2: _Verdict | None = None
        if redo:
            result, v2 = await self._redo(request_data, messages, tools, response, first_tier, redo, sess, step,
                                      state, view)
        if v2 is not None:
            verdicts.append(v2) # the "one more look" asks are logged too (the standalone merges them the same way)
        await settle(self.engine, sess, step, key, response, first_tier, verdicts)
        return result if result is not response else None

    # ------------------------------------------------------------------ the checks
    async def _stop_check(self, resp: Any, c: Any, tier: str, state: str, view: Any) -> _Verdict | None:
        on_strong = tier == "strong"
        if c.mode in ("strong", "trigger"):
            return None if on_strong else _Verdict(True, "finish_check")
        if on_strong and c.scope != "all":
            return None
        a = await self.engine.asker.ask(
            "stop", c.mode, f"{state}\n\nPROPOSED FINAL REPLY: {proposal_text(resp)}", STOP,
            self.engine.cfg.decision_timeout_s, c.tau, "done", view)
        a.extra["tier"] = tier
        if c.mode == "shadow": # behaves as `strong`; the strong redo labels the answer
            a.effect = "logged"
            return _Verdict(not on_strong, "finish_check", asks=[a])
        if not a.ok: # guard without an answer: the finish check (safe default)
            a.effect = "error_accept" if on_strong else "error_redo"
            return _Verdict(not on_strong, "finish_check", asks=[a])
        if a.p_of() >= c.tau:
            a.effect = "accepted"
            return _Verdict(False, "stop_guard", asks=[a])
        a.acted, a.effect = True, "redo"
        return _Verdict(True, "stop_guard", STOP_REJECT_NOTE if on_strong else None, [a])

    async def _tool_guard(self, fut: Any, resp: Any, tier: str) -> _Verdict:
        c = self.engine.points.tool_select
        a = await fut
        a.label, a.label_tier, a.fit_ok = llm_choice(resp), tier, True # the guard saw the real reply
        if not a.ok:
            a.effect = "error_accept"
            return _Verdict(False, "tool_guard", asks=[a])
        if a.choice != "reply_without_tool" and (a.confidence or 0.0) >= c.tau and a.choice != a.label:
            a.acted, a.effect = True, "redo"
            return _Verdict(True, "tool_guard", tool_guard_note(resp), [a])
        a.effect = "accepted"
        return _Verdict(False, "tool_guard", asks=[a])

    async def _route_guard(self, resp: Any, c: Any, tier: str, state: str, view: Any) -> _Verdict:
        # the fast tier wrote this step: does the decision model keep it, or re-do it on the strong tier?
        a = await self.engine.asker.ask(
            "routing", "guard", f"{state}\n\nPROPOSED NEXT STEP: {proposal_text(resp)}", ROUTE_GUARD,
            self.engine.cfg.decision_timeout_s, c.tau, "redo", view)
        a.extra["tier"] = tier
        if not a.ok: # no answer: the router's fail-safe tier decides
            redo = self.engine.cfg.on_error == "strong"
            a.effect = "error_redo" if redo else "error_accept"
            return _Verdict(redo, "route_guard", asks=[a])
        if a.p_of() >= c.tau:
            a.acted, a.effect = True, "redo"
            return _Verdict(True, "route_guard", asks=[a])
        a.effect = "accepted"
        return _Verdict(False, "route_guard", asks=[a])

    async def _action_check(self, resp: Any, tier: str, state: str, view: Any) -> _Verdict | None:
        c = self.engine.points.action
        flt = self.engine.action_filter(c)
        cands = flt.candidates(resp_tool_calls(resp))[:MAX_ACTION_ASKS]
        if not cands:
            return None
        asks = list(await asyncio.gather(*[self.engine.asker.ask(
            "action", c.mode, f"{state}\n\nPROPOSED ACTION: {call_text(x, CHARS_PROPOSAL)}", ACTION,
            self.engine.cfg.decision_timeout_s, c.tau, "risky", replace(view, risky=flt.is_risky(x)))
            for x in cands]))
        flagged: list[Any] = []
        for x, a in zip(cands, asks):
            a.extra.update({"tool": tc_name(x), "tier": tier, "pattern": flt.is_risky(x)})
            if not a.ok:
                a.effect = "error_recheck" if c.on_error == "recheck" and c.mode == "guard" else "error_accept"
                if a.effect == "error_recheck":
                    flagged.append(x)
            elif a.p_of() >= c.tau and c.mode == "guard":
                a.acted, a.effect = True, "recheck"
                flagged.append(x)
            else:
                a.effect = "logged" if c.mode == "shadow" else "accepted"
        if flagged:
            return _Verdict(True, "action_guard", action_note(flagged), asks)
        return _Verdict(False, "action_guard", asks=asks)

    # ------------------------------------------------------------------ the redo
    async def _redo(self, request_data: dict, messages: list[dict], tools: list[Any], first_resp: Any,
                    first_tier: str, redo: list[_Verdict], sess: Any, step: int, state: str, view: Any,
                    ) -> tuple[Any, _Verdict | None]:
        cfg = self.engine.cfg
        v2: _Verdict | None = None # the "one more look" verdict, returned so the caller can log its asks
        order = {r: i for i, r in enumerate(REDO_ORDER)} # a reason added later sorts last, not crash
        redo.sort(key=lambda v: order.get(v.reason, len(order)))
        reasons = [v.reason for v in redo]
        notes = [v.note for v in redo if v.note]
        if not cfg.strong:
            for v in redo:
                for a in v.asks:
                    a.effect = "error_kept"
            return first_resp, v2
        resp, lat = await self._rerun(cfg.strong, with_notes(messages, notes), tools, request_data)
        if resp is None: # the re-run failed: keep the original reply (the deployment already succeeded)
            for v in redo:
                for a in v.asks:
                    a.effect = "error_kept"
            return first_resp, v2
        # JAH's Router.escalated(): the step was re-done on the strong tier
        sess.fast_streak = 0
        sess.last_tier = "strong"
        sess.decided[step] = {**sess.decided[step], "tier": "strong", "reason": reasons[0]}
        sess.redo_calls[step] = [{
            "tier": first_tier, "model": getattr(first_resp, "model", None),
            "in": _num(getattr(first_resp, "usage", None), "prompt_tokens"),
            "out": _num(getattr(first_resp, "usage", None), "completion_tokens"), "latency_s": None}]
        sess.redo_calls[step].append(_call_entry(resp, lat))
        self._label_redo(redo, first_resp, resp, notes, sess)
        # the redo proposes a risky action: one more look (like the standalone proxy)
        if self.engine.points.action.on and "action_guard" not in reasons and resp_tool_calls(resp):
            v2 = await self._action_check(resp, "strong", state, view)
            if v2 is not None and v2.redo:
                resp2, lat2 = await self._rerun(cfg.strong, with_notes(messages, [v2.note] if v2.note else []),
                                                tools, request_data)
                if resp2 is not None:
                    for a in v2.asks:
                        if a.acted:
                            a.extra["recheck"] = recheck_kind(resp, resp2)
                            flt = self.engine.action_filter(self.engine.points.action)
                            a.extra["recheck_risky"] = any(flt.is_risky(c) for c in resp_tool_calls(resp2))
                    sess.redo_calls[step].append(_call_entry(resp2, lat2))
                    resp = resp2
        return resp, v2

    async def _rerun(self, model: str, messages: list[dict], tools: list[Any],
                     request_data: dict) -> tuple[Any | None, float]:
        tok = in_check.set(True)
        t0 = time.monotonic()
        try:
            return await self._recall(model, messages, tools, pass_through(request_data)), \
                round(time.monotonic() - t0, 3)
        except Exception: # noqa: BLE001 - fail-safe: keep the original reply, the turn ends
            return None, 0.0
        finally:
            in_check.reset(tok)

    async def _recall_default(self, model: str, messages: list[dict[str, Any]], tools: list[Any],
                              kwargs: dict) -> Any:
        """The re-run: a direct ``litellm.acompletion`` on the strong model, like LiteLLM's own skills hook."""
        from litellm import acompletion
        api_key = _expand_env(self.engine.cfg.strong_api_key)
        if api_key:
            kwargs["api_key"] = api_key
        api_base = _expand_env(self.engine.cfg.strong_api_base)
        if api_base:
            kwargs["api_base"] = api_base
        return await acompletion(model=model, messages=messages, tools=tools, **kwargs)

    # ------------------------------------------------------------------ labels and logs
    def _label_redo(self, redo: list[_Verdict], first_resp: Any, resp: Any, noted: list[str], sess: Any) -> None:
        """What the strong redo says about the checked reply: labels for `jah fit` (stop) and the logs."""
        dc = self.engine.points
        submits = submit_re(dc.stop)
        for v in redo:
            for a in v.asks:
                if a.point == "stop" and a.extra.get("tier") != "strong":
                    a.label = "done" if ends_turn(resp, submits) else "not_done"
                    a.label_tier, a.fit_ok = "strong", not bool(noted)
                elif a.point == "action" and a.acted:
                    flt = self.engine.action_filter(dc.action)
                    a.extra["recheck"] = recheck_kind(first_resp, resp)
                    a.extra["recheck_risky"] = any(flt.is_risky(c) for c in resp_tool_calls(resp))
        # the tool guard: how the strong redo chose to act (like the standalone proxy). A future that
        # ended in an exception is skipped: re-raising it here would abort the whole post-call and lose
        # every settled ask row.
        if dc.tool_select.mode == "guard":
            fut = sess.pending.get("tool_select")
            if fut is not None and fut.done() and not fut.cancelled() and fut.exception() is None:
                fut.result().extra["after"] = llm_choice(resp)

jah_post_call = JahPostCall()


async def settle(engine: JAHRouter, sess: Any, step: int, key: str, first_resp: Any, first_tier: str,
                 verdicts: list[_Verdict] | None = None) -> None:
    """Collect the pre-call questions that did not decide anything (shadow, guard) and label them with the
    LLM's own choice; write one decisions.jsonl row per decision call of this step. Also called on a
    failed request, so the questions that finished are logged, like the standalone proxy."""
    asks: dict[int, Ask] = {}
    for v in verdicts or []:
        for a in v.asks:
            asks[id(a)] = a
    if sess.pending:
        await asyncio.gather(*sess.pending.values(), return_exceptions=True)
        for name, fut in sess.pending.items():
            if fut.cancelled() or fut.exception() is not None:
                continue
            a = fut.result()
            if name == "tool_select" and first_resp is not None:
                if a.label is None:
                    # like the standalone proxy: a guard ask already labelled by _tool_guard
                    # keeps fit_ok=True (it judged the real reply); only unlabelled asks
                    # (shadow / assist) score against their own choice
                    a.label, a.label_tier = llm_choice(first_resp), first_tier
                    a.fit_ok = not a.acted
                if a.mode == "shadow":
                    a.effect = "logged"
                if a.extra.get("forced_tool"):
                    a.extra["followed"] = a.label == a.extra["forced_tool"]
            elif name == "reflection" and a.mode == "shadow":
                a.effect = "logged"
            asks[id(a)] = a
    sess.pending = {}
    all_asks = list(asks.values())
    if engine.journal is not None and all_asks:
        for a in all_asks:
            try:
                await engine.journal.decision({**head(key, engine.cfg.model_name, step), **a.record()})
            except Exception: # noqa: BLE001 - logging must never break the request
                pass
    sess.step_points[step] = {"cost_usd": sum(a.cost_usd for a in all_asks), "n_asks": len(all_asks)}
    for old in [s for s in sess.step_points if s < step - 4]:
        sess.step_points.pop(old, None)
    for old in [s for s in sess.decided if s < step - 4]:
        sess.decided.pop(old, None)
    for old in [s for s in sess.redo_calls if s < step - 4]:
        sess.redo_calls.pop(old, None)
    try:
        await engine.store.save(sess) # the shared (Redis) store: this step's state is durable now
    except Exception: # noqa: BLE001 - the store must never break the request (a no-op for the in-memory store)
        pass
