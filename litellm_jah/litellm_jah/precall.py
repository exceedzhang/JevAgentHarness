"""JAH's pre-call decision points as a LiteLLM pre-call hook: tool select, reflection and the stop trigger.

Mirrors the standalone proxy's ``_start_pre`` / ``_apply_pre``: the questions that can change this step are
launched with the request; the ones that decide (the stop trigger, a triggered reflection, the tool-select
assist/cascade) are awaited and applied to the request (harness notes on the last message, a forced
``tool_choice``, a shortlisted tool set) before the model runs. The non-deciding ones (shadow, guard) run in
parallel with the model call; their answers are settled and labelled by the post-call hook.

Register ``jah_pre_call`` under ``litellm_settings.callbacks``. It shares the routing engine's session store,
so the routing decision (made by the classifier/routing plugin on the same engine) is reused, never asked
twice. A retry of the same step reuses its plan (notes, forced tool, shortlist) from the session, like the
standalone proxy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .context import in_check
from .engine import JAHRouter
from .points import (
    MOVABLE,
    NO_TOOL,
    REFLECT,
    REFLECT_NOTE,
    STOP_TRIGGER,
    STOP_TRIGGER_NOTE,
    TOOL_TEXT,
    tool_options,
    used_tools,
    with_notes,
)
from .policy import commit
from .state import render_state, session_key

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError: # litellm not installed: a stand-in so the module imports and unit-tests without it
    class CustomLogger: # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass


class JahPreCall(CustomLogger):
    """LiteLLM pre-call hook: the JAH questions that can shape this step's request."""

    def __init__(self, engine: JAHRouter | None = None) -> None:
        super().__init__()
        if engine is None:
            from .classifier import jah_classifier
            engine = jah_classifier.router
        self.engine = engine

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict,
                                  call_type: Any) -> dict | None:
        try:
            return await self._apply(data or {}, call_type)
        except Exception: # noqa: BLE001 - the decision points must never break the request
            return None

    async def _apply(self, data: dict, call_type: Any) -> dict | None:
        if in_check.get(False):
            return None
        if getattr(call_type, "value", call_type) not in ("completion", "acompletion"):
            return None # chat completions only (the proxy takes the async path)
        messages = data.get("messages") or []
        if not messages:
            return None
        model = data.get("model")
        if model and model != self.engine.cfg.model_name:
            return None  # a model group this plugin does not serve
        metadata = data.get("metadata") or {}
        res = await self.engine.decide(messages, metadata)
        key = session_key(messages, metadata)
        sess = await self.engine.store.get(key)
        step, rec = res.step, res
        tools = data.get("tools") or []
        dc = self.engine.points
        ts, refl, stop = dc.tool_select, dc.reflection, dc.stop

        out = dict(data)
        plan = sess.plan.get(step)
        if plan is not None:  # a retried request: the same notes, tier and tool set as the first time
            if plan.get("notes"):
                out["messages"] = with_notes(list(data.get("messages") or []), list(plan["notes"]))
            if sess.decided[step]["tier"] == plan["tier"]:
                out = self._shape(out, {"tool_choice": plan.get("tool_choice"), "keep": plan.get("keep")})
            # after an escalation the tier moved on: like the standalone proxy, the retry goes out with the
            # notes only (the tool set was already re-decided by the redo)
            return out
        opening = step < self.engine.cfg.first_steps_strong
        forced = rec.reason in ("first_steps", "stick", "fast_streak")
        opts = tool_options(tools, data.get("tool_choice")) if ts.on else None
        ask_tool = bool(opts) and (ts.mode == "shadow" or (not opening and (
            (ts.mode == "assist" and len(tools) >= ts.min_tools) or ts.mode == "guard"
            or (ts.mode == "cascade" and not forced))))
        ask_refl = refl.mode in ("trigger", "shadow") and step >= 1
        ask_stop = stop.mode == "trigger" and step >= 1
        for f in list(sess.pending.values()):
            if not f.done():
                f.cancel()  # an unsettled ask from an earlier step must not outlive this request
        if ask_tool or ask_refl or ask_stop:
            state, view = render_state(messages)
            timeout = self.engine.cfg.decision_timeout_s
            pending: dict[str, Any] = {}
            if ask_tool and opts:
                pending["tool_select"] = asyncio.ensure_future(self.engine.asker.ask_choice(
                    "tool_select", ts.mode, state, TOOL_TEXT, opts[0], opts[1], timeout, ts.tau,
                    ts.max_options, keep=max(ts.k, 3)))
            if ask_refl:
                pending["reflection"] = asyncio.ensure_future(self.engine.asker.ask(
                    "reflection", refl.mode, state, REFLECT, timeout, refl.tau, "yes", view))
            if ask_stop:
                pending["stop_trigger"] = asyncio.ensure_future(self.engine.asker.ask(
                    "stop", "trigger", state, STOP_TRIGGER, timeout, stop.tau, "yes", view))
            sess.pending = pending
        else:
            sess.pending = {}
        blocking = [k for k in sess.pending
                    if k == "stop_trigger" or (k == "reflection" and refl.mode == "trigger")
                    or (k == "tool_select" and ts.mode in ("assist", "cascade"))]
        if blocking:
            try:
                await asyncio.gather(*(sess.pending[k] for k in blocking))
            except asyncio.CancelledError:
                # the request went away mid-flight: drop the in-flight asks so they
                # cannot outlive it, then let the cancellation propagate
                for f in sess.pending.values():
                    if not f.done():
                        f.cancel()
                sess.pending = {}
                raise
        notes: list[str] = []
        fired = None
        if "reflection" in sess.pending and refl.mode == "trigger":
            a = sess.pending["reflection"].result()
            if a.ok and a.p_of() >= refl.tau:
                a.acted, a.effect = True, "reflect_on_strong"
                notes.append(REFLECT_NOTE)
                fired = "reflection"
            else:
                a.effect = "error_skip" if not a.ok else "no"
        elif refl.mode == "always" and step >= 1 and step % refl.every == 0:
            notes.append(REFLECT_NOTE)
            fired = "reflection"
        if "stop_trigger" in sess.pending:
            a = sess.pending["stop_trigger"].result()
            if a.ok and a.p_of() >= stop.tau:
                a.acted, a.effect = True, "finish_note_on_strong"
                notes.append(STOP_TRIGGER_NOTE)
                fired = fired or "stop_trigger"
            else:
                a.effect = "error_skip" if not a.ok else "no"
        if fired: # the step goes to the strong tier; the routing plugins see the updated record
            sess.decided[step] = {**sess.decided[step], "tier": "strong", "reason": fired}
            commit(sess, "strong")
        force: str | None = None
        keep: list[str] | None = None
        if "tool_select" in sess.pending and ts.mode in ("assist", "cascade"):
            a = sess.pending["tool_select"].result()
            if ts.mode == "cascade":
                if not a.ok:
                    a.effect = "error_defer"
                elif a.choice == NO_TOOL:
                    a.effect = "defer_no_tool"
                elif (a.confidence or 0.0) < ts.tau:
                    a.effect = "defer_unsure"
                elif fired or rec.reason not in MOVABLE:
                    a.effect = f"defer_{rec.reason}" # a hard or forced-strong step stays the strong tier's
                else:
                    tier = self._cascade_tier(ts)
                    a.acted, a.effect = True, "forced" if ts.force else "tier"
                    a.extra["forced_tool"] = a.choice
                    sess.decided[step] = {**sess.decided[step], "tier": tier, "reason": "tool_cascade"}
                    if tier != rec.tier:
                        commit(sess, tier)
                    force = a.choice if ts.force else None
            elif not a.ok:
                a.effect = "error_all_tools"
            else:
                top = [o for o, _ in sorted(a.p.items(), key=lambda kv: -kv[1]) if o != NO_TOOL][:ts.k]
                wanted = set(top) | used_tools(messages)
                names = []
                for t in tools:
                    f = t.get("function") if isinstance(t, dict) else None
                    name = str(f.get("name") or "") if f else ""
                    if name in wanted:
                        names.append(name)
                if names and len(names) < len(tools):
                    keep = names
                    a.acted, a.effect = True, "shortlist"
                    a.extra["kept_tools"] = len(names)
                else:
                    a.effect = "all_tools"
        if notes or force or keep or sess.pending:
            sess.plan[step] = {"tier": sess.decided[step]["tier"], "reason": sess.decided[step]["reason"],
                               "notes": notes, "tool_choice": force, "keep": keep}
            for old in [s for s in sess.plan if s < step - 4]:
                sess.plan.pop(old, None)
        return self._shape(out, {"notes": notes, "tool_choice": force, "keep": keep}) if (notes or force or keep) \
            else out

    @staticmethod
    def _cascade_tier(ts: Any) -> str:
        return "fast" if ts.tier in (None, "fast") else "strong" if ts.tier == "strong" else str(ts.tier)

    @staticmethod
    def _shape(data: dict, plan: dict) -> dict:
        """The request with the plan applied: notes on the last message, a forced tool, a shortlisted tool set."""
        out = dict(data)
        if plan.get("notes"):
            out["messages"] = with_notes(list(data.get("messages") or []), list(plan["notes"]))
        if plan.get("tool_choice"):
            out["tool_choice"] = {"type": "function", "function": {"name": plan["tool_choice"]}}
        keep = plan.get("keep")
        if keep:
            keep_set = set(keep)
            out["tools"] = [t for t in (data.get("tools") or []) if
                            (t.get("function", {}).get("name") if isinstance(t, dict) else None) in keep_set]
        return out


jah_pre_call = JahPreCall()
