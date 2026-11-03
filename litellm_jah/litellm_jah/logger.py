"""JAH steps.jsonl from the LiteLLM proxy: one row per completed step, in the schema ``jah report`` reads.

Register ``jah_step_logger`` (or the class ``JahStepLogger``) under ``litellm_settings.callbacks``. The logger
shares the routing engine's session store, so it records the tier and the reason the plugin chose for each
step (the ``decision`` dict in the row carries the decision model's latency and cost). Steps the plugin did not
route (other model groups) are not logged.

Cost uses LiteLLM's own cost map when the model is priced there; otherwise the call is logged with
``priced: false`` (the report counts it in ``unpriced_calls``).
"""

from __future__ import annotations

import os
from typing import Any

from .context import current_step, in_check
from .engine import JAHRouter
from .journal import Journal, head
from .postcall import JahPostCall, settle
from .state import session_key, step_index

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # litellm not installed: a stand-in so the module imports and unit-tests without it
    class CustomLogger:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass


def _num(obj: Any, *names: str) -> int:
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def _elapsed(start_time: Any, end_time: Any) -> float:
    """Seconds between the event's start/end, whatever their type: LiteLLM passes datetimes (a
    timedelta apart); the unit tests pass epoch floats. 0.0 when either is missing."""
    if not start_time or not end_time:
        return 0.0
    delta = end_time - start_time
    secs = delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
    return round(secs, 3)


class JahStepLogger(CustomLogger):
    """LiteLLM logging callback: the JAH step row per completed request the plugin routed."""

    def __init__(self, engine: JAHRouter | None = None, log_dir: str | None = None) -> None:
        super().__init__()
        if engine is None:
            from .classifier import jah_classifier
            engine = jah_classifier.router
        self.engine = engine
        # note: log_dir re-points the (possibly shared) engine's journal; the plugin's router owns
        # that engine (the classifier singleton, or the one the proxy wiring passed in)
        if log_dir is not None:
            self.engine.cfg.log_dir = log_dir
            self.engine.journal = Journal(log_dir)
        self.journal = self.engine.journal
        self._postcall = JahPostCall(engine=self.engine)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        if os.environ.get("JAH_DEBUG"):
            import sys
            import time
            print(f"JAH logger: success event t={time.monotonic() % 1000:.1f} "
                  f"model={kwargs.get('model')} nmsg={len(kwargs.get('messages') or [])} "
                  f"in_check={in_check.get(False)}", file=sys.stderr, flush=True)
        try:
            await self._log(kwargs, response_obj, start_time, end_time, error=None)
        except Exception:  # noqa: BLE001 - logging must never break the request
            if os.environ.get("JAH_DEBUG"):
                import traceback
                traceback.print_exc()

    async def async_log_stream_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        # streamed completions never reach the success event (the base class no-ops stream events),
        # so the step row and the pre-call settle happen here
        if os.environ.get("JAH_DEBUG"):
            import sys
            import time
            print(f"JAH logger: stream event t={time.monotonic() % 1000:.1f} "
                  f"model={kwargs.get('model')} nmsg={len(kwargs.get('messages') or [])} "
                  f"in_check={in_check.get(False)}", file=sys.stderr, flush=True)
        try:
            await self._log(kwargs, response_obj, start_time, end_time, error=None)
        except Exception: # noqa: BLE001 - logging must never break the request
            if os.environ.get("JAH_DEBUG"):
                import traceback
                traceback.print_exc()

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            err = str(kwargs.get("exception") or kwargs.get("e") or "upstream error")[:300]
            await self._log(kwargs, response_obj, start_time, end_time, error=err)
            if not in_check.get(False): # an upstream error: the decision questions that finished are logged
                rid = self._route_id(kwargs)
                if rid is not None:
                    key, step = rid
                    sess = await self.engine.store.get(key)
                    if sess.pending:
                        rec = sess.decided.get(step)
                        await settle(self.engine, sess, step, key, None,
                                     rec.get("tier") if rec else None, [])
        except Exception: # noqa: BLE001
            if os.environ.get("JAH_DEBUG"):
                import traceback
                traceback.print_exc()

    @staticmethod
    def _route_id(kwargs: dict) -> tuple[str, int] | None:
        """The (session, step) this event belongs to, if the plugin routed it. Prefers the identity
        ``JAHRouter.decide`` recorded in the request context (the logging event's kwargs carry no request
        metadata); falls back to recomputing it from the messages.

        The fallback is safe here and only here: the logger writes rows and never changes session
        state, so recording a step it cannot map exactly beats dropping the row. ``stream_label``
        must not fall back: there the identity decides whether a strong re-run runs."""
        kv = current_step.get(None)
        if kv is not None:
            return kv
        data = kwargs.get("data") or {}
        messages = kwargs.get("messages") or data.get("messages") or []
        if not messages:
            return None
        metadata = kwargs.get("metadata") or data.get("metadata") or {}
        return session_key(messages, metadata), step_index(messages)

    async def _log(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any, error: str | None) -> None:
        dbg = os.environ.get("JAH_DEBUG")
        if self.journal is None or in_check.get(False):
            if dbg:
                import sys
                print(f"JAH logger: skip journal={self.journal is not None} "
                      f"in_check={in_check.get(False)}", file=sys.stderr)
            return # an internal strong re-run, not a client step
        md = kwargs.get("metadata") or (kwargs.get("data") or {}).get("metadata") or {}
        group = str(md.get("model_group") or (kwargs.get("data") or {}).get("model") or "")
        if group and group != self.engine.cfg.model_name:
            return None  # a model group this plugin does not serve
        rid = self._route_id(kwargs)
        if rid is None:
            if dbg:
                import sys
                print("JAH logger: skip unrouted (no route id)", file=sys.stderr)
            return # not a step this plugin routed
        key, step = rid
        sess = await self.engine.store.get(key)
        rec = sess.decided.get(step)
        if rec is None:
            if dbg:
                import sys
                print(f"JAH logger: skip unrouted key={key} step={step} decided={list(sess.decided)}", file=sys.stderr)
            return # not a step this plugin routed
        data = kwargs.get("data") or {}
        if error is None and (kwargs.get("stream") or data.get("stream")):
            # the post-call hook skips streams (no reply in hand): the label-only stream check settles
            # the pre-call asks against the complete reply, and for a fast final answer under stop:
            # shadow it runs the shadow redo for its label (the streamed reply is never replaced).
            # error is None: the failure path settles on its own, and a second settle would clobber
            # the step point counts
            if dbg:
                import sys
                print("JAH logger: stream gate -> stream_label", file=sys.stderr, flush=True)
            await self._postcall.stream_label(kwargs, response_obj)
            rec = sess.decided.get(step) or rec  # a shadow redo may have escalated the step's tier
        if dbg:
            import sys
            print(f"JAH logger: writing key={key} step={step} tier={rec.get('tier')}", file=sys.stderr)
        data = kwargs.get("data") or {}
        # the group the client asked for: model_group in real events, data["model"] in legacy shapes
        model_name = str(md.get("model_group") or data.get("model") or self.engine.cfg.model_name)
        usage = getattr(response_obj, "usage", None) if response_obj is not None else None
        in_t = _num(usage, "prompt_tokens")
        out_t = _num(usage, "completion_tokens")
        details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
        # the serving model: the event's own model (the deployment's), the response's (the proxy may
        # echo the requested alias, e.g. "jah-auto")
        model = kwargs.get("model") or getattr(response_obj, "model", None)
        fast_lat = _elapsed(start_time, end_time)
        # a strong re-run (finish check, stop guard, ...): this event's response is the final reply (the hook
        # replaced it); the step row records every call of the step, like the standalone proxy
        calls_log = sess.redo_calls.get(step)
        if calls_log:
            calls = []
            llm_cost = lat = 0.0
            for c in calls_log:
                # the first call was written without a latency (the post-call hook sees no event times): it takes
                # the event span, which on litellm 1.106.x covers the first call only - the deployment hook (the
                # checks and the re-run) runs after the success event's end_time, so the re-run is counted once
                c_lat = fast_lat if c.get("latency_s") is None else float(c.get("latency_s") or 0.0)
                cost, priced = self._cost(c.get("model"), c.get("in", 0), c.get("out", 0),
                                      {"cached_tokens": c.get("cached", 0)})
                calls.append({"tier": c.get("tier"), "model": c.get("model"), "latency_s": round(c_lat, 3),
                              "attempts": 1, "cost_usd": round(cost, 8), "priced": priced})
                llm_cost += cost
                lat += c_lat
        else:
            llm_cost, priced = self._cost(
            model, in_t, out_t, {"cached_tokens": _num(details, "cached_tokens"),
                                 "cache_write_tokens": _num(details, "cache_write_tokens")})
            calls = [{"tier": rec.get("tier"), "model": model, "latency_s": fast_lat, "attempts": 1,
                      "cost_usd": round(llm_cost, 8), "priced": priced}]
            lat = fast_lat
        d = rec.get("decision") or {}
        points = sess.step_points.get(step) or {}
        decision_cost = float(d.get("cost_usd") or 0.0) + float(points.get("cost_usd") or 0.0)
        row = {
            **head(key, model_name, step),
            "tier": rec.get("tier"), "reason": str(rec.get("reason") or "").removeprefix("repeat:") or "unknown",
            "p_hard": rec.get("p_hard"), "decision": d,
            "calls": calls,
            "llm_cost_usd": round(llm_cost, 8),
            "decision_cost_usd": round(decision_cost, 8),
            "cost_usd": round(llm_cost + decision_cost, 8),
            "currency": self.engine.cfg.currency,
            "latency_s": round(lat, 3),
            "input_tokens": in_t, "cached_tokens": _num(details, "cached_tokens"), "cache_write_tokens": 0,
            "output_tokens": out_t, "reasoning_tokens": _num(_details(usage, "completion_tokens_details"),
            "reasoning_tokens"),
            "n_tool_calls": self._tool_calls(response_obj), "finish": self._finish(response_obj),
        }
        if error:
            row["error"] = error
        await self.journal.step(row)

    def _cost(self, model: str | None, in_t: int, out_t: int,
              details: dict[str, Any] | None = None) -> tuple[float, bool]:
        """One call via litellm's cost map (the router registers each deployment's model_info and
        the engine the config's ``prices``). OpenAI usage shape: ``prompt_tokens`` already includes
        the cached tokens, so they go in via ``usage_object`` (an explicit ``cache_read_input_tokens``
        argument would make litellm treat them as additive to the prompt)."""
        if not model:
            return 0.0, False
        self.engine.ensure_price(model)
        try:
            from types import SimpleNamespace

            import litellm
            # the provider-prefixed key first: cost_per_token infers the provider from the name,
            # which raises for an unknown bare name even when the bare key is registered
            m = f"openai/{model}" if f"openai/{model}" in litellm.model_cost else model
            if m not in litellm.model_cost: # not in the cost map: log the call as unpriced
                return 0.0, False
            d = details or {}
            # cost_per_token reads completion_tokens_details: it must exist or the pricing
            # raises and the call is logged unpriced
            usage = SimpleNamespace(prompt_tokens=in_t, completion_tokens=out_t,
                                    prompt_tokens_details=SimpleNamespace(
                                        cached_tokens=int(d.get("cached_tokens", 0) or 0),
                                        cache_write_tokens=int(d.get("cache_write_tokens", 0) or 0)),
                                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
            prompt_c, completion_c = litellm.cost_per_token(
                model=m, prompt_tokens=in_t, completion_tokens=out_t, usage_object=usage)
            return float(prompt_c) + float(completion_c), True
        except Exception: # noqa: BLE001 - unknown model or no litellm: log the call as unpriced
            if os.environ.get("JAH_DEBUG"):
                import traceback
                traceback.print_exc()
            return 0.0, False


    @staticmethod
    def _tool_calls(response_obj: Any) -> int:
        try:
            c = response_obj.choices[0]
            m = c.message if hasattr(c, "message") else c.get("message")
            tcs = m.tool_calls if hasattr(m, "tool_calls") else m.get("tool_calls")
            return len(tcs) if tcs else 0
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _finish(response_obj: Any) -> str | None:
        try:
            c = response_obj.choices[0]
            return c.finish_reason if hasattr(c, "finish_reason") else c.get("finish_reason")
        except Exception:  # noqa: BLE001
            return None


def _details(usage: Any, name: str) -> Any:
    if usage is None:
        return None
    return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)


jah_step_logger = JahStepLogger()
