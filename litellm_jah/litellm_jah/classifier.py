"""The LiteLLM-facing plugin classes: two wiring styles, one engine (:class:`litellm_jah.engine.JAHRouter`).

1. ``JahClassifier`` - a complexity-router *classifier plugin* (``classifier_type: custom``, LiteLLM >= 1.99):
   per request it answers JAH's routing question and returns the tier label; the complexity router then serves
   that tier's models. A classifier failure/timeout falls back the router's ``default_model`` - set it to the
   strong model to keep JAH's fail-safe (a decision-model outage routes the step to the strong tier). In
   litellm 1.106.x the fallback only reaches ``default_model`` with
   ``classifier_fallback: default_model`` in the ``complexity_router_config`` (the 1.106.x default,
   ``heuristic``, scores the prompt locally instead and can pick the fast tier).
2. ``JahRoutingPlugin`` - a plain *routing plugin* (LiteLLM >= 1.94): it narrows the candidate pool of one model
   group to the chosen tier's models (``strong_models``/``fast_models`` in the config).

Neither class imports litellm: they duck-type the routing context (``structured_messages``, ``metadata``,
``candidate_models``, ``signals``), so the engine stays unit-testable without a LiteLLM install.

Every plugin is built on the single process-wide shared engine by default: these two module singletons, the
pre-call / post-call / step-logger callbacks, and any further default-constructed instance. That keeps the
journal, the session state and the routing rows single no matter which dotted path the config names. Pass
an explicit ``cfg`` or ``router`` to build a dedicated engine (tests do).
"""

from __future__ import annotations

from typing import Any

from .config import JAHCfg
from .engine import JAHRouter
from .policy import STRONG


class _JahPluginBase:
    def __init__(self, cfg: JAHCfg | None = None, router: JAHRouter | None = None) -> None:
        if cfg is None and router is None:
            cfg, router = _shared_engine() # default: the process-wide shared engine
        self.cfg = cfg or JAHCfg.from_env()
        self.router = router or JAHRouter(self.cfg)

    def __deepcopy__(self, memo: dict) -> "_JahPluginBase":
        # The proxy deep-copies its config (``_ConfigWithBaseline``, deployment
        # snapshots), which would duplicate the plugin and — with the default
        # object protocol — its whole JAHRouter: a second journal, session store
        # and decision writer while the hooks keep using the original. The
        # engine is process-wide state, so copies share it.
        clone = object.__new__(type(self))
        clone.cfg = self.cfg
        clone.router = self.router
        return clone

    async def close(self) -> None:
        await self.router.close()

    async def _route(self, context: Any) -> "Any":
        messages = list(getattr(context, "structured_messages", None) or getattr(context, "raw_messages", None) or [])
        metadata = getattr(context, "metadata", None) or {}
        return await self.router.decide(messages, metadata)


class JahClassifier(_JahPluginBase):
    """Complexity-router classifier plugin: ``classifier_type: custom`` + ``classifier_plugin: <dotted path>``."""

    async def classify(self, context: Any) -> str | None:
        """The tier label for this request (``COMPLEX`` -> strong models, ``SIMPLE`` -> fast), or None to decline
        (the router falls back to its heuristic scorer / default_model)."""
        try:
            res = await self._route(context)
        except Exception:
            return None
        return self.cfg.strong_label if res.tier == STRONG else self.cfg.fast_label


class JahRoutingPlugin(_JahPluginBase):
    """Routing plugin (``router_settings.plugins`` / ``Router(plugins=[...])``): narrows the candidate pool to the
    chosen tier's models. Fail-safe: on any failure the strong models are kept."""

    async def run(self, context: Any) -> Any:
        # Fail-safe: on ANY failure (route error, or a context lacking the attributes
        # the router is expected to provide) the original context is returned untouched.
        try:
            res = await self._route(context)
            tier = res.tier if res is not None else STRONG
            keep = set(self.cfg.strong_models if tier == STRONG else self.cfg.fast_models)
            if keep:
                candidates = list(getattr(context, "candidate_models", None) or [])
                narrowed = [m for m in candidates if m in keep]
                if narrowed:
                    context.candidate_models = narrowed
            if res is not None:
                signals = getattr(context, "signals", None)
                if isinstance(signals, dict):
                    signals["jah-router"] = {"tier": res.tier, "reason": res.reason, "p_hard": res.p_hard}
        except Exception: # noqa: BLE001
            pass
        return context


_shared_cfg: JAHCfg | None = None
_shared_router: JAHRouter | None = None


def _shared_engine() -> tuple[JAHCfg, JAHRouter]:
    """The process-wide engine: every plugin built without an explicit ``cfg`` / ``router`` shares it, so the
    journal, the session state and the routing rows stay single."""
    global _shared_cfg, _shared_router
    if _shared_router is None:
        _shared_cfg = JAHCfg.from_env()
        _shared_router = JAHRouter(_shared_cfg)
    return _shared_cfg, _shared_router


# module-level instances for the config.yaml dotted paths (litellm resolves them at startup);
# both wiring styles default to the shared engine.
jah_classifier = JahClassifier()
jah_routing_plugin = JahRoutingPlugin()
