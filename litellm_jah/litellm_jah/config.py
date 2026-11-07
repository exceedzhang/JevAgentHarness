"""Plugin configuration: the JAH router settings, the tier labels, and the session store.

A file shaped like the standalone proxy's config works: a top-level ``router:`` block is read exactly like
``jevagentharness.config.Config.from_dict`` (its ``fast`` key is accepted and ignored). Extra keys are flat:
``mode``, the tier labels, the strong/fast model lists (routing-plugin variant), the finish-check recall
(``strong`` / ``strong_api_key`` / ``strong_api_base``) and the session store.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .points import Decisions, check_preset, resolve

MODES = ("cascade", "shadow", "off", "guard")
TIERS = ("strong", "fast")

# keys of the standalone proxy's router block that this plugin does not use yet
_IGNORED_ROUTER_KEYS = {"fast"}

DEFAULTS: dict[str, Any] = {
    "decision": {"type": "rules"},  # offline baseline; set {type: jev, provider: typesafe} for the real model
    "tau": 0.75,
    "first_steps_strong": 1,
    "stick_strong": 0,
    "max_fast_streak": 0,
    "mode": "cascade",
    "on_error": "strong",
    # the finish check (P2): a fast-tier final reply (no tool call) is re-done on this model
    "verify_finish": True,
    "strong": None, # model name for the finish-check re-run (litellm SDK name, e.g. "gemini/gemini-4-5-flash")
    "strong_api_key": None, # re-run credentials, when not in the environment (os.environ/NAME is read at call time)
    "strong_api_base": None,
    "decision_timeout_s": 8.0,
    "strong_label": "COMPLEX",
    "fast_label": "SIMPLE",
    "strong_models": (),
    "fast_models": (),
    "max_sessions": 5000,
    "session_ttl_s": 3600.0,
    # the session store: in-process by default; {type: redis, url: ...} for a shared store across
    # replicas/workers (needs the redis package, see the redis extra)
    "session_store": None,
    # logging in the JAH schema (steps.jsonl / decisions.jsonl), readable by `jah report` / `jah fit`
    "log_dir": None,
    # when on, successful decision rows also carry the exact state string sent to the decision model
    # (post-truncation): the decision point + state + question + options is then a complete prompt for
    # training-data harvests (the GRPO datasets are built from these rows)
    "log_decision_state": False,
    "model_name": "jah-auto",
    # the remaining decision points (tool select, stop, reflection, action): a preset and/or a decisions block
    "preset": None, # route | paper | safe | shadow
    "decisions": None, # {tool_select: {mode: shadow}, stop: {mode: guard}, ...}
    # the billing currency of `prices` and the step log's cost columns (one per deployment)
    "currency": "USD",
    # per-model prices for the step log's cost column: {model: {input, output, cached_input[, cache_write]}}
    # in the billing currency per million tokens (the engine registers them into litellm's cost map)
    "prices": None,
}


@dataclass
class JAHCfg:
    decision: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["decision"]))
    tau: float = DEFAULTS["tau"]
    first_steps_strong: int = DEFAULTS["first_steps_strong"]
    stick_strong: int = DEFAULTS["stick_strong"]
    max_fast_streak: int = DEFAULTS["max_fast_streak"]
    mode: str = DEFAULTS["mode"]
    on_error: str = DEFAULTS["on_error"]
    verify_finish: bool = DEFAULTS["verify_finish"]
    strong: str | None = None
    strong_api_key: str | None = None
    strong_api_base: str | None = None
    decision_timeout_s: float = DEFAULTS["decision_timeout_s"]
    strong_label: str = DEFAULTS["strong_label"]
    fast_label: str = DEFAULTS["fast_label"]
    strong_models: tuple[str, ...] = ()
    fast_models: tuple[str, ...] = ()
    max_sessions: int = DEFAULTS["max_sessions"]
    session_ttl_s: float = DEFAULTS["session_ttl_s"]
    session_store: dict[str, Any] | None = None
    log_dir: str | None = None
    log_decision_state: bool = False
    model_name: str = DEFAULTS["model_name"]
    currency: str = DEFAULTS["currency"]
    preset: str | None = None
    decisions: dict[str, Any] | None = None
    prices: dict[str, Any] | None = None # per-model prices (currency per million tokens) for the cost column

    @property
    def points(self) -> Decisions:
        """The resolved decision points (routing in ``mode`` + the finish check, then the preset, then the
        ``decisions`` block)."""
        return resolve(self.tau, self.verify_finish, check_preset(self.preset), self.decisions, self.mode)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "JAHCfg":
        d = dict(d or {})
        router = d.pop("router", None)
        if router is not None and not isinstance(router, Mapping):
            raise ValueError("router: must be a mapping")
        merged = dict(router or {})
        for k, v in d.items():
            if v is not None:
                merged[k] = v
        unknown = set(merged) - set(DEFAULTS) - _IGNORED_ROUTER_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs = {k: (list(v) if isinstance(v, list) and k in ("strong_models", "fast_models") else v)
                  for k, v in merged.items() if k in DEFAULTS}
        cfg = cls(**kwargs)
        cfg.strong_models = tuple(str(m) for m in cfg.strong_models)
        cfg.fast_models = tuple(str(m) for m in cfg.fast_models)
        cls._validate(cfg)
        return cfg

    @classmethod
    def _validate(cls, c: "JAHCfg") -> None:
        if c.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {c.mode!r}")
        if c.on_error not in TIERS:
            raise ValueError(f"on_error must be one of {TIERS}, got {c.on_error!r}")
        if not 0.0 <= c.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {c.tau!r}")
        for k in ("first_steps_strong", "stick_strong", "max_fast_streak"):
            if getattr(c, k) < 0:
                raise ValueError(f"{k} must be >= 0")
        if c.decision_timeout_s <= 0:
            raise ValueError("decision_timeout_s must be > 0")
        if c.strong_label == c.fast_label:
            raise ValueError("strong_label and fast_label must differ")
        if c.max_sessions <= 0 or c.session_ttl_s <= 0:
            raise ValueError("max_sessions and session_ttl_s must be > 0")
        if c.model_name in (None, ""):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(c.currency, str) or not c.currency.strip():
            raise ValueError("currency must be a non-empty string")
        if not isinstance(c.log_decision_state, bool):
            raise ValueError("log_decision_state must be a boolean")
        if c.prices is not None and (not isinstance(c.prices, Mapping) or not c.prices):
            raise ValueError("prices must be a non-empty mapping of model -> {input, output, ...}")
        for m, spec in (c.prices or {}).items():
            if not isinstance(spec, Mapping) or "input" not in spec or "output" not in spec:
                raise ValueError(f"prices[{m!r}]: expected a mapping with at least 'input' and 'output'")
            for k in ("input", "output", "cached_input", "cache_write"):
                v = spec.get(k)
                if v is None:
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError(f"prices[{m!r}][{k}]: expected a non-negative number")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JAHCfg":
        d = yaml.safe_load(Path(path).expanduser().read_text())
        if d is None:
            return cls()
        return cls.from_dict(d)

    @classmethod
    def from_env(cls) -> "JAHCfg":
        """``$LITELLM_JAH_CONFIG`` (path), else ``~/.jah/litellm.yaml`` if present, else the built-in defaults."""
        p = os.environ.get("LITELLM_JAH_CONFIG")
        if p:
            return cls.from_yaml(p)
        home = Path.home() / ".jah" / "litellm.yaml"
        if home.is_file():
            return cls.from_yaml(home)
        return cls()
