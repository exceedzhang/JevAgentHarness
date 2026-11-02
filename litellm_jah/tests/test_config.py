"""The plugin config: JAH router-block compatibility and validation."""

from __future__ import annotations

import pytest
from litellm_jah.config import JAHCfg


def test_defaults():
    c = JAHCfg()
    assert c.tau == 0.75 and c.first_steps_strong == 1 and c.mode == "cascade" and c.on_error == "strong"
    assert c.strong_label == "COMPLEX" and c.fast_label == "SIMPLE"


def test_router_block_compatibility():
    # a file shaped like the standalone proxy's config: the router block's keys are read as-is
    c = JAHCfg.from_dict({"router": {"tau": 0.6, "decision": {"type": "jev", "provider": "typesafe"},
                                     "stick_strong": 2, "verify_finish": True, "strong": "strong",
                                     "fast": "fast", "decision_timeout_s": 5}})
    assert c.tau == 0.6 and c.stick_strong == 2 and c.decision == {"type": "jev", "provider": "typesafe"}
    assert c.decision_timeout_s == 5


def test_flat_keys_and_normalisation():
    c = JAHCfg.from_dict({"tau": 0.5, "strong_models": ["a/x"], "fast_models": ["b/y"], "mode": "shadow"})
    assert c.strong_models == ("a/x",) and c.fast_models == ("b/y",) and c.mode == "shadow"


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        JAHCfg.from_dict({"tau": 0.5, "bogus_key": 1})
    with pytest.raises(ValueError, match="router: must be a mapping"):
        JAHCfg.from_dict({"router": "x"})


@pytest.mark.parametrize("bad", [
    {"mode": "cascadee"},
    {"on_error": "medium"},
    {"tau": 1.5},
    {"tau": -0.1},
    {"first_steps_strong": -1},
    {"decision_timeout_s": 0},
    {"strong_label": "SIMPLE", "fast_label": "SIMPLE"},
    {"max_sessions": 0},
])
def test_validation(bad):
    with pytest.raises(ValueError):
        JAHCfg.from_dict(bad)


def test_from_env_defaults_without_env(monkeypatch):
    monkeypatch.delenv("LITELLM_JAH_CONFIG", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-jah-home")  # no ~/.jah/litellm.yaml
    c = JAHCfg.from_env()
    assert c == JAHCfg()


def test_guard_mode_is_accepted():
    c = JAHCfg.from_dict({"mode": "guard"})
    assert c.mode == "guard" and c.points.routing.mode == "guard"


def test_decisions_can_override_the_routing_mode():
    c = JAHCfg.from_dict({"decisions": {"routing": {"mode": "guard"}}})
    assert c.mode == "cascade"  # the top-level default stands
    assert c.points.routing.mode == "guard"  # the override reaches the resolved point
