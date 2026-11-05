"""Unit tests for the decision points: config validation, the pre-filter, and the shared helpers."""

from __future__ import annotations

import pytest
from litellm_jah.backend import Decision
from litellm_jah.points import (
    ALIASES,
    MODES,
    NO_TOOL,
    PRESETS,
    STOP,
    TAU,
    ActionFilter,
    Ask,
    Asker,
    PointCfg,
    call_text,
    ends_turn,
    llm_choice,
    pass_through,
    proposal_text,
    recheck_kind,
    resolve,
    same_calls,
    submit_re,
    tool_options,
    used_tools,
    validate,
    with_notes,
)

from conftest import Scripted, p_over


def tc(name, args='{"command": "ls"}'):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def resp_like(tool_calls=None, content="final answer"):
    class M:
        pass

    class C:
        pass

    m = M()
    m.tool_calls = tool_calls
    m.content = None if tool_calls else content
    c = C()
    c.message = m
    r = C()
    r.choices = [c]
    return r


# ---------------------------------------------------------------------------------------------- validation
def test_validate_expands_string_shorthands():
    out = validate({"stop": "guard", "tool_select": {"mode": "shadow", "k": 3}})
    assert out["stop"] == {"mode": "guard"}
    assert out["tool_select"]["mode"] == "shadow" and out["tool_select"]["k"] == 3


def test_validate_resolves_aliases():
    out = validate({"routing": "trigger"})
    assert out["routing"]["mode"] == ALIASES[("routing", "trigger")] == "cascade"


def test_validate_yaml_off_is_not_false():
    out = validate({"stop": False})
    assert out["stop"]["mode"] == "off"


@pytest.mark.parametrize("block, why", [
    ({"stop": "nope"}, "bad mode"),
    ({"bogus": "off"}, "unknown point"),
    ({"stop": {"mode": "guard", "wat": 1}}, "unknown key"),
    ({"stop": {"tau": 2.0}}, "tau out of range"),
    ({"stop": {"scope": "sometimes"}}, "bad scope"),
    ({"action": {"on_error": "retry"}}, "bad on_error"),
    ({"stop": {"submit": "[unclosed"}}, "bad regex"),
    ({"tool_select": {"max_options": 256}}, "too many options"),
    ({"stop": 42}, "not a mapping"),
])
def test_validate_rejects_bad_blocks(block, why):
    with pytest.raises(ValueError, match="decisions|must|expected"):
        validate(block)


def test_every_preset_is_valid():
    for name, preset in PRESETS.items():
        d = resolve(0.75, True, name)
        assert d.modes() and all(m in MODES[p] for p, m in d.modes().items()), name


def test_resolve_layers_default_preset_and_block():
    d = resolve(0.75, True, "shadow", {"stop": "strong", "tool_select": {"mode": "cascade", "tau": 0.9}})
    assert d.routing.mode == "cascade" and d.routing.tau == 0.75
    assert d.stop.mode == "strong" and d.stop.tau == TAU["stop"]
    assert d.tool_select.mode == "cascade" and d.tool_select.tau == 0.9
    assert d.reflection.mode == "shadow" and d.action.mode == "shadow"


def test_resolve_without_finish_check_turns_stop_off():
    assert resolve(0.75, False).stop.mode == "off"


# ---------------------------------------------------------------------------------------------- tool options
TOOLS = [tc(n) for n in ("bash", "read_file", "web_search")]


def test_tool_options_adds_the_no_tool_option():
    opts, descs = tool_options(TOOLS, None)
    assert opts == ["bash", "read_file", "web_search", NO_TOOL]
    assert len(descs) == 4


def test_tool_options_required_has_no_no_tool_option():
    opts, _ = tool_options(TOOLS, "required")
    assert opts == ["bash", "read_file", "web_search"]


@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "bash"}}])
def test_tool_options_none_when_the_choice_is_not_open(choice):
    assert tool_options(TOOLS, choice) is None
    assert tool_options([], None) is None


def test_tool_options_none_with_fewer_than_two_options():
    assert tool_options([tc("bash")], "required") is None


def test_used_tools():
    m = [{"role": "assistant", "tool_calls": [tc("bash")]}, {"role": "tool", "content": "ok"}]
    assert used_tools(m) == {"bash"}


# ---------------------------------------------------------------------------------------------- the action pre-filter
def action_filter(**kw):
    return ActionFilter(PointCfg(point="action", **kw))


def test_risky_patterns_match_the_documented_hazards():
    flt = action_filter()
    risky = [
        tc("bash", '{"command": "rm -rf /tmp/data"}'),
        tc("bash", '{"command": "sudo reboot"}'),
        tc("bash", '{"command": "git push --force origin main"}'),
        tc("bash", '{"command": "DROP TABLE users;"}'),
        tc("delete_file", '{"path": "/a/b"}'),
    ]
    safe = [
        tc("bash", '{"command": "ls -la"}'),
        tc("bash", '{"command": "grep -rn rm ."}'),
        tc("read_file", '{"path": "/a/b"}'),
    ]
    assert all(flt.is_risky(c) for c in risky)
    assert not any(flt.is_risky(c) for c in safe)


def test_candidates_per_prefilter_mode():
    calls = [tc("bash", '{"command": "rm x"}'), tc("write_file", '{"path": "y"}')]
    assert action_filter(prefilter="risky").candidates(calls) == [calls[0]]
    assert action_filter(prefilter="command").candidates(calls) == calls  # both are command-like names
    assert action_filter(prefilter="all").candidates(calls) == calls


def test_custom_patterns_and_tool_names():
    flt = action_filter(patterns=[r"\bnuke\b"])
    assert flt.is_risky(tc("bash", '{"command": "nuke it"}'))
    assert not action_filter().is_risky(tc("bash", '{"command": "nuke it"}'))  # custom only
    assert not action_filter(patterns=[r"\bnuke\b"]).is_risky(tc("bash", '{"command": "ls"}'))
    # extra tool names count as command-like in command mode only
    assert action_filter(tools=[r"explode"]).candidates([tc("explode_all", '{}')]) == []
    assert action_filter(prefilter="command", tools=[r"explode"]).candidates([tc("explode_all", '{}')]) \
        == [tc("explode_all", '{}')]


# ---------------------------------------------------------------------------------------------- reply helpers
def test_ends_turn_and_submit_patterns():
    assert ends_turn(resp_like(), None)
    assert not ends_turn(resp_like(tool_calls=[tc("bash")]), None)
    submit = submit_re(PointCfg(point="stop", submit=(r"finish\b",)))
    assert ends_turn(resp_like(tool_calls=[tc("finish", '{"done": true}'), tc("bash")]), submit)
    assert not ends_turn(resp_like(tool_calls=[tc("bash")]), submit)
    assert submit_re(PointCfg(point="stop")) is None


def test_same_calls_ignores_json_formatting_and_whitespace():
    a = resp_like(tool_calls=[tc("bash", '{"command": "ls -la", "timeout": 5}'), tc("read_file", '{"path": "a"}')])
    b = resp_like(tool_calls=[tc("bash", '{"timeout": 5, "command": "ls   -la"}'), tc("read_file", '{"path":"a"}')])
    assert same_calls(a.choices[0].message.tool_calls, b.choices[0].message.tool_calls)
    c = resp_like(tool_calls=[tc("bash", '{"command": "pwd"}')])
    assert not same_calls(a.choices[0].message.tool_calls, c.choices[0].message.tool_calls)


def test_recheck_kind():
    a = resp_like(tool_calls=[tc("bash", '{"command": "ls"}')])
    assert recheck_kind(None, a) == "changed"
    assert recheck_kind(a, resp_like(tool_calls=[tc("bash", '{"command": "ls"}')])) == "same"
    assert recheck_kind(a, resp_like(tool_calls=[tc("bash", '{"command": "pwd"}')])) == "changed"
    assert recheck_kind(a, resp_like()) == "no_call"


def test_llm_choice_and_call_text():
    assert llm_choice(resp_like()) == NO_TOOL
    assert llm_choice(resp_like(tool_calls=[tc("bash"), tc("read_file")])) == "bash"
    assert call_text(tc("bash", '{"a": 1}')) == 'bash({"a": 1})'


def test_with_notes_appends_to_the_last_message_or_adds_one():
    m = [{"role": "system", "content": "s"}, {"role": "tool", "content": "done"}]
    out = with_notes(m, ["NOTE1", "NOTE2"])
    assert out is not m and out[0] is m[0]
    assert out[-1]["content"] == "done\n\nNOTE1\n\nNOTE2"
    m2 = m + [{"role": "assistant", "content": "x"}]
    out2 = with_notes(m2, ["N"])
    assert out2[-1] == {"role": "user", "content": "N"}
    assert with_notes(m, []) is m


def test_proposal_text_keeps_head_and_tail():
    long = "a" * 4000
    s = proposal_text(resp_like(content=long))
    assert s.startswith("message: aaaa") and s.endswith("aaaa") and " ... " in s
    assert len(s) <= 1500


# ---------------------------------------------------------------------------------------------- asking
def test_ask_record_shape_and_agree():
    a = Ask(point="stop", mode="guard", question="task_done", options=list(STOP.options), tau=0.9,
            act_option="done", p={"done": 0.9, "not_done": 0.1}, label="done", label_tier="strong")
    a.fit_ok = True
    rec = a.record()
    assert rec["point"] == "stop" and rec["choice"] == "done" and rec["p_act"] == 0.9
    assert rec["agree"] is True and rec["fit_ok"] is True
    assert Ask(point="stop", mode="guard", question="q", options=["done"]).ok is False


def test_asker_records_failures_as_asks(run):
    async def main():
        class Boom:
            name = "boom"

            async def decide(self, state, q):
                raise RuntimeError("down")

        a = await Asker(Boom()).ask("stop", "guard", "STATE", STOP, 1.0, 0.9, "done")
        assert not a.ok and a.error.startswith("RuntimeError") and a.p == {}
        assert a.record()["error"] and a.record()["choice"] is None

    run(main())


def test_ask_choice_single_round(run):
    async def main():
        def answer(q):
            return Decision(p=p_over(list(q.options), "read_file", 0.99))

        backend = Scripted({"next_tool": answer})
        opts, descs = tool_options([tc("bash"), tc("read_file")], None)
        a = await Asker(backend).ask_choice("tool_select", "shadow", "S", "t", opts, descs, 5.0, 0.99, 100)
        assert a.ok and a.choice == "read_file" and a.confidence == 0.99 and a.rounds == 1
        assert a.options == opts

    run(main())


def test_ask_choice_knockout_over_many_tools(run):
    async def main():
        # 5 options, max_options 3 -> two chunks, then one final question
        def answer(q):
            return Decision(p=p_over(list(q.options), q.options[0], 0.9))

        backend = Scripted({"next_tool": answer})
        options = ["a", "b", "c", "d", NO_TOOL]
        a = await Asker(backend).ask_choice("tool_select", "shadow", "S", "t", options, [""] * 5, 5.0, 0.99, 3)
        assert a.ok and a.rounds == 2 and a.extra["finalists"] == 3
        assert a.options == options and a.choice == "a"
        assert backend.calls == ["next_tool", "next_tool", "next_tool"]

    run(main())


def test_ask_choice_knockout_error_in_first_round(run):
    async def main():
        def answer(q):
            if q.options == ["d", NO_TOOL]:
                raise RuntimeError("down")
            return Decision(p=p_over(list(q.options), q.options[0], 0.9))

        backend = Scripted({"next_tool": answer})
        a = await Asker(backend).ask_choice("tool_select", "shadow", "S", "t",
                                            ["a", "b", "c", "d", NO_TOOL], [""] * 5, 5.0, 0.99, 3)
        assert not a.ok and a.error is not None and a.rounds == 2
        assert a.options == ["a", "b", "c", "d", NO_TOOL]

    run(main())


def test_pass_through_drops_client_provider_redirects():
    # a client-controlled redirect must never ride along into the strong re-run: it would ship
    # the strong tier's credentials to the client's endpoint
    d = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [],
         "metadata": {"x": 1}, "temperature": 0.2, "max_tokens": 5,
         "base_url": "http://evil.example/v1", "custom_llm_provider": "openai",
         "extra_headers": {"Authorization": "Bearer steal"}, "extra_body": {"a": 1},
         "api_type": "azure", "azure_ad_token": "tok", "organization": "org", "api_key": "k"}
    out = pass_through(d)
    for k in ("base_url", "custom_llm_provider", "extra_headers", "extra_body", "api_type",
              "azure_ad_token", "organization", "api_key", "model", "messages", "tools", "metadata"):
        assert k not in out
    assert out["temperature"] == 0.2 and out["max_tokens"] == 5


def test_asker_carries_the_calibrated_flag(run):
    # decisions rows read `calibrated`: raw and calibrated probabilities are different spaces and one
    # fitted tau must never mix them (the fit eras are cut on model x calibrated)
    from litellm_jah.backend import Question as BQuestion

    class Caled:
        async def decide(self, state, q):
            return Decision(p={"done": 0.8, "not_done": 0.2}, model="m", calibrated=True)

    async def go():
        a = await Asker(Caled()).ask("stop", "shadow", "S",
                                     BQuestion(id="task_done", text="t", options=["done", "not_done"]), 5.0)
        assert a.calibrated and a.record()["calibrated"] is True
        a.error, a.p = "RuntimeError: boom", {}
        assert a.record()["calibrated"] is None  # a failed ask has no distribution, so nothing to place
    run(go())


def test_asker_carries_the_truncated_flag(run):
    # a truncated state is an era marker too: taus fitted on full states do not transfer to tail-only ones
    from litellm_jah.backend import Question as BQuestion

    class Trunc:
        async def decide(self, state, q):
            return Decision(p={"done": 0.8, "not_done": 0.2}, model="m", truncated=True)

    async def go():
        a = await Asker(Trunc()).ask("stop", "shadow", "S",
                                     BQuestion(id="task_done", text="t", options=["done", "not_done"]), 5.0)
        assert a.truncated and a.record()["truncated"] is True
    run(go())
