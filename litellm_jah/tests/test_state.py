"""State rendering on plain OpenAI messages: the format the decision models were trained on."""

from __future__ import annotations

from litellm_jah.state import (
    LAST_OUTPUT_CHARS,
    TASK_CHARS,
    _trunc,
    episode,
    render_state,
    session_key,
    step_index,
)

from conftest import msgs


def test_step_index_counts_assistant_turns():
    assert step_index(msgs(0)) == 0
    assert step_index(msgs(1)) == 1
    assert step_index(msgs(3)) == 3


def test_episode_task_and_steps():
    m = msgs(2, task="Fix the flaky test")
    task, steps = episode(m)
    assert task == "Fix the flaky test"
    assert len(steps) == 2
    assert steps[0].action == 'bash({"command": "ls -la 0"})'
    assert steps[0].output == "result 0: all good"
    assert steps[1].output == "result 1: all good"


def test_episode_message_step_and_user_observations():
    m = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "I will now do it."},
        {"role": "user", "content": "observation: ok"},
    ]
    task, steps = episode(m)
    assert task == "do it"
    assert steps[0].action == "message: I will now do it."
    assert steps[0].output == "observation: ok"  # user messages after an assistant are observations


def test_render_state_layout_and_truncation():
    big = "x" * (LAST_OUTPUT_CHARS + 100)
    m = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t0", "content": big},
    ]
    state, view = render_state(m)
    assert state.startswith("TASK: task\n\nSTEPS SO FAR: 1\n[1] ACTION: bash({})\n    RESULT: ")
    # the last step's result keeps its tail at LAST_OUTPUT_CHARS
    assert state.endswith(big[-LAST_OUTPUT_CHARS:])
    assert len(state.split("    RESULT: ")[1]) == LAST_OUTPUT_CHARS + 4
    assert view.last_action == "bash({})"
    assert view.last_output == big and view.prev_action == ""


def test_render_state_many_steps_keeps_last_six():
    state, _ = render_state(msgs(9))
    assert "STEPS SO FAR: 9" in state
    assert "[4] ACTION:" in state and "[1] ACTION:" not in state
    assert "[9] ACTION:" in state


def test_render_state_truncates_long_task():
    task = "t" * (TASK_CHARS + 50)
    m = [{"role": "user", "content": task}]
    state, _ = render_state(m)
    task_line = state.splitlines()[0]
    assert len(task_line) <= TASK_CHARS + len("TASK: ") + 1


def test_msg_text_handles_content_parts():
    m = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "u"},
                                      {"type": "text", "text": " world"}]}]
    task, _ = episode(m)
    assert task == "hello  world"


def test_episode_skips_an_empty_first_user():
    m = [{"role": "user", "content": ""}, {"role": "user", "content": "do it"}]
    task, _ = episode(m)
    assert task == "do it" # the standalone takes the first user message WITH text


def test_session_key_matches_the_standalone_fallback():
    # the standalone hashes ALL non-empty system messages joined with a blank line, and the first user
    # message with text: a second system message and an empty first user turn must not change the key
    m = [{"role": "system", "content": "base"},
         {"role": "user", "content": ""},
         {"role": "user", "content": "do it"},
         {"role": "system", "content": "extra"}]
    ref = [{"role": "system", "content": "base\n\nextra"}, {"role": "user", "content": "do it"}]
    assert session_key(m) == session_key(ref)
    assert session_key(m) != session_key([{"role": "system", "content": "base"}, {"role": "user", "content": "do it"}])


def test_session_key_explicit_and_hashed():
    m = msgs(0, task="Fix the bug")
    assert session_key(m, {"jah_session": "abc"}) == "abc"
    assert session_key(m, {"session_id": "s1"}) == "s1"
    k1 = session_key(m)
    k2 = session_key(msgs(1, task="Fix the bug"))  # the same run, one step later
    assert k1 == k2 and len(k1) == 16
    assert k1 != session_key(msgs(0, task="A different task"))
    assert k1 != session_key(msgs(0, task="Fix the bug", system="other"))


def test_trunc_small_n_never_exceeds_its_budget():
    for n in range(1, 15):
        assert len(_trunc("abcdefgh ijklmnop", n)) <= n
