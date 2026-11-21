"""The force-field helpers: counting stonewalled customer replies and the note that forces the correct ask."""

from __future__ import annotations

from litellm_jah.points import _count_stonewalls, force_ask_note

STONEWALLS = [
    "I think I already gave you everything I have.",
    "That's all I can tell you.",
    "No further info on my end.",
    "Nothing more to add.",
    "I don't have anything else.",
]
NORMAL = [
    "Please ship my order ORD-1234 as soon as possible, thanks so much for your help.",
    "I would like a refund to my original payment method for the damaged item.",
    "What is the current status of my order and when will it arrive at my house?",
]


def test_count_stonewalls_counts_deflections_ignores_normal():
    msgs = [{"role": "user", "content": t} for t in STONEWALLS + NORMAL]
    assert _count_stonewalls(msgs) == len(STONEWALLS)


def test_count_stonewalls_ignores_long_and_empty():
    # a stonewall phrase but longer than 150 chars is not a deflection; empty content counts as nothing
    long = "I think I already told you. " * 12  # >150 chars
    msgs = [{"role": "user", "content": long}, {"role": "user", "content": ""},
            {"role": "user", "content": "Nothing else."}]
    assert _count_stonewalls(msgs) == 1


def test_count_stonewalls_handles_list_content_parts():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "That's all on my end."}]},
        {"role": "user", "content": [{"type": "text", "text": "Tell me about your shipping options."},
                                       {"type": "text", "text": "No further info."}]},
        {"role": "user", "content": [{"type": "text", "text": "What should I do about my refund?"}]},
    ]
    assert _count_stonewalls(msgs) == 2


def test_count_stonewalls_zero_for_a_clean_history():
    assert _count_stonewalls([{"role": "user", "content": t} for t in NORMAL]) == 0


def test_force_ask_note_names_the_field_and_ask_user():
    for tool, field in (("ship_order", "shipping_speed"), ("refund", "refund_method"),
                        ("cancel_order", "order_id")):
        note = force_ask_note(tool, 3)
        assert note is not None
        assert "ask_user" in note
        assert f"missing='{field}'" in note
        assert "Do NOT call" in note


def test_force_ask_note_is_none_for_a_non_force_tool():
    assert force_ask_note("read", 5) is None
    assert force_ask_note("bash", 5) is None


def test_count_stonewalls_empty_history():
    assert _count_stonewalls([]) == 0


def test_count_stonewalls_ignores_agent_and_system_messages():
    # the agent's own "I think…" (assistant) and system messages are NOT customer deflections:
    # only the customer's user/tool messages count
    msgs = [
        {"role": "assistant", "content": "I think I should ship ORD-1234 now."},
        {"role": "system", "content": "That's all the information available on this order."},
        {"role": "user", "content": "I think I already gave you everything."},
        {"role": "tool", "content": "Customer: Nothing more to add."},
    ]
    assert _count_stonewalls(msgs) == 2
