import pytest

from exo.worker.engines.mlx.vendor.deepseek_v41_encoding import (
    ASSISTANT_SP_TOKEN,
    USER_SP_TOKEN,
    dsml_token,
    encode_messages,
)


@pytest.mark.parametrize("effort", [50, 75, 100])
def test_numeric_reasoning_effort_is_rendered_exactly(effort: int) -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "hello"}],
        thinking_mode="thinking",
        reasoning_effort=effort,
    )

    assert f"Reasoning Effort: {effort} (range 1-100" in prompt


def test_mid_conversation_system_message_preserves_turn_order() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "initial"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "system", "content": "updated"},
            {"role": "user", "content": "second"},
        ],
        thinking_mode="thinking",
        reasoning_effort=75,
    )

    first_user = prompt.index(f"{USER_SP_TOKEN}first")
    prior_assistant = prompt.index(f"{ASSISTANT_SP_TOKEN}</think>answer")
    updated_system = prompt.index("<｜System｜>updated")
    second_user = prompt.index(f"{USER_SP_TOKEN}second")
    assert first_user < prior_assistant < updated_system < second_user
    assert prompt.endswith(f"{ASSISTANT_SP_TOKEN}<think>")


def test_namespaced_tool_and_following_result_keep_protocol_order() -> None:
    messages = [
        {
            "role": "system",
            "content": "",
            "tools": [
                {
                    "type": "function",
                    "namespace": {"name": "search", "description": "Lookup"},
                    "function": {
                        "name": "find",
                        "description": "Find a record",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search::find", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "found"},
        {"role": "user", "content": "continue"},
    ]

    prompt = encode_messages(
        messages,
        thinking_mode="thinking",
        reasoning_effort=75,
    )

    schema = '"name": "search::find"'
    invocation = f'<{dsml_token} invoke name="search::find">'
    result = "<tool_result>found</tool_result>"
    continuation = f"{USER_SP_TOKEN}"
    assert prompt.index(schema) < prompt.index(invocation)
    assert prompt.index(invocation) < prompt.rindex(continuation)
    assert prompt.rindex(continuation) < prompt.index(result)
    assert prompt.index(result) < prompt.index("continue")