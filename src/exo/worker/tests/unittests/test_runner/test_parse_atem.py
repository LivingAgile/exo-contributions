import json
from collections.abc import Generator
from typing import Any, cast

import pytest
from mlx_lm.models.muse_glimmer import Model as MuseGlimmerModel

from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    apply_all_parsers,
    parse_atem_output,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather.get_forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clock.now",
            "parameters": {
                "type": "object",
                "properties": {"utc": {"type": "boolean"}},
                "required": ["utc"],
            },
        },
    },
]


def _responses(parts: list[str]) -> Generator[GenerationResponse]:
    for index, part in enumerate(parts):
        yield GenerationResponse(
            text=part,
            token=index,
            finish_reason="stop" if index == len(parts) - 1 else None,
            usage=None,
        )


def _collect(parts: list[str]):
    return list(parse_atem_output(_responses(parts), TOOLS))


def _collect_with_finish_reason(text: str, finish_reason: str):
    responses = iter(
        [
            GenerationResponse(
                text=text,
                token=0,
                finish_reason=finish_reason,
                usage=None,
            )
        ]
    )
    return list(parse_atem_output(responses, TOOLS))


def test_plain_text_passes_through_and_state_is_per_request():
    first = _collect(["plain ", "answer"])
    second = _collect(["another answer"])

    assert "".join(item.text for item in first if isinstance(item, GenerationResponse)) == "plain answer"
    assert first[-1].finish_reason == "stop"
    assert second[0].text == "another answer"
    assert second[0].finish_reason == "stop"


def test_reasoning_and_visible_content_survive_every_character_split():
    output = "to=self<|message|>private plan<|eom|>to=user<|message|>public answer<|eot|>"

    for split in range(len(output) + 1):
        results = _collect([output[:split], output[split:]])
        generated = [item for item in results if isinstance(item, GenerationResponse)]
        assert "".join(item.text for item in generated if item.is_thinking) == "private plan"
        assert "".join(item.text for item in generated if not item.is_thinking) == "public answer"
        assert generated[-1].finish_reason == "stop"
        assert all("<|" not in item.text and "to=" not in item.text for item in generated)


def test_apply_all_parsers_routes_muse_tool_calls_to_structured_chunk():
    body = (
        "to=weather<|message|><atem:function_calls>"
        '<atem:invoke name="weather.get_forecast">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="days">3</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>"
    )
    chunks = [
        chunk
        for chunk in apply_all_parsers(
            _responses(list(body)),
            prompt="",
            tool_parser=None,
            tokenizer=cast(Any, object()),
            model_type=MuseGlimmerModel,
            model_id=ModelId("meta-models/Muse-Glimmer-30B"),
            tools=TOOLS,
        )
        if chunk is not None
    ]

    assert len(chunks) == 1
    assert isinstance(chunks[0], ToolCallChunk)
    assert chunks[0].finish_reason == "tool_calls"
    assert chunks[0].tool_calls[0].name == "weather.get_forecast"
    assert json.loads(chunks[0].tool_calls[0].arguments) == {"city": "Paris", "days": 3}


def test_multiple_calls_preserve_source_order_and_types():
    body = (
        "to=functions<|message|><atem:function_calls>"
        '<atem:invoke name="weather.get_forecast">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="days">3</atem:parameter>'
        "</atem:invoke>"
        '<atem:invoke name="clock.now">'
        '<atem:parameter name="utc">true</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>"
    )
    results = _collect([body])

    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert [call.name for call in results[0].tool_calls] == [
        "weather.get_forecast",
        "clock.now",
    ]
    assert json.loads(results[0].tool_calls[1].arguments) == {"utc": True}


@pytest.mark.parametrize(
    "body",
    [
        "to=self<|message|>unfinished",
        "to=weather<|message|><atem:function_calls></atem:function_calls><|eot|>",
        "to=clock<|message|><atem:function_calls>"
        '<atem:invoke name="weather.get_forecast">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="days">3</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>",
        "to=weather<|message|><atem:function_calls>"
        '<atem:invoke name="weather.get_forecast">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="city">Lyon</atem:parameter>'
        '<atem:parameter name="days">3</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>",
        "to=weather<|message|><atem:function_calls>"
        '<atem:invoke name="weather.get_forecast">'
        '<atem:parameter name="city">Paris</atem:parameter>'
        '<atem:parameter name="days">many</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>",
        "to=user<|message|>answer<|eot|>trailing",
    ],
)
def test_malformed_or_truncated_protocol_fails_closed(body: str):
    results = _collect(list(body))

    assert len(results) == 1
    assert isinstance(results[0], GenerationResponse)
    assert results[0].finish_reason == "error"
    assert "<atem:" not in results[0].text
    assert "to=" not in results[0].text


def test_complete_protocol_with_non_stop_finish_reason_fails_closed():
    results = _collect_with_finish_reason(
        "to=user<|message|>partial answer<|eot|>", "length"
    )

    assert len(results) == 1
    assert isinstance(results[0], GenerationResponse)
    assert results[0].finish_reason == "error"
    assert "partial answer" not in results[0].text


def test_malformed_protocol_maps_to_error_chunk_at_production_boundary():
    chunks = [
        chunk
        for chunk in apply_all_parsers(
            _responses(["to=self<|message|>unfinished"]),
            prompt="",
            tool_parser=None,
            tokenizer=cast(Any, object()),
            model_type=MuseGlimmerModel,
            model_id=ModelId("meta-models/Muse-Glimmer-30B"),
            tools=TOOLS,
        )
        if chunk is not None
    ]

    assert len(chunks) == 1
    assert isinstance(chunks[0], ErrorChunk)
    assert not isinstance(chunks[0], TokenChunk)