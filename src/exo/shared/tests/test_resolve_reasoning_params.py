import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.adapters.responses import responses_request_to_text_generation
from exo.api.types import ChatCompletionRequest
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.models.model_cards import card_cache, load_builtin_model_card
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import resolve_reasoning_params


@pytest.mark.parametrize(
    ("settings", "expected_effort", "expected_thinking"),
    [
        ({}, "low", True),
        ({"enable_thinking": True}, "low", True),
        ({"reasoning_effort": "high"}, "high", True),
        ({"reasoning_effort": "medium", "enable_thinking": True}, "medium", True),
        ({"enable_thinking": False}, "none", False),
        ({"reasoning_effort": "none"}, "none", False),
    ],
)
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("cache_state", ["builtin", "empty", "legacy"])
async def test_adapters_use_card_reasoning_default(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, str | bool],
    expected_effort: str,
    expected_thinking: bool,
    api: str,
    cache_state: str,
) -> None:
    model_id = ModelId("mlx-community/GLM-5.3-MLX-8bit")
    card = await load_builtin_model_card(model_id)
    assert card is not None
    assert card.default_reasoning_effort == "low"
    monkeypatch.setattr(card_cache, "cc", {})
    if cache_state == "builtin":
        card_cache.cc[model_id] = card
    elif cache_state == "legacy":
        card_cache.cc[model_id] = card.model_copy(
            update={"default_reasoning_effort": None, "is_custom": True}
        )
    if api == "chat":
        request = ChatCompletionRequest.model_validate(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return 17."}],
                **settings,
            }
        )
        params = await chat_request_to_text_generation(request)
    else:
        responses_settings: dict[str, object] = dict(settings)
        if "reasoning_effort" in responses_settings:
            responses_settings["reasoning"] = {
                "effort": responses_settings.pop("reasoning_effort")
            }
        responses_request = ResponsesRequest.model_validate(
            {"model": model_id, "input": "Return 17.", **responses_settings}
        )
        params = await responses_request_to_text_generation(responses_request)
    assert params.reasoning_effort == expected_effort
    assert params.enable_thinking == expected_thinking

    unrelated = await chat_request_to_text_generation(
        ChatCompletionRequest.model_validate(
            {
                "model": "unrelated/model",
                "messages": [{"role": "user", "content": "Return 17."}],
                "enable_thinking": True,
            }
        )
    )
    assert unrelated.reasoning_effort == "medium"
    assert unrelated.enable_thinking is True


async def test_both_none_returns_none_none() -> None:
    assert await resolve_reasoning_params(None, None) == (None, None)


async def test_both_set_passes_through_unchanged() -> None:
    assert await resolve_reasoning_params("high", True) == ("high", True)
    assert await resolve_reasoning_params("none", True) == ("none", True)
    assert await resolve_reasoning_params("low", False) == ("low", False)


async def test_enable_thinking_true_derives_medium() -> None:
    assert await resolve_reasoning_params(None, True) == ("medium", True)


async def test_enable_thinking_false_derives_none() -> None:
    assert await resolve_reasoning_params(None, False) == ("none", False)


async def test_reasoning_effort_none_derives_thinking_false() -> None:
    assert await resolve_reasoning_params("none", None) == ("none", False)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
async def test_non_none_effort_derives_thinking_true(effort: str) -> None:
    assert await resolve_reasoning_params(effort, None) == (effort, True)  # pyright: ignore[reportArgumentType]
