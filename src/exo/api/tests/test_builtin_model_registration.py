from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from exo.api.main import API
from exo.api.types import AddCustomModelParams
from exo.shared.election import ElectionMessage
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.commands import (
    AddCustomModelCard,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import IndexedEvent
from exo.utils.channels import Receiver, channel


class RegistrationAPI(API):
    def close_event_log(self) -> None:
        self._event_log.close()


@pytest.fixture
def registration_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[RegistrationAPI, Receiver[ForwarderCommand]]]:
    monkeypatch.setattr("exo.api.main._API_EVENT_LOG_DIR", tmp_path / "events")
    monkeypatch.setattr("exo.api.main.EXO_IMAGE_CACHE_DIR", tmp_path / "images")
    with ExitStack() as resources:
        event_sender, event_receiver = channel[IndexedEvent]()
        command_sender, command_receiver = channel[ForwarderCommand]()
        download_sender, download_receiver = channel[ForwarderDownloadCommand]()
        election_sender, election_receiver = channel[ElectionMessage]()
        for resource in (
            event_sender,
            event_receiver,
            command_sender,
            command_receiver,
            download_sender,
            download_receiver,
            election_sender,
            election_receiver,
        ):
            resources.enter_context(resource)
        api = RegistrationAPI(
            NodeId("registration-test"),
            port=0,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
        )
        resources.callback(api.close_event_log)
        yield api, command_receiver


@pytest.mark.asyncio
async def test_add_builtin_deepseek_v4_preserves_card(
    monkeypatch: pytest.MonkeyPatch,
    registration_api: tuple[RegistrationAPI, Receiver[ForwarderCommand]],
) -> None:
    monkeypatch.setattr(model_cards.card_cache, "cc", {})

    async def fail_fetch(_model_id: object) -> ModelCard:
        raise AssertionError("built-in models must not be re-fetched from Hugging Face")

    monkeypatch.setattr(ModelCard, "fetch_from_hf", fail_fetch)

    api, commands = registration_api

    result = await api.add_custom_model(
        AddCustomModelParams(model_id=ModelId("mlx-community/DeepSeek-V4-Flash"))
    )

    assert result.base_model == "DeepSeek V4 Flash"
    assert result.supports_tensor is True
    assert result.is_custom is False
    assert commands.collect() == []


@pytest.mark.asyncio
async def test_add_builtin_deepseek_v41_preserves_exact_official_card(
    monkeypatch: pytest.MonkeyPatch,
    registration_api: tuple[RegistrationAPI, Receiver[ForwarderCommand]],
) -> None:
    monkeypatch.setattr(model_cards.card_cache, "cc", {})

    async def fail_fetch(_model_id: object) -> ModelCard:
        raise AssertionError("built-in models must not be re-fetched from Hugging Face")

    monkeypatch.setattr(ModelCard, "fetch_from_hf", fail_fetch)

    api, commands = registration_api
    builtin = await model_cards.load_builtin_model_card(
        ModelId("deepseek-ai/DeepSeek-V4.1-Flash")
    )
    assert builtin is not None
    assert builtin.n_layers == 40
    assert builtin.hidden_size == 5120

    result = await api.add_custom_model(
        AddCustomModelParams(model_id=ModelId("deepseek-ai/DeepSeek-V4.1-Flash"))
    )

    assert result.base_model == "DeepSeek V4.1 Flash"
    assert result.supports_tensor is True
    assert result.is_custom is False
    assert commands.collect() == []


@pytest.mark.asyncio
async def test_add_unknown_model_fetches_and_synchronizes_custom_card(
    monkeypatch: pytest.MonkeyPatch,
    registration_api: tuple[RegistrationAPI, Receiver[ForwarderCommand]],
) -> None:
    monkeypatch.setattr(model_cards.card_cache, "cc", {})
    builtin = await model_cards.load_builtin_model_card(
        ModelId("mlx-community/DeepSeek-V4-Flash")
    )
    assert builtin is not None
    custom = builtin.model_copy(
        update={
            "model_id": ModelId("custom/model"),
            "base_model": "",
            "is_custom": True,
        }
    )

    async def fetch(_model_id: object) -> ModelCard:
        return custom

    monkeypatch.setattr(ModelCard, "fetch_from_hf", fetch)

    api, commands = registration_api

    result = await api.add_custom_model(
        AddCustomModelParams(model_id=ModelId("custom/model"))
    )

    assert result.id == "custom/model"
    assert result.is_custom is True
    sent = commands.collect()
    assert len(sent) == 1
    assert isinstance(sent[0].command, AddCustomModelCard)
    assert sent[0].command.model_card == custom
