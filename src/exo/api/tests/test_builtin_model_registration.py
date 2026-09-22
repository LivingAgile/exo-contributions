import asyncio
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from exo.api.main import API
from exo.api.types import AddCustomModelParams, ModelListModel
from exo.shared.election import ElectionMessage
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    AddCustomModelCard,
    ForwarderCommand,
    ForwarderDownloadCommand,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    Event,
    IndexedEvent,
    InstanceCreated,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import ShardAssignments
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker


class RegistrationAPI(API):
    def close_event_log(self) -> None:
        self._event_log.close()

    async def apply_events(self) -> None:
        await self._apply_state()

    def has_stream(self, command_id: CommandId) -> bool:
        return command_id in self._text_generation_queues


class CardReplayWorker(Worker):
    async def replay_cards(self) -> None:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(self._event_applier)
            tasks.start_soon(self._reconcile_custom_cards)


@pytest.fixture
def registration_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[RegistrationAPI, Receiver[ForwarderCommand], Sender[IndexedEvent]]]:
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
        yield api, command_receiver, event_sender


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "base_model", "default_effort"),
    [
        ("mlx-community/DeepSeek-V4-Flash", "DeepSeek V4 Flash", None),
        ("mlx-community/GLM-5.3-MLX-8bit", "", "low"),
    ],
)
async def test_add_builtin_preserves_card(
    monkeypatch: pytest.MonkeyPatch,
    registration_api: tuple[
        RegistrationAPI, Receiver[ForwarderCommand], Sender[IndexedEvent]
    ],
    model_id: str,
    base_model: str,
    default_effort: str | None,
) -> None:
    monkeypatch.setattr(model_cards.card_cache, "cc", {})

    async def fail_fetch(_model_id: object) -> ModelCard:
        raise AssertionError("built-in models must not be re-fetched from Hugging Face")

    monkeypatch.setattr(ModelCard, "fetch_from_hf", fail_fetch)

    api, commands, _ = registration_api

    async with AsyncClient(
        transport=ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post("/models/add", json={"model_id": model_id})
    assert response.status_code == 200
    result = ModelListModel.model_validate_json(response.content)
    assert result.base_model == base_model
    assert result.supports_tensor is True
    assert result.is_custom is False
    card = model_cards.card_cache.get(ModelId(model_id))
    assert card is not None
    assert card.default_reasoning_effort == default_effort
    assert commands.collect() == []


@pytest.mark.asyncio
async def test_add_unknown_model_fetches_and_synchronizes_custom_card(
    monkeypatch: pytest.MonkeyPatch,
    registration_api: tuple[
        RegistrationAPI, Receiver[ForwarderCommand], Sender[IndexedEvent]
    ],
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

    api, commands, _ = registration_api

    result = await api.add_custom_model(
        AddCustomModelParams(model_id=ModelId("custom/model"))
    )

    assert result.id == "custom/model"
    assert result.is_custom is True
    sent = commands.collect()
    assert len(sent) == 1
    assert isinstance(sent[0].command, AddCustomModelCard)
    assert sent[0].command.model_card == custom


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("replay_legacy", [False, True])
@pytest.mark.parametrize(
    ("settings", "effort", "thinking"),
    [
        ({}, "low", True),
        ({"enable_thinking": True}, "low", True),
        ({"reasoning_effort": "high"}, "high", True),
        ({"enable_thinking": False}, "none", False),
    ],
)
async def test_http_generation_preserves_reasoning_after_card_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registration_api: tuple[
        RegistrationAPI, Receiver[ForwarderCommand], Sender[IndexedEvent]
    ],
    endpoint: str,
    replay_legacy: bool,
    settings: dict[str, str | bool],
    effort: str,
    thinking: bool,
) -> None:
    model_id = ModelId("mlx-community/GLM-5.3-MLX-8bit")
    builtin = await model_cards.load_builtin_model_card(model_id)
    assert builtin is not None
    legacy = builtin.model_copy(
        update={"default_reasoning_effort": None, "is_custom": True}
    )
    monkeypatch.setattr(model_cards.card_cache, "cc", {})
    monkeypatch.setattr(
        model_cards, "_custom_cards_dir", anyio.Path(tmp_path / "cards")
    )
    api, commands, events = registration_api
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=model_id, node_to_runner={}, runner_to_shard={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    payload: dict[str, object] = {"model": str(model_id), "max_tokens": 16, **settings}
    if endpoint == "/v1/chat/completions":
        payload["messages"] = [{"role": "user", "content": "Return 17."}]
    else:
        payload["input"] = "Return 17."
        if "reasoning_effort" in payload:
            payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}

    with ExitStack() as resources:
        replay_sender, replay_receiver = channel[IndexedEvent]()
        output_sender, output_receiver = channel[Event]()
        for resource in (
            replay_sender,
            replay_receiver,
            output_sender,
            output_receiver,
        ):
            resources.enter_context(resource)
        worker = CardReplayWorker(
            NodeId("card-replay-test"),
            event_receiver=replay_receiver,
            event_sender=output_sender,
            command_sender=api.command_sender,
            download_command_sender=api.download_command_sender,
            api_port=0,
        )
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(api.apply_events)
                if replay_legacy:
                    tasks.start_soon(worker.replay_cards)
                    await replay_sender.send(
                        IndexedEvent(
                            idx=0, event=CustomModelCardAdded(model_card=legacy)
                        )
                    )
                    persisted = anyio.Path(
                        tmp_path / "cards" / (model_id.normalize() + ".toml")
                    )
                    while not await persisted.exists():
                        await anyio.sleep(0.01)
                    assert model_cards.card_cache.get(model_id) == legacy
                    restored = await ModelCard.load_from_path(persisted)
                    assert restored.default_reasoning_effort is None

                await events.send(
                    IndexedEvent(idx=0, event=InstanceCreated(instance=instance))
                )
                while instance.instance_id not in api.state.instances:
                    await anyio.sleep(0.01)
                async with AsyncClient(
                    transport=ASGITransport(app=api.app), base_url="http://test"
                ) as client:
                    request_task = asyncio.create_task(
                        client.post(endpoint, json=payload)
                    )
                    try:
                        forwarded = await commands.receive()
                        command = forwarded.command
                        assert isinstance(command, TextGeneration)
                        assert command.task_params.model == model_id
                        assert command.task_params.reasoning_effort == effort
                        assert command.task_params.enable_thinking is thinking
                        while not api.has_stream(command.command_id):
                            await anyio.sleep(0.01)
                        await events.send(
                            IndexedEvent(
                                idx=1,
                                event=ChunkGenerated(
                                    command_id=command.command_id,
                                    chunk=TokenChunk(
                                        model=model_id,
                                        text="17",
                                        token_id=17,
                                        finish_reason="stop",
                                        usage=None,
                                    ),
                                ),
                            )
                        )
                        response = await request_task
                        assert response.status_code == 200
                        assert "17" in response.text
                        finished = (await commands.receive()).command
                        assert isinstance(finished, TaskFinished)
                        assert finished.finished_command_id == command.command_id
                    finally:
                        request_task.cancel()
                        await asyncio.gather(request_task, return_exceptions=True)
                tasks.cancel_scope.cancel()
