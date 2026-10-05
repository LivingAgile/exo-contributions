import os
import signal
from typing import cast

import anyio
import pytest

from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CreateRunner,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import channel, mp_channel
from exo.worker.main import Worker
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _DeadProcess:
    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = -6

    def is_alive(self) -> bool:
        return False


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_worker_shutdown_forwards_terminal_task_and_runner(
    request: pytest.FixtureRequest,
) -> None:
    logger_setup(None)
    request.addfinalizer(logger_cleanup)
    event_sender, event_receiver = channel[Event](max_buffer_size=0)
    indexed_sender, indexed_receiver = channel[IndexedEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    node_id = NodeId("shutdown-node")
    runner_id = RunnerId("shutdown-runner")
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("removed-instance"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=runner_id,
        node_id=node_id,
    )
    worker = Worker(
        node_id,
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=0,
    )
    supervisor = await RunnerSupervisor.create(
        bound_instance=bound_instance, event_sender=event_sender.clone()
    )
    worker.runners[runner_id] = supervisor
    shutdown_task: Shutdown | None = None
    observed: list[Event] = []

    with (
        event_sender,
        event_receiver,
        indexed_sender,
        indexed_receiver,
        command_sender,
        command_receiver,
        download_sender,
        download_receiver,
        anyio.fail_after(20),
    ):
        async with anyio.create_task_group() as task_group:
            try:
                task_group.start_soon(supervisor.run)
                observed.append(await event_receiver.receive())
                task_group.start_soon(worker.plan_step)
                while True:
                    event = await event_receiver.receive()
                    observed.append(event)
                    if isinstance(event, TaskCreated) and isinstance(
                        event.task, Shutdown
                    ):
                        shutdown_task = event.task
                    if isinstance(event, RunnerStatusUpdated) and isinstance(
                        event.runner_status, RunnerShuttingDown
                    ):
                        break

                await anyio.sleep(0.1)
                with anyio.move_on_after(1):
                    while True:
                        event = await event_receiver.receive()
                        observed.append(event)
                        if isinstance(event, RunnerStatusUpdated) and isinstance(
                            event.runner_status, RunnerShutdown
                        ):
                            break
            finally:
                task_group.cancel_scope.cancel()

    assert shutdown_task is not None
    assert any(
        isinstance(event, TaskStatusUpdated)
        and event.task_id == shutdown_task.task_id
        and event.task_status == TaskStatus.Complete
        for event in observed
    ), observed
    assert any(
        isinstance(event, RunnerStatusUpdated)
        and event.runner_id == runner_id
        and isinstance(event.runner_status, RunnerShutdown)
        for event in observed
    ), observed


@pytest.mark.anyio
@pytest.mark.skipif(
    not hasattr(signal, "SIGSTOP"), reason="requires native POSIX child suspension"
)
async def test_worker_waits_for_old_process_before_planning_replacement(
    request: pytest.FixtureRequest,
) -> None:
    logger_setup(None)
    request.addfinalizer(logger_cleanup)
    event_sender, event_receiver = channel[Event](max_buffer_size=0)
    indexed_sender, indexed_receiver = channel[IndexedEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    node_id = NodeId("replacement-node")
    runner_id = RunnerId("replacement-runner")
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("replacement-instance"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=runner_id,
        node_id=node_id,
    )
    peer_runner_id = next(
        assigned_runner
        for assigned_runner in bound_instance.instance.shard_assignments.runner_to_shard
        if assigned_runner != runner_id
    )
    worker = Worker(
        node_id,
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=0,
    )
    worker.state = worker.state.model_copy(
        update={
            "instances": {
                bound_instance.instance.instance_id: bound_instance.instance,
            },
            "runners": {
                peer_runner_id: RunnerFailed(
                    error_message="peer failed", diagnostics=[]
                ),
            },
        }
    )
    supervisor = await RunnerSupervisor.create(
        bound_instance=bound_instance, event_sender=event_sender.clone()
    )
    worker.runners[runner_id] = supervisor
    shutdown_task: Shutdown | None = None
    shutdown_timed_out = False

    with (
        event_sender,
        event_receiver,
        indexed_sender,
        indexed_receiver,
        command_sender,
        command_receiver,
        download_sender,
        download_receiver,
        anyio.fail_after(60),
    ):
        async with anyio.create_task_group() as task_group:
            try:
                task_group.start_soon(supervisor.run)
                await event_receiver.receive()
                os.kill(supervisor.runner_process.pid, signal.SIGSTOP)
                task_group.start_soon(worker.plan_step)
                while True:
                    event = await event_receiver.receive()
                    if isinstance(event, TaskCreated) and isinstance(
                        event.task, Shutdown
                    ):
                        shutdown_task = event.task
                        worker.state = worker.state.model_copy(
                            update={
                                "runners": {
                                    **worker.state.runners,
                                    peer_runner_id: RunnerIdle(),
                                },
                            }
                        )
                    if (
                        isinstance(event, TaskStatusUpdated)
                        and shutdown_task is not None
                        and event.task_id == shutdown_task.task_id
                        and event.task_status == TaskStatus.TimedOut
                    ):
                        shutdown_timed_out = True
                    if isinstance(event, TaskCreated) and isinstance(
                        event.task, CreateRunner
                    ):
                        assert shutdown_timed_out
                        assert not supervisor.runner_process.is_alive(), (
                            "Replacement planned while the old runner process is alive"
                        )
                        break
            finally:
                if supervisor.runner_process.is_alive():
                    os.kill(supervisor.runner_process.pid, signal.SIGCONT)
                supervisor.shutdown()
                task_group.cancel_scope.cancel()
