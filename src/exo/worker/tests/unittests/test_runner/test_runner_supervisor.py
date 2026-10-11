import contextlib
import ctypes
import gc
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from multiprocessing.process import BaseProcess
from typing import BinaryIO, cast

import anyio
import pytest
from anyio import to_thread

from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    CreateRunner,
    GenerationTask,
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
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Sender, channel, mp_channel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.main import Worker
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.runner import Runner
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


def _process_with_exit_gate(
    target: Callable[..., object],
    target_args: tuple[object, ...],
    target_kwargs: Mapping[str, object],
    ready_sender: MpSender[None],
    release_receiver: MpReceiver[None],
) -> None:
    target(*target_args, **target_kwargs)
    try:
        ready_sender.send(None)
        release_receiver.receive()
    finally:
        ready_sender.close()
        release_receiver.close()


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
async def test_worker_cancels_while_queued_generation_awaits_acknowledgement() -> None:
    compute_entered = threading.Event()
    compute_released = threading.Event()

    class BlockingCompute(Engine):
        def __init__(self) -> None:
            self.task: GenerationTask | None = None
            self.blocked = False
            self._cancelled_tasks = set()

        def warmup(self) -> None:
            pass

        def submit(self, task: GenerationTask) -> None:
            self.task = task

        def step(self) -> Iterable[tuple[TaskId, FinishedResponse]]:
            assert self.task is not None
            if not self.blocked:
                self.blocked = True
                compute_entered.set()
                assert compute_released.wait(10), "Test compute was not released"
            return [(self.task.task_id, FinishedResponse())]

        def close(self) -> None:
            compute_released.set()

        def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
            raise NotImplementedError

    with ExitStack() as resources:
        event_sender, event_receiver = channel[Event]()
        indexed_sender, indexed_receiver = channel[IndexedEvent]()
        command_sender, command_receiver = channel[ForwarderCommand]()
        download_sender, download_receiver = channel[ForwarderDownloadCommand]()
        task_sender, task_receiver = mp_channel[Task]()
        cancel_sender, cancel_receiver = mp_channel[TaskId]()
        runner_event_sender, runner_event_receiver = mp_channel[
            Event | RunnerTerminationError
        ]()
        for resource in (
            event_sender,
            event_receiver,
            indexed_sender,
            indexed_receiver,
            command_sender,
            command_receiver,
            download_sender,
            download_receiver,
            task_sender,
            cancel_sender,
            cancel_receiver,
            runner_event_sender,
            runner_event_receiver,
        ):
            resources.enter_context(resource)
        node_id = NodeId("cancel-ack-node")
        runner_id = RunnerId("cancel-ack-runner")
        bound = get_bound_mlx_ring_instance(
            instance_id=InstanceId("cancel-ack-instance"),
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
        process = cast(AsyncProcess, cast(object, _DeadProcess()))
        handler = await RunnerStdioHandler.create(
            stdout_rx=process.stdout, stderr_rx=process.stderr
        )
        supervisor = RunnerSupervisor(
            shard_metadata=bound.bound_shard,
            bound_instance=bound,
            runner_process=process,
            _runner_stdio_handler=handler,
            initialize_timeout=400,
            _ev_recv=runner_event_receiver,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _cancel_sender=cancel_sender,
        )
        compute = BlockingCompute()
        runner = Runner(
            bound,
            cast(Builder, cast(object, compute)),
            cast(MpSender[Event], cast(object, runner_event_sender)),
            task_receiver,
        )
        runner.generator = compute
        runner.update_status(RunnerReady())
        worker.runners[runner_id] = supervisor
        tasks = [
            TextGeneration(
                task_id=TaskId(name),
                instance_id=bound.instance.instance_id,
                command_id=CommandId(name),
                task_params=TextGenerationTaskParams(
                    model=bound.bound_shard.model_card.model_id,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                    stream=True,
                ),
            )
            for name in ("active-generation", "queued-generation", "third-generation")
        ]
        worker.state = worker.state.model_copy(
            update={
                "instances": {bound.instance.instance_id: bound.instance},
                "runners": {
                    assigned: RunnerReady()
                    for assigned in bound.instance.shard_assignments.runner_to_shard
                },
                "tasks": {tasks[0].task_id: tasks[0]},
            }
        )
        runner_thread = threading.Thread(target=runner.main)
        runner_thread.start()
        try:
            with anyio.fail_after(8):
                async with anyio.create_task_group() as task_group:
                    try:
                        task_group.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
                        while not isinstance(supervisor.status, RunnerReady):
                            await anyio.sleep(0.01)
                        task_group.start_soon(worker.plan_step)
                        while not compute_entered.is_set():
                            await anyio.sleep(0.01)
                        worker.state = worker.state.model_copy(
                            update={
                                "tasks": {
                                    **worker.state.tasks,
                                    tasks[1].task_id: tasks[1],
                                }
                            }
                        )
                        while tasks[1].task_id not in supervisor.pending:
                            await anyio.sleep(0.01)
                        assert not compute_released.is_set()
                        worker.state = worker.state.model_copy(
                            update={
                                "tasks": {
                                    **worker.state.tasks,
                                    tasks[2].task_id: tasks[2],
                                }
                            }
                        )
                        await anyio.sleep(0.25)
                        assert tasks[2].task_id not in supervisor.in_progress
                        assert set(supervisor.pending) == {tasks[1].task_id}
                        worker.state = worker.state.model_copy(
                            update={
                                "tasks": {
                                    task.task_id: task.model_copy(
                                        update={"task_status": TaskStatus.Cancelled}
                                    )
                                    for task in tasks
                                }
                            }
                        )
                        dispatched: TaskId | None = None
                        with anyio.move_on_after(1) as cancellation_deadline:
                            dispatched = await cancel_receiver.receive_async()
                        assert not cancellation_deadline.cancel_called, (
                            "Cancellation dispatcher is blocked behind queued generation acknowledgement"
                        )
                        assert dispatched in {task.task_id for task in tasks}
                        assert not compute_released.is_set()
                    finally:
                        task_group.start_soon(
                            supervisor.start_task,
                            Shutdown(
                                instance_id=bound.instance.instance_id,
                                runner_id=runner_id,
                            ),
                        )
                        compute_released.set()
                        try:
                            with anyio.CancelScope(shield=True):
                                await to_thread.run_sync(runner_thread.join, 5)
                                assert not runner_thread.is_alive()
                        finally:
                            task_group.cancel_scope.cancel()
        finally:
            compute_released.set()
            task_sender.close()
            runner_thread.join(timeout=5)
            assert not runner_thread.is_alive()


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


@pytest.mark.anyio
@pytest.mark.skipif(
    not hasattr(signal, "SIGTERM"), reason="requires native child termination"
)
@pytest.mark.parametrize(
    ("drain_enabled", "peer_failed", "allocation_unreadable", "removed_instance"),
    [
        (False, False, False, False),
        pytest.param(
            True,
            False,
            False,
            False,
            marks=pytest.mark.skipif(
                sys.platform != "darwin", reason="native IOKit boundary"
            ),
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            marks=pytest.mark.skipif(
                sys.platform != "darwin", reason="native IOKit boundary"
            ),
        ),
        pytest.param(
            True,
            False,
            True,
            False,
            marks=pytest.mark.skipif(
                sys.platform != "darwin", reason="native IOKit boundary"
            ),
        ),
        pytest.param(
            True,
            False,
            False,
            True,
            marks=pytest.mark.skipif(
                sys.platform != "darwin", reason="native IOKit boundary"
            ),
        ),
    ],
)
async def test_workers_wait_for_all_old_processes_before_replacement(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    drain_enabled: bool,
    peer_failed: bool,
    allocation_unreadable: bool,
    removed_instance: bool,
) -> None:
    logger_setup(None)
    request.addfinalizer(gc.collect)
    request.addfinalizer(logger_cleanup)
    monkeypatch.setenv("EXO_GPU_DRAIN_ENABLED", "1" if drain_enabled else "0")
    allocation_permitted = anyio.Event()
    allocation_rejected = anyio.Event()
    rejection_checked = anyio.Event()
    native_library = ctypes.CDLL

    def allocation_value(number: int, value_type: int, output: int) -> bool:
        if allocation_unreadable:
            allocation_rejected.set()
            return False
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int64))[0] = (
            0 if allocation_permitted.is_set() else 64 * 1024**3
        )
        return True

    allocation_function = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    )(allocation_value)

    def load_native_library(name: str, *, use_errno: bool = False) -> ctypes.CDLL:
        library = native_library(name, use_errno=use_errno)
        if name.endswith("CoreFoundation.framework/CoreFoundation"):
            library.__dict__["CFNumberGetValue"] = allocation_function
        return library

    if drain_enabled:
        monkeypatch.setattr(ctypes, "CDLL", load_native_library)
    node_id = NodeId("global-order-node")
    runner_id = RunnerId("global-order-runner")
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("global-order-instance"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=runner_id,
        node_id=node_id,
    )
    peer_node_id, peer_runner_id = next(
        (assigned_node, assigned_runner)
        for assigned_node, assigned_runner in (
            bound_instance.instance.shard_assignments.node_to_runner.items()
        )
        if assigned_runner != runner_id
    )
    peer_instance = BoundInstance(
        instance=bound_instance.instance,
        bound_node_id=peer_node_id,
        bound_runner_id=peer_runner_id,
    )
    ready_sender, ready_receiver = mp_channel[None]()
    release_sender, release_receiver = mp_channel[None]()
    native_process = mp.Process

    def create_native_process(
        *,
        target: Callable[..., object],
        name: str | None,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        daemon: bool | None,
    ) -> BaseProcess:
        if (
            not peer_failed
            and len(args) > 3
            and isinstance(args[3], BoundInstance)
            and args[3].bound_runner_id == peer_runner_id
        ):
            return native_process(
                target=_process_with_exit_gate,
                name=name,
                args=(target, args, kwargs, ready_sender, release_receiver),
                daemon=daemon,
            )
        return native_process(
            target=target, name=name, args=args, kwargs=kwargs, daemon=daemon
        )

    monkeypatch.setattr(mp, "Process", create_native_process)
    workers: list[Worker] = []
    indexed_senders: list[Sender[IndexedEvent]] = []
    old_supervisors: list[RunnerSupervisor] = []
    peer_exit_ready = anyio.Event()
    failed_worker_ready = anyio.Event()
    peer_shutting_down = False
    failed_worker_started = False
    replacement_planned = False
    failed_shutdown_task_id: TaskId | None = None
    shutdown_statuses: dict[TaskId, TaskStatus] = {}
    terminal_runners: set[RunnerId] = set()
    event_index = 0

    async def observe_peer_exit_ready() -> None:
        await ready_receiver.receive_async()
        peer_exit_ready.set()

    async def release_peer_exit() -> None:
        if not peer_failed:
            await peer_exit_ready.wait()
        await failed_worker_ready.wait()
        await anyio.sleep(0.2)
        if not peer_failed:
            release_sender.send(None)
        if drain_enabled:
            with anyio.fail_after(20):
                while any(old.runner_process.is_alive() for old in old_supervisors):
                    await anyio.sleep(0.01)
            await anyio.sleep(0.5)
            assert not replacement_planned, (
                "Replacement admitted while post-stop allocation remained high"
            )
            if allocation_unreadable:
                await allocation_rejected.wait()
                assert all(
                    isinstance(worker.state.runners.get(assigned), RunnerShuttingDown)
                    for worker in workers
                    for assigned in (runner_id, peer_runner_id)
                )
                rejection_checked.set()
            else:
                allocation_permitted.set()

    with ExitStack() as resources:
        event_sender, event_receiver = channel[Event](max_buffer_size=0)
        command_sender, command_receiver = channel[ForwarderCommand]()
        download_sender, download_receiver = channel[ForwarderDownloadCommand]()
        for resource in (
            event_sender,
            event_receiver,
            command_sender,
            command_receiver,
            download_sender,
            download_receiver,
            ready_sender,
            ready_receiver,
            release_sender,
            release_receiver,
        ):
            resources.enter_context(resource)
        for assigned in (bound_instance, peer_instance):
            indexed_sender, indexed_receiver = channel[IndexedEvent]()
            resources.enter_context(indexed_sender)
            resources.enter_context(indexed_receiver)
            indexed_senders.append(indexed_sender)
            worker = Worker(
                assigned.bound_node_id,
                event_receiver=indexed_receiver,
                event_sender=event_sender.clone(),
                command_sender=command_sender.clone(),
                download_command_sender=download_sender.clone(),
                api_port=0,
            )
            supervisor = await RunnerSupervisor.create(
                bound_instance=assigned, event_sender=event_sender.clone()
            )
            worker.runners[assigned.bound_runner_id] = supervisor
            workers.append(worker)
            old_supervisors.append(supervisor)

        with anyio.fail_after(180 if drain_enabled else 60):
            async with anyio.create_task_group() as task_group:
                try:
                    for supervisor in old_supervisors:
                        task_group.start_soon(supervisor.run)
                    initial_statuses: dict[RunnerId, RunnerStatus] = {}
                    while len(initial_statuses) < 2:
                        event = await event_receiver.receive()
                        assert isinstance(event, RunnerStatusUpdated)
                        initial_statuses[event.runner_id] = event.runner_status
                    for worker in workers:
                        worker.state = worker.state.model_copy(
                            update={
                                "instances": {
                                    bound_instance.instance.instance_id: (
                                        bound_instance.instance
                                    ),
                                },
                                "runners": initial_statuses,
                            }
                        )
                    if not removed_instance:
                        os.kill(old_supervisors[0].runner_process.pid, signal.SIGTERM)
                    if peer_failed:
                        os.kill(old_supervisors[1].runner_process.pid, signal.SIGTERM)
                    failures_observed: set[RunnerId] = set()
                    while not removed_instance:
                        event = await event_receiver.receive()
                        if isinstance(event, RunnerStatusUpdated) and isinstance(
                            event.runner_status, RunnerFailed
                        ):
                            for worker in workers:
                                worker.state = worker.state.model_copy(
                                    update={
                                        "runners": {
                                            **worker.state.runners,
                                            event.runner_id: event.runner_status,
                                        }
                                    }
                                )
                            failures_observed.add(event.runner_id)
                            if len(failures_observed) == (2 if peer_failed else 1):
                                break
                    if not removed_instance:
                        assert not old_supervisors[0].runner_process.is_alive()
                    if not peer_failed:
                        task_group.start_soon(observe_peer_exit_ready)
                    task_group.start_soon(release_peer_exit)
                    task_group.start_soon(workers[1].run)
                    if drain_enabled:
                        task_group.start_soon(workers[0].run)
                        failed_worker_started = True
                        failed_worker_ready.set()
                    if removed_instance:
                        for worker, indexed_sender in zip(
                            workers, indexed_senders, strict=True
                        ):
                            await indexed_sender.send(
                                IndexedEvent(
                                    idx=event_index,
                                    event=InstanceDeleted(
                                        instance_id=bound_instance.instance.instance_id
                                    ),
                                )
                            )
                            with anyio.fail_after(2):
                                while worker.state.last_event_applied_idx < event_index:
                                    await anyio.sleep(0)
                        event_index += 1
                    while True:
                        event: Event | None = None
                        with anyio.move_on_after(0.05):
                            event = await event_receiver.receive()
                        if event is None:
                            if rejection_checked.is_set():
                                assert not replacement_planned
                                assert failed_shutdown_task_id is not None
                                assert (
                                    shutdown_statuses.get(failed_shutdown_task_id)
                                    == TaskStatus.Complete
                                )
                                break
                            continue
                        if (
                            isinstance(event, TaskCreated)
                            and isinstance(event.task, Shutdown)
                            and event.task.runner_id == runner_id
                        ):
                            failed_shutdown_task_id = event.task.task_id
                        if isinstance(event, TaskStatusUpdated):
                            shutdown_statuses[event.task_id] = event.task_status
                        if isinstance(event, RunnerStatusUpdated):
                            if isinstance(event.runner_status, RunnerShutdown):
                                terminal_runners.add(event.runner_id)
                            if event.runner_id == peer_runner_id:
                                if isinstance(event.runner_status, RunnerShuttingDown):
                                    peer_shutting_down = True
                                if isinstance(event.runner_status, RunnerShutdown):
                                    assert not old_supervisors[
                                        1
                                    ].runner_process.is_alive()
                            if (
                                event.runner_id == runner_id
                                and isinstance(event.runner_status, RunnerIdle)
                                and replacement_planned
                            ):
                                assert all(
                                    not old.runner_process.is_alive()
                                    for old in old_supervisors
                                )
                                assert workers[0].runners[
                                    runner_id
                                ].runner_process.pid != (
                                    old_supervisors[0].runner_process.pid
                                )
                                assert failed_shutdown_task_id is not None
                                assert (
                                    shutdown_statuses.get(failed_shutdown_task_id)
                                    == TaskStatus.Complete
                                ), (
                                    "Failed-runner Shutdown remained nonterminal before replacement"
                                )
                                break
                        if isinstance(event, TaskCreated) and isinstance(
                            event.task, CreateRunner
                        ):
                            assert not drain_enabled or allocation_permitted.is_set(), (
                                "Replacement admitted before post-stop allocation release"
                            )
                            assert all(
                                not old.runner_process.is_alive()
                                for old in old_supervisors
                            ), "Replacement admitted before the old peer process exited"
                            replacement_planned = True
                        for worker, indexed_sender in zip(
                            workers, indexed_senders, strict=True
                        ):
                            if worker is workers[1] or failed_worker_started:
                                await indexed_sender.send(
                                    IndexedEvent(idx=event_index, event=event)
                                )
                                with anyio.fail_after(2):
                                    while (
                                        worker.state.last_event_applied_idx
                                        < event_index
                                    ):
                                        await anyio.sleep(0)
                            else:
                                worker.state = workers[1].state
                        event_index += 1
                        if removed_instance and terminal_runners == {
                            runner_id,
                            peer_runner_id,
                        }:
                            assert not replacement_planned
                            assert all(
                                not old.runner_process.is_alive()
                                for old in old_supervisors
                            )
                            assert failed_shutdown_task_id is not None
                            assert (
                                shutdown_statuses.get(failed_shutdown_task_id)
                                == TaskStatus.Complete
                            )
                            break
                        if peer_shutting_down and not failed_worker_started:
                            if peer_failed:
                                assert not old_supervisors[1].runner_process.is_alive()
                            else:
                                await peer_exit_ready.wait()
                                assert old_supervisors[1].runner_process.is_alive()
                            workers[0].state = workers[1].state
                            task_group.start_soon(workers[0].run)
                            failed_worker_started = True
                            failed_worker_ready.set()
                finally:
                    with contextlib.suppress(Exception):
                        release_sender.send(None)
                    for supervisor in old_supervisors:
                        supervisor.shutdown()
                    task_group.cancel_scope.cancel()
