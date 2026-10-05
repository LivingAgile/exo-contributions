import multiprocessing as mp
import time

import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    create_task_group,
    fail_after,
    sleep,
)
from loguru import logger

from exo.shared.logging import logger_cleanup, logger_setup
from exo.utils.channels import ErrorOverride, MpReceiver, MpSender, channel, mp_channel


class CustomClosedResourceError(ClosedResourceError):
    pass


class CustomBrokenResourceError(BrokenResourceError):
    pass


class CustomEndOfStream(EndOfStream):
    pass


class CustomWouldBlock(WouldBlock):
    pass


ERROR_OVERRIDE = ErrorOverride(
    closed_resource_error=CustomClosedResourceError,
    broken_resource_error=CustomBrokenResourceError,
    end_of_stream=CustomEndOfStream,
    would_block=CustomWouldBlock,
)


def foo(recv: MpReceiver[str]):
    expected = ["hi", "hi 2", "bye"]
    with recv as r:
        for item in r:
            assert item == expected.pop(0)


def bar(send: MpSender[str]):
    logger.warning("hi")
    send.send("hi")
    time.sleep(0.1)
    logger.warning("hi 2")
    send.send("hi 2")
    time.sleep(0.1)
    logger.warning("bye")
    send.send("bye")
    time.sleep(0.1)
    send.close()
    send.join()


@pytest.mark.anyio
async def test_channel_ipc():
    with fail_after(0.5):
        s, r = mp_channel[str]()
        p1 = mp.Process(target=foo, args=(r,))
        p2 = mp.Process(target=bar, args=(s,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()


def test_channel_error_override_replaces_sync_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)

    with pytest.raises(CustomWouldBlock) as would_block_info:
        send.send_nowait(1)
    assert type(would_block_info.value.__cause__) is WouldBlock

    recv.close()
    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        send.send_nowait(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send.close()
    with pytest.raises(CustomClosedResourceError) as closed_resource_info:
        send.send_nowait(1)
    assert type(closed_resource_info.value.__cause__) is ClosedResourceError


@pytest.mark.anyio
async def test_channel_error_override_replaces_async_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    recv.close()

    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        await send.send(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send, recv = channel[int](error_override_config=ERROR_OVERRIDE)
    send.close()
    with pytest.raises(CustomEndOfStream) as end_of_stream_info:
        await recv.receive()
    assert type(end_of_stream_info.value.__cause__) is EndOfStream


@pytest.mark.anyio
async def test_channel_error_override_is_preserved_by_clones():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    send_clone = send.clone()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await send_clone.send(1)

    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    cloned_send = recv.clone_sender()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await cloned_send.send(1)


@pytest.mark.parametrize("read_mode", ["receive", "collect"])
def test_mp_channel_sender_close_drains_fifo(
    read_mode: str, request: pytest.FixtureRequest
):
    logger_setup(None)
    request.addfinalizer(logger_cleanup)
    sender, receiver = mp_channel[str]()
    producer = mp.Process(target=bar, args=(sender,))
    expected = ["hi", "hi 2", "bye"]
    try:
        producer.start()
        producer.join(5)
        assert producer.exitcode == 0
        if read_mode == "collect":
            observed = receiver.collect()
        else:
            observed = [receiver.receive() for _ in expected]
            with pytest.raises(EndOfStream):
                receiver.receive()
        assert observed == expected
    finally:
        if producer.is_alive():
            producer.terminate()
        if producer.pid is not None:
            producer.join(2)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("read_mode", ["receive", "collect"])
def test_mp_receiver_close_rejects_queued_values(read_mode: str):
    sender, receiver = mp_channel[int]()
    try:
        sender.send(73)
        receiver.close()
        with pytest.raises(ClosedResourceError):
            if read_mode == "collect":
                receiver.collect()
            else:
                receiver.receive()
    finally:
        sender.close()
        receiver.close()


@pytest.mark.anyio
async def test_mp_receiver_close_ends_blocked_read():
    sender, receiver = mp_channel[int]()
    try:
        with fail_after(2):
            with pytest.raises(BaseExceptionGroup) as read_closed:
                async with create_task_group() as task_group:
                    task_group.start_soon(receiver.receive_async)
                    await sleep(0.05)
                    receiver.close()
            assert all(
                isinstance(error, (ClosedResourceError, EndOfStream))
                for error in read_closed.value.exceptions
            )
    finally:
        sender.close()
        receiver.close()
