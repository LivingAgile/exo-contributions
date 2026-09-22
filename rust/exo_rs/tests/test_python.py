import asyncio
import os

import pytest
from _pytest.capture import CaptureFixture
from exo_rs import (
    NetworkingHandle,
    Pidfile,
    FromSwarm,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items(
    unused_tcp_port: int, unused_udp_port: int
) -> None:
    print("PYTHON: starting handle")
    identity = os.urandom(16).hex().lstrip("0") or "1"
    handle = NetworkingHandle.new(
        identity, f"exo-binding-test-{identity}", unused_tcp_port, unused_udp_port
    )
    print("PYTHON: handle started")

    receiver = asyncio.create_task(_await_recv(handle))
    try:
        async with asyncio.timeout(30):
            for tick in range(10):
                await asyncio.sleep(1)
                await handle.gossipsub_publish("topic", f"tick-{tick}".encode())
                assert not receiver.done()
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    a = Pidfile("/tmp/lock.pid", 0o0600)


if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
