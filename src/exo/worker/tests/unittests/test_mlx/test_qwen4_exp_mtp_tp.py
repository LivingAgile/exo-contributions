# type: ignore
"""Two-rank Tensor-parallel check for Qwen4-exp MTP decode.

Every rank must take the same accept, replay and fallback decisions, or the
next sharded MoE all_sum deadlocks or the ranks diverge. Each rank runs the
same sequence of decodes from the same seed; the parent compares the token
streams across ranks and checks greedy MTP against greedy plain decode.
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="MLX distributed requires Metal"
)

WORLD_SIZE = 2
PORT_BASE = 32010
MAX_TOKENS = 24


def _rank_worker(rank: int, hostfile: str, out_path: str, queue: Any) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
            Qwen4ExpMTPDraftModel,
        )

        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel
        from exo.worker.tests.unittests.test_mlx import test_qwen4_exp_mtp as base

        group = mx.distributed.init(backend="ring", strict=True)
        model = base._target(5)
        loader = tensor_auto_parallel(model, group)
        while True:
            try:
                next(loader)
            except StopIteration as completed:
                model = completed.value
                break
        mx.eval(model.parameters())

        results: dict[str, list[int]] = {}
        reference, _ = base._plain(model, MAX_TOKENS)
        results["plain-greedy"] = reference

        drafter = base.OracleDrafter(base._drafter_config(), reference, [1, -1, 0, 2])
        state = base._attach(model, drafter)
        state.configured_block_size = 4
        results["mtp-greedy-oracle"], _ = base._mtp(model, state, MAX_TOKENS)

        mx.random.seed(3)
        untrained = Qwen4ExpMTPDraftModel(base._drafter_config())
        mx.eval(untrained.parameters())
        untrained.bind(model)
        state.drafter = untrained
        state.disabled = False
        for temperature in (0.0, 0.7):
            state.reset_request()
            mx.random.seed(42)
            sampler = make_sampler(temp=temperature)
            tokens, _ = base._mtp(model, state, MAX_TOKENS, sampler=sampler)
            results[f"mtp-untrained-t{temperature}"] = tokens
            results[f"accepts-untrained-t{temperature}"] = list(state.accept_lens)

        with open(f"{out_path}.r{rank}", "w", encoding="utf-8") as handle:
            json.dump(results, handle)
        queue.put((rank, True, None))
    except BaseException as error:  # noqa: BLE001 - report child-process failures
        queue.put((rank, False, f"{error}\n{traceback.format_exc()}"))


def test_two_rank_mtp_decisions_agree_and_greedy_matches_plain() -> None:
    directory = tempfile.mkdtemp()
    out_path = f"{directory}/mtp"
    hosts = [f"127.0.0.1:{PORT_BASE + i}" for i in range(WORLD_SIZE)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile = f.name
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_rank_worker, args=(rank, hostfile, out_path, queue))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [queue.get(timeout=300) for _ in range(WORLD_SIZE)]
    finally:
        for process in processes:
            process.join(60)
            if process.is_alive():
                process.kill()
    for rank, ok, payload in outcomes:
        if not ok:
            pytest.fail(f"rank {rank} failed: {str(payload)[:1500]}")

    ranks = []
    for rank in range(WORLD_SIZE):
        with open(f"{out_path}.r{rank}", encoding="utf-8") as handle:
            ranks.append(json.load(handle))
    for other in ranks[1:]:
        assert other == ranks[0]
    first = ranks[0]
    assert len(first["plain-greedy"]) == MAX_TOKENS
    assert first["mtp-greedy-oracle"] == first["plain-greedy"]
    assert first["mtp-untrained-t0.0"] == first["plain-greedy"]
    assert len(first["mtp-untrained-t0.7"]) == MAX_TOKENS
