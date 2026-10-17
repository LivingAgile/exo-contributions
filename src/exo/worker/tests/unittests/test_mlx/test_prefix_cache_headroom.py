# type: ignore
"""A long prefill must not start on top of a prefix cache that leaves it no GPU memory.

Entries used to be evicted only when a new one was added, after its prefill. A long prompt
arriving while the cache was full could take Metal past its working set, and Metal aborts the
process (kIOGPUCommandBufferCallbackErrorOutOfMemory). These tests drive the runners' sequence,
`get_kv_cache` followed by `prefill`, with a real small model and real KV caches; only MLX's
memory readings are faked.
"""

import json
import multiprocessing as mp
import os
import tempfile
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models import llama
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import exo.worker.engines.mlx.cache as cache_module
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import prefill

VOCAB = 256
WEIGHTS = 10 * 1024**3
PROMPT_TOKENS = 64


def _model() -> llama.Model:
    mx.random.seed(0)
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
        )
    )
    mx.eval(model.parameters())
    return model


def _tokenizer() -> TokenizerWrapper:
    vocab = {f"w{i}": i for i in range(VOCAB)}
    vocab["[UNK]"] = VOCAB
    native = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    return TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=native, unk_token="[UNK]"),
        eos_token_ids={1},
    )


def _prompt(first: int, length: int = PROMPT_TOKENS) -> mx.array:
    # Token ids 2..VOCAB-1, starting at `first`.
    return mx.array([(first - 2 + 7 * i) % (VOCAB - 2) + 2 for i in range(length)])


def _cached_bytes(prefix_cache: KVPrefixCache) -> int:
    return sum(int(layer.nbytes) for entry in prefix_cache.caches for layer in entry)


def _serve(model, tokenizer, prefix_cache: KVPrefixCache, tokens: mx.array) -> dict:
    """One request as the runners serve it; records the cache when prefill begins."""
    cache, remaining, matched, _ = prefix_cache.get_kv_cache(model, tokens)
    at_prefill = {
        "first_tokens": [int(p[0].item()) for p in prefix_cache.prompts],
        "matched": None if matched is None else int(prefix_cache.prompts[matched][0].item()),
        "in_use": mx.get_active_memory(),
    }
    _, _, snapshots = prefill(model, tokenizer, make_sampler(0.0), remaining, cache,
                              group=None, on_prefill_progress=None,
                              distributed_prompt_progress_callback=None)
    if matched is None:
        prefix_cache.add_kv_cache(tokens, cache, snapshots)
    else:
        prefix_cache.update_kv_cache(matched, tokens, cache, snapshots, len(tokens) - len(remaining))
    return at_prefill


@pytest.fixture(autouse=True)
def no_reserve_and_no_psutil_pressure(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The reserve is configuration; zero keeps the arithmetic in KV bytes. psutil pressure is the
    # add-time policy, not what these tests are about.
    monkeypatch.setattr(cache_module, "_PREFILL_RESERVE_BYTES", 0, raising=False)
    monkeypatch.setattr(cache_module, "_GPU_BUDGET_BYTES", -1, raising=False)
    with patch.object(cache_module, "get_memory_used_percentage", return_value=0.1):
        yield


def _history(model, tokenizer, prefix_cache: KVPrefixCache) -> int:
    """C first (so it is least recently used), then A and B. Returns one entry's bytes."""
    for first in (30, 10, 20):
        _serve(model, tokenizer, prefix_cache, _prompt(first))
    entry = _cached_bytes(prefix_cache) // 3
    assert entry > 0
    return entry


def _faked_memory(prefix_cache: KVPrefixCache, working_set: int):
    # GPU memory in use is the weights plus whatever the prefix cache still holds.
    return (
        patch.object(mx, "get_active_memory", lambda: WEIGHTS + _cached_bytes(prefix_cache)),
        patch.object(mx, "get_cache_memory", lambda: 0),
        patch.object(mx, "device_info", lambda: {"max_recommended_working_set_size": working_set}),
    )


def test_older_entries_are_evicted_before_a_prefill_that_would_not_fit() -> None:
    model, tokenizer = _model(), _tokenizer()
    prefix_cache = KVPrefixCache(None)
    entry = _history(model, tokenizer, prefix_cache)
    # A request extending C needs about two entries' worth for its 128 tokens. With A, B and C
    # held that is five entries; the budget has room for three and a half.
    working_set = WEIGHTS + int(3.5 * entry)
    active, cached, info = _faked_memory(prefix_cache, working_set)
    with active, cached, info:
        at_prefill = _serve(model, tokenizer, prefix_cache,
                            mx.concatenate([_prompt(30), _prompt(99)]))

    assert at_prefill["first_tokens"] == [30], at_prefill
    assert at_prefill["matched"] == 30
    assert at_prefill["in_use"] + 2 * entry <= working_set


def test_nothing_is_evicted_when_memory_is_ample() -> None:
    model, tokenizer = _model(), _tokenizer()
    prefix_cache = KVPrefixCache(None)
    entry = _history(model, tokenizer, prefix_cache)
    active, cached, info = _faked_memory(prefix_cache, WEIGHTS + 1000 * entry)
    with active, cached, info:
        at_prefill = _serve(model, tokenizer, prefix_cache,
                            mx.concatenate([_prompt(30), _prompt(99)]))

    assert sorted(at_prefill["first_tokens"]) == [10, 20, 30]
    assert at_prefill["matched"] == 30


def test_a_new_prompt_evicts_from_the_least_recently_used_end() -> None:
    model, tokenizer = _model(), _tokenizer()
    prefix_cache = KVPrefixCache(None)
    entry = _history(model, tokenizer, prefix_cache)
    # A 64-token prompt needs one entry's worth; room for three leaves one more to go.
    active, cached, info = _faked_memory(prefix_cache, WEIGHTS + int(3.5 * entry))
    with active, cached, info:
        at_prefill = _serve(model, tokenizer, prefix_cache, _prompt(77))

    assert at_prefill["matched"] is None
    assert at_prefill["first_tokens"] == [10, 20], at_prefill


# -- Rank agreement: two real ranks on a localhost ring, each with its own GPU reading. --


def _rank(rank: int, hostfile: str, results: Any) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        cache_module._PREFILL_RESERVE_BYTES = 0
        cache_module._GPU_BUDGET_BYTES = -1
        cache_module.get_memory_used_percentage = lambda: 0.1
        model, tokenizer = _model(), _tokenizer()
        prefix_cache = KVPrefixCache(None)
        entry = _history(model, tokenizer, prefix_cache)
        prefix_cache._group = group
        # Rank 0 has ample memory; rank 1 has room for three and a half entries.
        slack = 1000 * entry if rank == 0 else int(3.5 * entry)
        gathers = []
        real_gather = mx.distributed.all_gather

        def counting_gather(x, group=None):
            gathers.append(len(prefix_cache.caches))
            return real_gather(x, group=group)

        mx.get_active_memory = lambda: WEIGHTS + _cached_bytes(prefix_cache)
        mx.get_cache_memory = lambda: 0
        mx.device_info = lambda: {"max_recommended_working_set_size": WEIGHTS + slack}
        mx.distributed.all_gather = counting_gather
        tokens = mx.concatenate([_prompt(30), _prompt(99)])
        _, _, matched, _ = prefix_cache.get_kv_cache(model, tokens)
        results.put((rank, {
            "first_tokens": [int(p[0].item()) for p in prefix_cache.prompts],
            "matched": None if matched is None else int(prefix_cache.prompts[matched][0].item()),
            "gathers": gathers,
        }))
    except Exception as error:
        results.put((rank, f"{type(error).__name__}: {error}"))


def test_ranks_evict_the_same_entries_when_only_one_lacks_memory() -> None:
    context = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([f"127.0.0.1:{29630 + i}" for i in range(2)], f)
        hostfile = f.name
    try:
        results = context.Queue()
        ranks = [context.Process(target=_rank, args=(r, hostfile, results)) for r in range(2)]
        for process in ranks:
            process.start()
        for process in ranks:
            process.join(timeout=120)
        seen = dict(results.get(timeout=5) for _ in range(2))
    finally:
        os.unlink(hostfile)

    assert all(isinstance(seen[r], dict) for r in range(2)), seen
    assert seen[0]["first_tokens"] == seen[1]["first_tokens"] == [30], seen
    assert seen[0]["matched"] == seen[1]["matched"] == 30
    # Checked before each eviction and once more after the last: 3, 2, then 1 entry held.
    assert seen[0]["gathers"] == seen[1]["gathers"] == [3, 2, 1], seen
