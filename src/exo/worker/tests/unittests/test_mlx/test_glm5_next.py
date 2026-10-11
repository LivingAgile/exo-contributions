import copy
import gc
import json
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.models.cache import ArraysCache as VlmArraysCache
from mlx_vlm.models.cache import CacheList as VlmCacheList
from mlx_vlm.models.cache import KVCache as VlmKVCache
from mlx_vlm.models.glm5_next import language as native_language
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.utils.channels import MpReceiver, MpSender, MpState
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    copy_snapshot_entry,
    has_non_kv_caches,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.generator import generate as generation
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.glm5_next import Model, ModelArgs
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.engines.mlx.types import Model as ExoModel
from exo.worker.engines.mlx.utils_mlx import load_model_for_exo
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator
from exo.worker.tests.unittests.test_mlx.test_tp_bit_exact import MODEL_CONFIGS


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_flash_sparse_prefill_fallback_bounds_tile_memory(dtype: mx.Dtype):
    if mx.default_device() != mx.gpu:
        pytest.skip("Peak allocation requires the native GPU device")
    mx.random.seed(42)
    queries = mx.random.normal((1, 8, 256, 256)).astype(dtype)
    keys = mx.random.normal((1, 8, 4096, 256)).astype(dtype)
    values = mx.random.normal((1, 8, 4096, 256)).astype(dtype)
    indices = mx.broadcast_to(mx.arange(512, dtype=mx.int32)[None, None], (1, 256, 512))
    mx.eval(queries, keys, values, indices)
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    sparse_prefill_attention = cast(
        Callable[[mx.array, mx.array, mx.array, mx.array, float], mx.array],
        vars(native_language)["_sparse_prefill_attention"],
    )
    actual = sparse_prefill_attention(queries, keys, values, indices, 0.0625)
    mx.eval(actual)
    mx.synchronize()
    extra_peak = mx.get_peak_memory() - baseline
    reference_tiles: list[mx.array] = []
    for query_index in range(queries.shape[2]):
        reference = sparse_prefill_attention(
            queries[:, :, query_index : query_index + 1],
            keys,
            values,
            indices[:, query_index : query_index + 1],
            0.0625,
        )
        mx.eval(reference)
        reference_tiles.append(reference)
    expected = mx.concatenate(reference_tiles, axis=2)
    mx.eval(expected)
    assert actual.dtype == dtype
    assert actual.shape == (1, 8, 256, 256)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    gathered_tile_bytes = 2 * 16 * 8 * 512 * 256 * 4
    assert extra_peak < 2 * gathered_tile_bytes


@pytest.mark.parametrize("cache_family", ["arrays", "vlm_arrays", "vlm_list"])
@pytest.mark.parametrize("entrypoint", ["copy", "snapshot", "restore"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_snapshot_detaches_backing_and_preserves_metadata(
    cache_family: str, entrypoint: str, dtype: mx.Dtype
):
    if mx.default_device() != mx.gpu:
        pytest.skip("GPU allocation ownership requires the native GPU device")
    warmup = mx.zeros((32, 32))
    mx.eval(warmup)
    del warmup
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    backing = mx.full((8192, 512), 7, dtype=dtype)
    mx.eval(backing)
    original_entry: ArraysCache | VlmArraysCache | VlmCacheList
    if cache_family == "vlm_list":
        keys_cache = VlmKVCache()
        keys_cache.keys = backing[None, None, :1, :]
        keys_cache.values = mx.zeros((1, 1, 1, 0), dtype=dtype)
        keys_cache.offset = 1
        original_entry = VlmCacheList(keys_cache)
        del keys_cache
    else:
        arrays_cache = (
            ArraysCache(size=2) if cache_family == "arrays" else VlmArraysCache(size=2)
        )
        arrays_cache.state = [backing[:1, :], None]
        arrays_cache.left_padding = mx.array([3])
        arrays_cache.lengths = mx.array([9])
        original_entry = arrays_cache
        del arrays_cache
    original_cache: KVCacheType = [original_entry]
    if entrypoint == "copy":
        copied_entry = copy_snapshot_entry(original_entry)
    elif entrypoint == "snapshot":
        captured = snapshot_ssm_states(original_cache)
        copied_entry = captured.states[0]
        del captured
    else:
        captured = CacheSnapshot([original_entry], token_count=1)
        trim_cache(original_cache, 0, captured)
        copied_entry = original_cache[0]
        del captured
    assert copied_entry is not None
    assert type(copied_entry) is type(original_entry)
    if isinstance(copied_entry, VlmCacheList):
        copied_keys = cast(VlmKVCache, list(copied_entry)[0])
        copied_array = copied_keys.keys
        assert copied_keys.offset == 1
        assert copied_array is not None
        assert copied_array.shape == (1, 1, 1, 512)
        del copied_keys
    else:
        copied_arrays = cast(ArraysCache | VlmArraysCache, copied_entry)
        assert copied_arrays.left_padding is not None
        assert copied_arrays.lengths is not None
        assert copied_arrays.left_padding.tolist() == [3]
        assert copied_arrays.lengths.tolist() == [9]
        assert copied_arrays.state[1] is None
        copied_array = copied_arrays.state[0]
        assert isinstance(copied_array, mx.array)
        assert copied_array.shape == (1, 512)
        del copied_arrays
    mx.eval(copied_array)
    assert copied_array.dtype == dtype
    assert mx.all(mx.equal(copied_array, 7)).item()
    del original_entry, original_cache, backing
    gc.collect()
    mx.clear_cache()
    assert mx.get_active_memory() - baseline < 1024 * 1024


@pytest.mark.parametrize("quantized", [False, True])
def test_flash_strict_loading_and_prefix_restore(tmp_path: Path, quantized: bool):
    configuration = copy.deepcopy(
        cast(dict[str, Any], MODEL_CONFIGS["glm5_next"]["args"])
    )
    configuration["model_type"] = "glm5_next"
    configuration["model_file"] = "untrusted.py"
    mx.random.seed(42)
    reference = Model(ModelArgs.from_dict(configuration))
    if quantized:
        configuration["quantization"] = {
            "bits": 4,
            "group_size": 64,
            "model.layers.1.mlp.shared_experts.gate_proj": {
                "bits": 6,
                "group_size": 64,
            },
            "model.layers.1.mlp.shared_experts.up_proj": {
                "bits": 6,
                "group_size": 64,
            },
            "model.layers.1.mlp.shared_experts.down_proj": {
                "bits": 5,
                "group_size": 64,
            },
        }
        nn.quantize(
            reference,
            bits=4,
            group_size=64,
            class_predicate=lambda path, module: (
                {
                    "bits": (
                        6
                        if path.endswith("shared_experts.gate_up_proj")
                        else 5
                        if path.endswith("shared_experts.down_proj")
                        else 4
                    ),
                    "group_size": 64,
                }
                if hasattr(module, "to_quantized") and ".mlp." in path
                else False
            ),
        )
    weights = dict(
        cast(list[tuple[str, mx.array]], tree_flatten(reference.parameters()))
    )
    for name in list(weights):
        if ".shared_experts.gate_up_proj." in name:
            gate, up = mx.split(weights.pop(name), 2, axis=0)
            weights[name.replace("gate_up_proj", "gate_proj")] = gate
            weights[name.replace("gate_up_proj", "up_proj")] = up
    weights["model.visual.excluded"] = mx.zeros((1,))
    (tmp_path / "config.json").write_text(json.dumps(configuration))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded, _ = load_model_for_exo(tmp_path)
    exo_model = cast(ExoModel, loaded)
    assert isinstance(loaded, Model)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] * 8])
    expected = reference(tokens, cache=reference.make_cache())
    cache = loaded.make_cache()
    mx.eval(loaded(tokens[:, :32], cache=cache))
    for entry in cache:
        if isinstance(entry, VlmCacheList):
            assert cast(VlmKVCache, list(entry)[-1]).size() == 0
    assert has_non_kv_caches(cache)
    snapshot = snapshot_ssm_states(cache)
    assert all(entry is not None for entry in snapshot.states)
    mx.eval(loaded(tokens[:, 32:], cache=cache))
    trim_cache(cache, 32, snapshot)
    resumed = loaded(tokens[:, 32:], cache=cache)
    mx.eval(expected, resumed)
    assert mx.allclose(expected[:, 32:], resumed, atol=1e-4, rtol=1e-4).item()
    prefix = KVPrefixCache(None)
    prefix.add_kv_cache(tokens[0], cache, [snapshot])
    reused, remaining, matched, _ = prefix.get_kv_cache(exo_model, tokens[0])
    assert matched == 0
    actual = loaded(remaining[None], cache=reused)
    mx.eval(actual)
    assert mx.allclose(expected[:, 32:], actual, atol=1e-4, rtol=1e-4).item()
    weights.pop("language_model.lm_head.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="lm_head.weight"):
        load_model_for_exo(tmp_path)


def test_flash_production_generation_prefix_reuse(monkeypatch: pytest.MonkeyPatch):
    configuration = cast(dict[str, Any], MODEL_CONFIGS["glm5_next"]["args"])
    mx.random.seed(42)
    module: nn.Module = Model(ModelArgs.from_dict(configuration))
    model = cast(ExoModel, cast(object, module))
    native_tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4}, unk_token="[UNK]"
        )
    )
    native_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=native_tokenizer, unk_token="[UNK]")
    )
    task = TextGenerationTaskParams(
        model=ModelId("test"), input=[], max_output_tokens=20, temperature=0.0
    )
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    with (
        MpSender(MpState[Event](0)) as event_sender,
        MpReceiver(MpState[TaskId](0)) as cancel_receiver,
    ):
        builder = MlxBuilder(
            model_id=ModelId("test"),
            event_sender=event_sender,
            cancel_receiver=cancel_receiver,
            inference_model=model,
            tokenizer=tokenizer,
        )
        assert isinstance(builder.build(), SequentialGenerator)
    progress: list[int] = []
    snapshot_references: list[weakref.ReferenceType[CacheSnapshot]] = []
    original_snapshot = generation.snapshot_ssm_states

    def observe_snapshot(cache: KVCacheType) -> CacheSnapshot:
        snapshot = original_snapshot(cache)
        snapshot_references.append(weakref.ref(snapshot))
        return snapshot

    def observe_progress(processed: int, total: int) -> None:
        progress.append(processed)
        live_snapshots = sum(
            reference() is not None for reference in snapshot_references
        )
        assert 1 <= live_snapshots <= 2

    monkeypatch.setattr(generation, "snapshot_ssm_states", observe_snapshot)
    list(
        mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=" ".join(["one"] * 600),
            kv_prefix_cache=None,
            group=None,
            on_prefill_progress=observe_progress,
        )
    )
    assert 512 in progress
    assert len(snapshot_references) > 2
    assert all(reference() is None for reference in snapshot_references)
    prefix = KVPrefixCache(None)
    for prompt in ("one two three", "one two three", "one two three four"):
        results = [
            [
                response.token
                for response in mlx_generate(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    prompt=prompt,
                    kv_prefix_cache=cache,
                    group=None,
                )
            ]
            for cache in (None, prefix)
        ]
        assert results[0]
        assert results[0] == results[1]
    long_prefix = KVPrefixCache(None)
    for prompt in (
        " ".join(["one"] * 1008),
        " ".join(["one"] * 1008 + ["two"] * 32),
        " ".join(["one"] * 1008 + ["two"] * 32 + ["three"] * 32),
        " ".join(["one"] * 1008 + ["two"] * 32 + ["four"] * 32),
        " ".join(["one"] * 1008 + ["four"] * 32),
    ):
        results = [
            [
                response.token
                for response in mlx_generate(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    prompt=prompt,
                    kv_prefix_cache=cache,
                    group=None,
                )
            ]
            for cache in (None, long_prefix)
        ]
        assert results[0]
        assert results[0] == results[1]
        retained_groups = cast(
            list[list[CacheSnapshot] | None], vars(long_prefix)["_snapshots"]
        )
        assert retained_groups
        for retained in retained_groups:
            assert retained is not None
            assert len(retained) <= 2
            for checkpoint in retained:
                for entry in checkpoint.states:
                    if isinstance(entry, VlmCacheList):
                        assert cast(VlmKVCache, list(entry)[-1]).size() == 0
