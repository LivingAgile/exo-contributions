# type: ignore
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import CacheList, KVCache

from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

_FULL_INDEXER_LAYERS = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
)
_EXACT_RUNTIME_SHA256 = (
    "3e5a6353795b8159ed092df2b7c32ee1028a25284521d6416869c7a0027fd4ec"
)


class FixtureIndexer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.index_topk = 2
        self.wq_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        query_inputs: mx.array,
        cache: KVCache | None,
    ) -> mx.array | None:
        batch, sequence, _ = inputs.shape
        queries = (
            self.wq_b(query_inputs)
            .reshape(batch, sequence, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = self.k_norm(self.wk(inputs)).reshape(batch, 1, sequence, self.head_dim)
        if cache is not None:
            keys, _ = cache.update_and_fetch(
                keys,
                mx.zeros((batch, 1, sequence, 0)),
            )
        if keys.shape[2] <= self.index_topk:
            return None
        scores = queries.astype(mx.float32) @ keys.astype(mx.float32).swapaxes(-1, -2)
        weights = self.weights_proj(inputs).astype(mx.float32)
        scores = (scores * weights.transpose(0, 2, 1)[:, :, :, None]).sum(
            axis=1,
            keepdims=True,
        )
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class FixtureHeadBank(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.weight = mx.random.normal((num_heads, head_dim, head_dim))

    def __call__(self, inputs: mx.array) -> mx.array:
        return mx.einsum("bhld,hde->bhle", inputs, self.weight)


class FixtureAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        full_indexer: bool,
    ):
        super().__init__()
        self.q_lora_rank = hidden_size
        self.head_dim = hidden_size // num_heads
        self.q_a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_b_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.embed_q = FixtureHeadBank(num_heads, self.head_dim)
        self.unembed_out = FixtureHeadBank(num_heads, self.head_dim)
        self.indexer = FixtureIndexer(hidden_size, num_heads) if full_indexer else None

    def __call__(
        self,
        inputs: mx.array,
        cache: CacheList | None,
        previous_topk: mx.array | None,
    ) -> tuple[mx.array, mx.array | None]:
        batch, sequence, _ = inputs.shape
        query_inputs = self.q_a_proj(inputs)
        queries = (
            self.q_b_proj(query_inputs)
            .reshape(batch, sequence, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = values = queries
        if cache is not None:
            keys, values = cache[0].update_and_fetch(keys, values)
        if self.indexer is not None:
            indexer_cache = cache[1] if cache is not None else None
            topk = self.indexer(inputs, query_inputs, indexer_cache)
        else:
            topk = previous_topk

        scores = (queries * (self.head_dim**-0.5)) @ keys.swapaxes(-1, -2)
        if topk is not None:
            sparse_mask = mx.zeros(scores.shape, dtype=mx.bool_)
            sparse_mask = mx.put_along_axis(
                sparse_mask,
                mx.broadcast_to(topk, scores.shape[:-1] + (topk.shape[-1],)),
                mx.array(True),
                axis=-1,
            )
            scores = mx.where(sparse_mask, scores, -float("inf"))
        probabilities = mx.softmax(scores, axis=-1)
        output = probabilities @ values
        output = self.unembed_out(self.embed_q(output))
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)
        return self.o_proj(output), topk


class FixtureDenseMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(inputs), self.up_proj(inputs)))


class FixtureGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = mx.zeros((4, hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((4,), dtype=mx.float32)


class FixtureMoe(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.shared_experts = FixtureDenseMlp(hidden_size, intermediate_size)
        self.switch_mlp = FixtureDenseMlp(hidden_size, intermediate_size)
        self.gate = FixtureGate(hidden_size)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs = self.shared_experts(inputs) + self.switch_mlp(inputs)
        if self.sharding_group is not None:
            outputs = mx.distributed.all_sum(outputs, group=self.sharding_group)
        return outputs


class FixtureDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        *,
        full_indexer: bool,
        dense: bool,
    ):
        super().__init__()
        self.self_attn = FixtureAttention(
            hidden_size,
            num_heads,
            full_indexer=full_indexer,
        )
        self.mlp = (
            FixtureDenseMlp(hidden_size, intermediate_size)
            if dense
            else FixtureMoe(hidden_size, intermediate_size)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: CacheList | None,
        previous_topk: mx.array | None,
    ) -> tuple[mx.array, mx.array | None]:
        attention, topk = self.self_attn(inputs, cache, previous_topk)
        hidden = inputs + attention
        return hidden + self.mlp(hidden), topk


class FixtureInnerModel(nn.Module):
    def __init__(self, *, invalid_shared_indexer: bool = False):
        super().__init__()
        indexer_types = [
            "full" if index in _FULL_INDEXER_LAYERS else "shared" for index in range(78)
        ]
        self.embed_tokens = nn.Embedding(128, 128)
        self.norm = nn.RMSNorm(128)
        self.layers = [
            FixtureDecoderLayer(
                128,
                128,
                4,
                full_indexer=(kind == "full" or invalid_shared_indexer and index == 3),
                dense=index < 3,
            )
            for index, kind in enumerate(indexer_types)
        ]

    def __call__(
        self,
        inputs: mx.array,
        cache: list[CacheList] | None,
    ) -> mx.array:
        layer_caches = cache if cache is not None else [None] * len(self.layers)
        topk = None
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            inputs, topk = layer(inputs, layer_cache, topk)
        return inputs


class FixtureGlmMoeDsaModel(nn.Module):
    def __init__(self, *, invalid_shared_indexer: bool = False, mixed: bool = False):
        super().__init__()
        self.model_type = "glm_moe_dsa"
        indexer_types = [
            "full" if index in _FULL_INDEXER_LAYERS else "shared" for index in range(78)
        ]
        self.args = type("Args", (), {"indexer_types": indexer_types})()
        self.model = FixtureInnerModel(invalid_shared_indexer=invalid_shared_indexer)
        self.lm_head = nn.Linear(128, 128, bias=False)
        if mixed:
            self._quantize_mixed()
        else:
            nn.quantize(
                self,
                group_size=32,
                bits=8,
                class_predicate=lambda path, module: isinstance(module, nn.Linear)
                and ".indexer." not in path,
            )

    @property
    def layers(self) -> list[FixtureDecoderLayer]:
        return self.model.layers

    def _quantize_mixed(self) -> None:
        nn.quantize(
            self,
            group_size=32,
            bits=8,
            class_predicate=lambda path, module: isinstance(
                module,
                (nn.Embedding, nn.Linear),
            )
            and path in ("model.embed_tokens", "lm_head"),
        )
        for layer in self.layers:
            nn.quantize(
                layer.self_attn,
                group_size=32,
                bits=8,
                class_predicate=lambda path, module: isinstance(module, nn.Linear)
                and ".indexer." not in path,
            )
            if isinstance(layer.mlp, FixtureDenseMlp):
                nn.quantize(layer.mlp, group_size=32, bits=8)
            else:
                nn.quantize(layer.mlp.shared_experts, group_size=32, bits=8)
                nn.quantize(layer.mlp.switch_mlp, group_size=32, bits=4)

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        return self.lm_head(self.model(inputs, cache))

    def make_cache(self) -> list[CacheList]:
        return [
            CacheList(KVCache(), KVCache())
            if layer.self_attn.indexer is not None
            else CacheList(KVCache())
            for layer in self.layers
        ]


FixtureGlmMoeDsaModel.__module__ = "custom_model"


def _consume(loader):
    progress = []
    while True:
        try:
            progress.append(next(loader))
        except StopIteration as completed:
            return completed.value, progress


def _exact_runtime_model() -> nn.Module | None:
    source_value = os.environ.get("GLM_MOE_DSA_MODEL_FILE")
    if source_value is None:
        return None
    source_path = Path(source_value)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == _EXACT_RUNTIME_SHA256
    spec = importlib.util.spec_from_file_location("custom_model", source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    indexer_types = [
        "full" if index in _FULL_INDEXER_LAYERS else "shared"
        for index in range(78)
    ]
    args = module.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=128,
        hidden_size=128,
        index_head_dim=32,
        index_n_heads=4,
        index_topk=2,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=78,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=indexer_types,
        num_nextn_predict_layers=0,
    )
    return module.Model(args)


def _mixed_eight_bit_module(path: str, module: nn.Module) -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    if path in ("model.embed_tokens", "lm_head"):
        return True
    if ".self_attn." in path and ".indexer." not in path:
        return True
    if ".mlp.shared_experts." in path:
        return True
    return any(path.startswith(f"model.layers.{index}.mlp.") for index in range(3))


def _build(*, seed: int, mixed: bool) -> nn.Module:
    mx.random.seed(seed)
    model = _exact_runtime_model()
    if model is None:
        model = FixtureGlmMoeDsaModel(mixed=mixed)
    elif mixed:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and model.quant_predicate(path, module)
            and not _mixed_eight_bit_module(path, module),
        )
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            class_predicate=_mixed_eight_bit_module,
        )
    else:
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and model.quant_predicate(path, module),
        )
    mx.eval(model.parameters())
    return model


def _parity_inputs() -> tuple[mx.array, mx.array]:
    if os.environ.get("GLM_MOE_DSA_MODEL_FILE") is not None:
        return (
            mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32),
            mx.array([[9], [10]], dtype=mx.int32),
        )
    return (
        mx.arange(1024, dtype=mx.float32).reshape(2, 4, 128) / 1024,
        mx.arange(256, dtype=mx.float32).reshape(2, 1, 128) / 256,
    )


def _run_rank(
    rank: int,
    world_size: int,
    hostfile: str,
    output_path: str,
    result_queue: Any,
    *,
    seed: int,
    mixed: bool,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model, progress = _consume(
            tensor_auto_parallel(_build(seed=seed, mixed=mixed), group)
        )
        prefill, continuation = _parity_inputs()
        caches = model.make_cache()
        prefill_output = model(prefill, cache=caches)
        mx.eval(prefill_output)
        outputs = model(continuation, cache=caches)
        mx.eval(outputs)
        path = output_path if rank == 0 else f"{output_path}.rank{rank}"
        np.savez(path, outputs=np.asarray(outputs.astype(mx.float32)))
        result_queue.put((rank, len(progress), None))
    except BaseException as error:
        result_queue.put((rank, 0, f"{error}\n{traceback.format_exc()}"))


def _run_reference(output_path: str, *, seed: int, mixed: bool) -> None:
    model = _build(seed=seed, mixed=mixed)
    prefill, continuation = _parity_inputs()
    caches = model.make_cache()
    prefill_output = model(prefill, cache=caches)
    mx.eval(prefill_output)
    outputs = model(continuation, cache=caches)
    mx.eval(outputs)
    np.savez(output_path, outputs=np.asarray(outputs.astype(mx.float32)))


def _assert_tensor_parity(*, world_size: int, mixed: bool, port: int) -> None:
    context = mp.get_context("spawn")
    result_queue: Any = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        reference_path = str(Path(temp_dir) / "reference.npz")
        output_path = str(Path(temp_dir) / "distributed.npz")
        _run_reference(reference_path, seed=19, mixed=mixed)
        hosts = [f"127.0.0.1:{port + rank}" for rank in range(world_size)]
        hostfile = str(Path(temp_dir) / "hosts.json")
        Path(hostfile).write_text(json.dumps(hosts))
        processes = [
            context.Process(
                target=_run_rank,
                args=(rank, world_size, hostfile, output_path, result_queue),
                kwargs={"seed": 19, "mixed": mixed},
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=300) for _ in processes]
        for process in processes:
            process.join(60)
        errors = {rank: error for rank, _, error in results if error is not None}
        assert not errors, errors
        assert {count for _, count, _ in results} == {78}
        reference = np.load(reference_path)["outputs"]
        distributed = np.load(output_path)["outputs"]
        np.testing.assert_allclose(distributed, reference, atol=1 / 128, rtol=0)


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX distributed requires Metal",
)


def test_glm_moe_dsa_rejects_shared_layer_with_indexer() -> None:
    group = mx.distributed.init()
    model = FixtureGlmMoeDsaModel(invalid_shared_indexer=True)

    with pytest.raises(ValueError, match="shared layer 3 must not define an indexer"):
        _consume(tensor_auto_parallel(model, group))


def test_glm_moe_dsa_fixture_preserves_indexshare_cache_geometry() -> None:
    model = FixtureGlmMoeDsaModel()
    full_layers = [
        index
        for index, layer in enumerate(model.layers)
        if layer.self_attn.indexer is not None
    ]
    caches = model.make_cache()

    assert full_layers == list(_FULL_INDEXER_LAYERS)
    assert sum(len(cache.caches) == 2 for cache in caches) == 21
    assert sum(len(cache.caches) == 1 for cache in caches) == 57


def test_glm_moe_dsa_mixed_fixture_matches_production_quantization_shape() -> None:
    model = FixtureGlmMoeDsaModel(mixed=True)
    dense = model.layers[0]
    moe = model.layers[3]

    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.lm_head, nn.QuantizedLinear)
    assert model.lm_head.bits == 8
    assert isinstance(dense.self_attn.q_b_proj, nn.QuantizedLinear)
    assert dense.self_attn.q_b_proj.bits == 8
    assert isinstance(dense.mlp.gate_proj, nn.QuantizedLinear)
    assert dense.mlp.gate_proj.bits == 8
    assert isinstance(moe.mlp.shared_experts.gate_proj, nn.QuantizedLinear)
    assert moe.mlp.shared_experts.gate_proj.bits == 8
    assert isinstance(moe.mlp.switch_mlp.gate_proj, nn.QuantizedLinear)
    assert moe.mlp.switch_mlp.gate_proj.bits == 4
    assert isinstance(moe.self_attn.indexer, type(None))


@pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_MODEL_FILE") is None,
    reason="exact bundled GLM runtime path was not supplied",
)
def test_exact_glm_moe_dsa_runtime_is_sha_pinned() -> None:
    assert _exact_runtime_model() is not None


@pytest.mark.parametrize(
    ("world_size", "mixed", "port"),
    [(2, False, 32120), (4, False, 32124), (2, True, 32128), (4, True, 32132)],
)
def test_glm_moe_dsa_tensor_parity(
    world_size: int,
    mixed: bool,
    port: int,
) -> None:
    _assert_tensor_parity(world_size=world_size, mixed=mixed, port=port)
