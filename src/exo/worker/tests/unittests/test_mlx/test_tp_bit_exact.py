# type: ignore
"""uv run pytest -v -m "" src/exo/worker/tests/unittests/test_mlx/test_tp_bit_exact.py"""

import importlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback

import numpy as np
import pytest

MODEL_CONFIGS = {
    "qwen4_exp": {
        "module": "mlx_lm.models.qwen4_exp",
        "args": {
            "model_type": "qwen4_exp",
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 512,
                "rms_norm_eps": 1e-6,
                "full_attention_interval": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "hc_count": 4,
                "hc_lowrank": 16,
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 16,
                "indexer_budget": 8,
                "indexer_compress_ratio": 4,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "split_ngram_parts": 4,
                "ple_embed_dim": 64,
                "ple_layer_ids": [1],
                "eos_token_id": 1,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    },
    "qwen4_exp_q8_uneven": {
        "module": "mlx_lm.models.qwen4_exp",
        "quantize": dict(group_size=64, bits=8, mode="affine"),
        "quantize_moe_only": True,
        "args": {
            "model_type": "qwen4_exp",
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 512,
                "rms_norm_eps": 1e-6,
                "full_attention_interval": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "hc_count": 4,
                "hc_lowrank": 16,
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 16,
                "indexer_budget": 8,
                "indexer_compress_ratio": 4,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "split_ngram_parts": 4,
                "ple_embed_dim": 64,
                "ple_layer_ids": [1],
                "eos_token_id": 1,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    },
    "llama": dict(
        module="mlx_lm.models.llama",
        args=dict(
            model_type="llama",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=512,
            max_position_embeddings=128,
            head_dim=32,
            rope_theta=10000.0,
        ),
    ),
    "muse_glimmer": {
        "module": "mlx_lm.models.muse_glimmer",
        "args": {
            "model_type": "muse_glimmer",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 512,
            "sliding_window": 8,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "rope_theta": 100.0,
        },
    },
    "qwen3_5_moe": dict(
        module="mlx_lm.models.qwen3_5_moe",
        args=dict(
            model_type="qwen3_5_moe",
            text_config=dict(
                model_type="qwen3_5_moe",
                vocab_size=512,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=32,
                max_position_embeddings=128,
                rms_norm_eps=1e-6,
                tie_word_embeddings=False,
                attention_bias=False,
                full_attention_interval=2,
                linear_num_value_heads=32,
                linear_num_key_heads=16,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                num_experts=16,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                shared_expert_intermediate_size=256,
                moe_intermediate_size=256,
                norm_topk_prob=True,
                rope_parameters={
                    "type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                },
            ),
        ),
    ),
    "qwen3_next": dict(
        module="mlx_lm.models.qwen3_next",
        args=dict(
            model_type="qwen3_next",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=512,
            attention_bias=False,
            full_attention_interval=2,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            num_experts=16,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=256,
            moe_intermediate_size=256,
            norm_topk_prob=True,
            mlp_only_layers=[],
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
        ),
    ),
    "deepseek_v3": dict(
        module="mlx_lm.models.deepseek_v3",
        args=dict(
            model_type="deepseek_v3",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            rope_theta=10000.0,
            rope_scaling={},
            attention_bias=False,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
        ),
    ),
    "deepseek_v3_q4": dict(
        module="mlx_lm.models.deepseek_v3",
        quantize=dict(group_size=32, bits=4, mode="affine"),
        args=dict(
            model_type="deepseek_v3",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            q_lora_rank=None,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=32,
            rope_theta=10000.0,
            rope_scaling={},
            attention_bias=False,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
        ),
    ),
    "glm4_moe_lite": dict(
        module="mlx_lm.models.glm4_moe_lite",
        args=dict(
            model_type="glm4_moe_lite",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=512,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            rope_theta=10000.0,
            attention_bias=False,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=32,
        ),
    ),
    "minimax": dict(
        module="mlx_lm.models.minimax",
        args=dict(
            model_type="minimax",
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=128,
            num_experts_per_tok=2,
            num_local_experts=8,
            shared_intermediate_size=256,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rotary_dim=32,
            vocab_size=512,
        ),
    ),
    "gpt_oss": dict(
        module="mlx_lm.models.gpt_oss",
        args=dict(
            model_type="gpt_oss",
            hidden_size=512,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=512,
            head_dim=32,
            rms_norm_eps=1e-6,
            num_local_experts=8,
            num_experts_per_tok=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=64,
            rope_theta=10000.0,
        ),
    ),
    "deepseek_v4": dict(
        module="mlx_lm.models.deepseek_v4",
        args=dict(
            model_type="deepseek_v4",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=1,
            head_dim=16,
            qk_rope_head_dim=8,
            sliding_window=32,
            compress_ratios=[0, 0, 4, 0, 0],
            index_n_heads=4,
            index_head_dim=16,
            index_topk=16,
            moe_intermediate_size=32,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=1,
            hc_mult=1,
            num_nextn_predict_layers=0,
            max_position_embeddings=2048,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 2,
                "original_max_position_embeddings": 1024,
                "type": "yarn",
            },
        ),
    ),
    "gemma4": dict(
        module="mlx_lm.models.gemma4",
        args=dict(
            model_type="gemma4",
            vocab_size=512,
            text_config=dict(
                vocab_size=512,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=32,
                global_head_dim=32,
                num_kv_shared_layers=0,
                vocab_size_per_layer_input=512,
                hidden_size_per_layer_input=512,
                rms_norm_eps=1e-6,
                max_position_embeddings=128,
                sliding_window=64,
                sliding_window_pattern=2,
                layer_types=[
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                enable_moe_block=True,
                num_experts=8,
                top_k_experts=2,
                moe_intermediate_size=256,
            ),
        ),
    ),
}

_PROMPT = [[1, 23, 45, 67, 89, 12, 34, 56]]
_QWEN4_EXP_MAX_ABS_DIFF = 1 / 256
_QWEN4_EXP_MAX_MEAN_DIFF = 0.001


def _build(name, seed=0):
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_map_with_path
    from mlx_lm.models.switch_layers import SwitchLinear

    import exo.worker.engines.mlx.auto_parallel  # noqa: F401

    cfg = MODEL_CONFIGS[name]
    module = importlib.import_module(cfg["module"])
    model_cls = module.Model
    model_args_cls = module.ModelArgs

    mx.random.seed(seed)
    args = model_args_cls(**cfg["args"])
    m = model_cls(args)

    def _to_bf16(_p, v):
        if hasattr(v, "dtype") and v.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return v.astype(mx.bfloat16)
        return v

    m.update(tree_map_with_path(_to_bf16, m.parameters()))
    if "quantize" in cfg:
        class_predicate = None
        if cfg.get("quantize_moe_only"):

            def class_predicate(path, module):
                return isinstance(module, (nn.Linear, SwitchLinear)) and (
                    ".mlp.switch_mlp" in path or ".mlp.shared_expert." in path
                )

        nn.quantize(m, **cfg["quantize"], class_predicate=class_predicate)
    mx.eval(m.parameters())
    return mx, m


def _run(name, out_path, shard, seed=0):
    import mlx.core as mx

    if shard:
        g = mx.distributed.init(backend="ring", strict=True)
    mx_, m = _build(name, seed)
    muse_kv_weights = None
    muse_attention_weights = None
    if name == "muse_glimmer":
        muse_kv_weights = [
            (
                np.asarray(layer.self_attn.k_proj.weight.astype(mx.float32)),
                np.asarray(layer.self_attn.v_proj.weight.astype(mx.float32)),
            )
            for layer in m.layers
        ]
        muse_attention_weights = [
            (
                np.asarray(layer.self_attn.q_proj.weight.astype(mx.float32)),
                np.asarray(layer.self_attn.gate_proj.weight.astype(mx.float32)),
            )
            for layer in m.layers
        ]
    if shard:
        from exo.worker.engines.mlx import auto_parallel

        loader = auto_parallel.tensor_auto_parallel(m, g)
        while True:
            try:
                next(loader)
            except StopIteration as completed:
                m = completed.value
                break
        mx_.eval(m.parameters())
        if muse_kv_weights is not None:
            for layer, (key_weight, value_weight) in zip(
                m.layers, muse_kv_weights, strict=True
            ):
                np.testing.assert_array_equal(
                    np.asarray(layer.self_attn.k_proj.weight.astype(mx.float32)),
                    key_weight,
                )
                np.testing.assert_array_equal(
                    np.asarray(layer.self_attn.v_proj.weight.astype(mx.float32)),
                    value_weight,
                )
        if muse_attention_weights is not None:
            n_kv_heads = 2
            heads_per_kv = 4
            head_dim = 16
            heads_per_rank = heads_per_kv // g.size()
            start = g.rank() * heads_per_rank
            end = start + heads_per_rank
            for layer, (query_weight, gate_weight) in zip(
                m.layers, muse_attention_weights, strict=True
            ):
                for projection, source in (
                    (layer.self_attn.q_proj, query_weight),
                    (layer.self_attn.gate_proj, gate_weight),
                ):
                    expected = source.reshape(
                        n_kv_heads, heads_per_kv, head_dim, -1
                    )[:, start:end].reshape(projection.weight.shape)
                    np.testing.assert_array_equal(
                        np.asarray(projection.weight.astype(mx.float32)), expected
                    )
    if name.startswith("qwen4_exp"):
        rows = [
            mx_.array([[1, 23, 45, 67, 89, 12, 34]], dtype=mx_.int32),
            mx_.array([[1, 23, 45, 67, 89]], dtype=mx_.int32),
        ]
        caches = [m.make_cache() for _ in rows]
        for row, cache in zip(rows, caches, strict=True):
            m(row, cache=cache)
        batch_cache = [
            items[0].merge(list(items)) for items in zip(*caches, strict=True)
        ]
        logits = m(mx_.array([[56], [56]], dtype=mx_.int32), cache=batch_cache)
    elif name == "muse_glimmer":
        inputs = mx_.array(_PROMPT, dtype=mx_.int32)
        cache = m.make_cache()
        prefill = m(inputs[:, :-1], cache=cache)
        decode = m(inputs[:, -1:], cache=cache)
        logits = mx_.concatenate([prefill, decode], axis=1)
    else:
        inputs = mx_.array(_PROMPT, dtype=mx_.int32)
        logits = m(inputs)
    mx_.eval(logits)
    np.savez(out_path, logits=np.asarray(logits.astype(mx_.float32)))


def _ref_worker(name, out_path, q, seed):
    try:
        _run(name, out_path, shard=False, seed=seed)
        q.put(True)
    except BaseException as e:
        q.put(f"{e}\n{traceback.format_exc()}")


def _tp_worker(name, rank, hf, out_path, q, seed):
    os.environ["MLX_HOSTFILE"] = hf
    os.environ["MLX_RANK"] = str(rank)
    try:
        path = out_path if rank == 0 else out_path + f".r{rank}"
        _run(name, path, shard=True, seed=seed)
        q.put((rank, True, None))
    except BaseException as e:
        q.put((rank, False, f"{e}\n{traceback.format_exc()}"))


def _tp_muse_invalid_gqa_worker(rank, hf, q):
    os.environ["MLX_HOSTFILE"] = hf
    os.environ["MLX_RANK"] = str(rank)
    try:
        import mlx.core as mx

        from exo.worker.engines.mlx.auto_parallel import tensor_auto_parallel

        group = mx.distributed.init(backend="ring", strict=True)
        _, model = _build("muse_glimmer")
        for layer in model.layers:
            layer.self_attn.n_heads = 6
        next(tensor_auto_parallel(model, group))
        q.put((rank, True, None))
    except Exception as error:  # noqa: BLE001 - report child-process failures
        q.put((rank, False, f"{error}\n{traceback.format_exc()}"))


def _run_compare(name, world_size, port_base, *, seed=0, atol=0.0, rtol=0.0):
    d = tempfile.mkdtemp()
    ref_path = f"{d}/ref.npz"
    tp_path = f"{d}/tp.npz"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    p = ctx.Process(target=_ref_worker, args=(name, ref_path, q, seed))
    p.start()
    p.join(300)
    r = q.get(timeout=10)
    if r is not True:
        pytest.fail(f"[{name}] ref FAIL: {str(r)[:500]}")

    hosts = [f"127.0.0.1:{port_base + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hf = f.name
    ps = [
        ctx.Process(target=_tp_worker, args=(name, rank, hf, tp_path, q, seed))
        for rank in range(world_size)
    ]
    for pp in ps:
        pp.start()
    results = [q.get(timeout=300) for _ in range(world_size)]
    for pp in ps:
        pp.join(60)
    for rank, ok, payload in results:
        if not ok:
            pytest.fail(f"[{name}] rank {rank} FAIL: {payload[:500]}")

    ref = np.load(ref_path)["logits"]
    tp = np.load(tp_path)["logits"]
    diff = np.abs(ref - tp)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    p99_diff = float(np.percentile(diff, 99))
    max_ref = float(np.abs(ref).max())
    if atol or rtol:
        np.testing.assert_allclose(tp, ref, atol=atol, rtol=rtol)
    else:
        assert max_diff == 0.0, (
            f"[{name} TP={world_size}] not bit-exact: max={max_diff} "
            f"p99={p99_diff} mean={mean_diff} max_ref={max_ref}"
        )
    return max_diff, p99_diff, mean_diff, max_ref


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin", reason="MLX distributed requires Metal"
    ),
]


@pytest.mark.parametrize(("seed", "port_base"), [(0, 31980), (17, 31984), (43, 31988)])
def test_qwen4_exp_tp_numerical_parity(seed, port_base):
    max_diff, p99_diff, mean_diff, _ = _run_compare(
        "qwen4_exp",
        4,
        port_base,
        seed=seed,
        atol=_QWEN4_EXP_MAX_ABS_DIFF,
    )
    assert max_diff <= _QWEN4_EXP_MAX_ABS_DIFF
    assert p99_diff <= _QWEN4_EXP_MAX_ABS_DIFF
    assert mean_diff <= _QWEN4_EXP_MAX_MEAN_DIFF


def test_qwen4_exp_q8_uneven_groups_tp_numerical_parity():
    max_diff, p99_diff, mean_diff, _ = _run_compare(
        "qwen4_exp_q8_uneven",
        4,
        31992,
        atol=_QWEN4_EXP_MAX_ABS_DIFF,
    )
    assert max_diff <= _QWEN4_EXP_MAX_ABS_DIFF
    assert p99_diff <= _QWEN4_EXP_MAX_ABS_DIFF
    assert mean_diff <= _QWEN4_EXP_MAX_MEAN_DIFF


@pytest.mark.parametrize(("world_size", "port_base"), [(2, 31994), (4, 31996)])
def test_muse_glimmer_tp_prefill_and_cached_decode_parity(world_size, port_base):
    _run_compare("muse_glimmer", world_size, port_base, atol=0.03)


def test_muse_glimmer_rejects_incompatible_gqa_geometry():
    world_size = 4
    context = mp.get_context("spawn")
    queue = context.Queue()
    hosts = [f"127.0.0.1:{32000 + rank}" for rank in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(hosts, file)
        hostfile = file.name
    processes = [
        context.Process(
            target=_tp_muse_invalid_gqa_worker,
            args=(rank, hostfile, queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=300) for _ in range(world_size)]
    for process in processes:
        process.join(60)

    assert all(not succeeded for _, succeeded, _ in results)
    assert all(
        "query heads per kv group (3) must be divisible by world size (4)"
        in (payload or "").lower()
        for _, _, payload in results
    )


@pytest.mark.skip("TP=2 is currently very different to TP=1. This test will not pass")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("name", list(MODEL_CONFIGS))
def test_tp_bit_exact(name, world_size):
    name_idx = list(MODEL_CONFIGS).index(name)
    port = 32000 + name_idx * 20 + world_size
    _run_compare(name, world_size, port)
