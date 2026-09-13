# type: ignore
import copy
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models import qwen4_exp

from exo.worker.engines.mlx import utils_mlx
from exo.worker.tests.unittests.test_mlx.test_tp_bit_exact import MODEL_CONFIGS


def _qwen_config(*, quantized_ngram: bool = False) -> dict:
    config = copy.deepcopy(MODEL_CONFIGS["qwen4_exp"]["args"])
    if quantized_ngram:
        config["text_config"]["ple_embed_dim"] = 128
        config["quantization"] = {"group_size": 32, "bits": 8}
    return config


def _convert_floating_parameters_to_bfloat16(model: qwen4_exp.Model) -> None:
    model.update(
        tree_map(
            lambda value: (
                value.astype(mx.bfloat16)
                if value.dtype in (mx.float16, mx.float32, mx.bfloat16)
                else value
            ),
            model.parameters(),
        )
    )


def _write_converted_checkpoint(
    model_path: Path,
    *,
    omit_weight: bool = False,
    quantized_ngram: bool = False,
) -> qwen4_exp.Model:
    config = _qwen_config(quantized_ngram=quantized_ngram)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(**config))
    _convert_floating_parameters_to_bfloat16(model)
    if quantized_ngram:
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            class_predicate=lambda path, _module: "ngram_embedding" in path,
        )
    weights = dict(tree_flatten(model.parameters()))
    if omit_weight:
        weights.pop("lm_head.weight")

    config["model_file"] = "bundled_model.py"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(model_path / "model.safetensors"), weights)
    return model


def _write_official_checkpoint(model_path: Path) -> qwen4_exp.Model:
    config = _qwen_config()
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(**config))
    _convert_floating_parameters_to_bfloat16(model)
    converted = dict(tree_flatten(model.parameters()))
    official = {}

    for layer_index in range(config["text_config"]["num_hidden_layers"]):
        converted_prefix = f"model.layers.{layer_index}.mlp.switch_mlp."
        official_prefix = (
            f"model.language_model.layers.{layer_index}.mlp.experts."
        )
        official[official_prefix + "gate_up_proj"] = mx.concatenate(
            [
                converted.pop(converted_prefix + "gate_proj.weight"),
                converted.pop(converted_prefix + "up_proj.weight"),
            ],
            axis=-2,
        )
        official[official_prefix + "down_proj"] = converted.pop(
            converted_prefix + "down_proj.weight"
        )

    fold_one = (
        "q_layernorm.weight",
        "k_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "hc_norm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    )
    for name, weight in converted.items():
        official_name = (
            "model.language_model." + name[len("model.") :]
            if name.startswith("model.")
            else name
        )
        if name.endswith("conv1d.weight") and weight.ndim == 3:
            weight = weight.transpose(0, 2, 1)
        if name.endswith(fold_one):
            weight = weight - 1.0
        official[official_name] = weight

    official["model.visual.placeholder"] = mx.zeros((1,))
    official["model.mtp.placeholder"] = mx.zeros((1,))
    config["model_file"] = "bundled_model.py"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(model_path / "model.safetensors"), official)
    return model


def _assert_parameters_equal(
    expected: qwen4_exp.Model, actual: qwen4_exp.Model
) -> None:
    expected_parameters = dict(tree_flatten(expected.parameters()))
    actual_parameters = dict(tree_flatten(actual.parameters()))
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_value in expected_parameters.items():
        actual_value = actual_parameters[name]
        if expected_value.dtype == mx.bfloat16:
            expected_value = expected_value.astype(mx.float32)
            actual_value = actual_value.astype(mx.float32)
        np.testing.assert_array_equal(
            np.asarray(actual_value),
            np.asarray(expected_value),
            err_msg=name,
        )


def test_qwen4_exp_native_loader_ignores_repository_code_and_is_strict(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete"
    _write_converted_checkpoint(complete)

    model, config = utils_mlx.load_model_for_exo(complete)

    assert isinstance(model, qwen4_exp.Model)
    assert config["model_file"] is None

    incomplete = tmp_path / "incomplete"
    _write_converted_checkpoint(incomplete, omit_weight=True)
    with pytest.raises(ValueError, match="lm_head.weight"):
        utils_mlx.load_model_for_exo(incomplete)


def test_qwen4_exp_loader_accepts_official_and_quantized_converted_layouts(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    official_reference = _write_official_checkpoint(official)
    official_model, _ = utils_mlx.load_model_for_exo(official)

    quantized = tmp_path / "quantized"
    quantized_reference = _write_converted_checkpoint(
        quantized, quantized_ngram=True
    )
    quantized_model, _ = utils_mlx.load_model_for_exo(quantized)
    ngram_shard = (
        quantized_model.model.layers[0]
        .ple.ple_embedding.ngram_embedding.shard_0
    )
    reference_ngram = (
        quantized_reference.model.layers[0]
        .ple.ple_embedding
    )
    loaded_ngram = quantized_model.model.layers[0].ple.ple_embedding
    previous_context = mx.array([[1, 9, 17]], dtype=mx.int32)
    input_ids = mx.array([[23, 31]], dtype=mx.int32)
    expected_embedding = reference_ngram(input_ids, previous_context)
    actual_embedding = loaded_ngram(input_ids, previous_context)
    mx.eval(
        official_reference.parameters(),
        official_model.parameters(),
        expected_embedding,
        actual_embedding,
    )

    assert isinstance(official_model, qwen4_exp.Model)
    _assert_parameters_equal(official_reference, official_model)
    assert isinstance(ngram_shard, nn.QuantizedEmbedding)
    assert ngram_shard.group_size == 32
    np.testing.assert_array_equal(
        np.asarray(actual_embedding.astype(mx.float32)),
        np.asarray(expected_embedding.astype(mx.float32)),
    )


def test_other_architectures_keep_non_strict_loading(tmp_path: Path) -> None:
    model_path = tmp_path / "llama"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
                "rms_norm_eps": 1e-6,
            }
        )
    )

    model, _ = utils_mlx.load_model_for_exo(model_path)

    assert model.model_type == "llama"