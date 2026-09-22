import copy
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.models.cache import CacheList as VlmCacheList
from mlx_vlm.models.cache import KVCache as VlmKVCache
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
    KVPrefixCache,
    has_non_kv_caches,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.glm5_next import Model, ModelArgs
from exo.worker.engines.mlx.types import Model as ExoModel
from exo.worker.engines.mlx.utils_mlx import load_model_for_exo
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator
from exo.worker.tests.unittests.test_mlx.test_tp_bit_exact import MODEL_CONFIGS


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
    list(
        mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=" ".join(["one"] * 600),
            kv_prefix_cache=None,
            group=None,
            on_prefill_progress=lambda processed, total: progress.append(processed),
        )
    )
    assert 512 in progress
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
