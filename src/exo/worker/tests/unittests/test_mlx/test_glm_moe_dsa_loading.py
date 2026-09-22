# type: ignore
import json
from pathlib import Path

import mlx.core as mx
import pytest

from exo.worker.engines.mlx import utils_mlx

_BUNDLED_MODEL = """
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int


class InnerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = [Layer(args.hidden_size)]


class Indexer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)


class Layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.indexer = Indexer(hidden_size)

    def __call__(self, inputs):
        return self.proj(inputs)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InnerModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs, cache=None):
        hidden = self.model.layers[0](inputs)
        return self.lm_head(hidden)

    def sanitize(self, weights):
        return weights
"""


def _write_bundled_checkpoint(
    model_path: Path,
    *,
    omit_indexer_weight: bool = False,
) -> None:
    model_path.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "model_file": "glm_moe_dsa.py",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "vocab_size": 16,
    }
    (model_path / "config.json").write_text(json.dumps(config))
    (model_path / "glm_moe_dsa.py").write_text(_BUNDLED_MODEL)
    weights = {
        "model.layers.0.proj.weight": mx.ones((8, 8), dtype=mx.float32),
        "model.layers.0.proj.bias": mx.zeros((8,), dtype=mx.float32),
        "model.layers.0.indexer.wk.weight": mx.ones((8, 8), dtype=mx.float32),
        "lm_head.weight": mx.ones((16, 8), dtype=mx.float32),
    }
    if omit_indexer_weight:
        weights.pop("model.layers.0.indexer.wk.weight")
    mx.save_safetensors(str(model_path / "model.safetensors"), weights)


def test_glm_moe_dsa_loader_keeps_bundled_runtime_and_is_strict(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete"
    _write_bundled_checkpoint(complete)

    model, config = utils_mlx.load_model_for_exo(complete)

    assert model.model_type == "glm_moe_dsa"
    assert type(model).__module__ == "custom_model"
    assert config["model_file"] == "glm_moe_dsa.py"

    incomplete = tmp_path / "incomplete"
    _write_bundled_checkpoint(incomplete, omit_indexer_weight=True)
    with pytest.raises(ValueError, match="model.layers.0.indexer.wk.weight"):
        utils_mlx.load_model_for_exo(incomplete)


def test_glm_moe_dsa_loader_rejects_installed_runtime_negative_control(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "native"
    _write_bundled_checkpoint(model_path)
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model_file"] = None
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="requires the bundled glm_moe_dsa.py runtime"):
        utils_mlx.load_model_for_exo(model_path)


def test_other_bundled_architectures_keep_non_strict_loading(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "other"
    _write_bundled_checkpoint(model_path, omit_indexer_weight=True)
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "external_test_model"
    config_path.write_text(json.dumps(config))

    model, _ = utils_mlx.load_model_for_exo(model_path)

    assert model.model_type == "external_test_model"
