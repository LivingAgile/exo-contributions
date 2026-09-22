from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import DecoderLayer, LanguageModel


@dataclass
class ModelArgs:
    text_config: TextConfig

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelArgs":
        return cls(TextConfig.from_dict(cast(dict[str, Any], config["text_config"])))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = LanguageModel(args.text_config)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.language_model.layers

    def __call__(self, inputs: mx.array, cache: object = None) -> mx.array:
        logits = self.language_model(inputs, cache=cache).logits
        if cache is not None:
            for entry in cast(list[ArraysCache | CacheList], cache):
                if isinstance(entry, CacheList):
                    entry.caches = (*entry.caches[:-1], KVCache())
        return logits

    def make_cache(self) -> list[ArraysCache | CacheList]:
        return self.language_model.make_cache()

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        text_weights: dict[str, mx.array] = {}
        for name, value in weights.items():
            if name.startswith(("model.visual.", "vision_tower.")):
                continue
            if name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith(("lm_head.", "model.")):
                name = "language_model." + name
            text_weights[name] = value
        return self.language_model.sanitize(text_weights)
