from mlx import core as mx, nn
from mlx_vlm.models.cache import ArraysCache, CacheList
from .config import TextConfig

class LanguageModelOutput:
    logits: mx.array

class Glm5NextMLP(nn.Module):
    gate_up_proj: nn.Linear | nn.QuantizedLinear
    down_proj: nn.Linear | nn.QuantizedLinear

class SwitchGLU(nn.Module):
    gate_proj: nn.Module
    up_proj: nn.Module
    down_proj: nn.Module

class Glm5NextMoE(nn.Module):
    shared_experts: Glm5NextMLP
    switch_mlp: SwitchGLU
    sharding_group: mx.distributed.Group

class DecoderLayer(nn.Module):
    mlp: Glm5NextMLP | Glm5NextMoE

class LanguageModel(nn.Module):
    layers: list[DecoderLayer]
    def __init__(self, config: TextConfig): ...
    def __call__(
        self, inputs: mx.array, cache: object = None
    ) -> LanguageModelOutput: ...
    def make_cache(self) -> list[ArraysCache | CacheList]: ...
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]: ...
