import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.cache import supports_prefix_cache
from exo.worker.engines.mlx.generator.generate import patch_embed_tokens
from exo.worker.engines.mlx.vision import DeepseekV41VisionProcessor


class _Group:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 2


def test_official_v41_load_is_strict_without_a_derivative_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    group = _Group()
    calls = []

    def fake_load_model(model_path: Path, **kwargs: object):
        calls.append((model_path, kwargs))
        return object(), {}

    monkeypatch.setattr(utils_mlx, "load_model", fake_load_model)

    utils_mlx.load_model_for_exo(
        tmp_path,
        group,  # type: ignore[arg-type]
        ModelId("deepseek-ai/DeepSeek-V4.1-Flash"),
    )

    assert calls == [
        (
            tmp_path,
            {"lazy": True, "strict": True, "shard_group": group},
        )
    ]


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        (
            utils_mlx.DEEPSEEK_V41_ENGRAM6_MODEL_ID,
            utils_mlx.DEEPSEEK_V41_ENGRAM6_REVISION,
        ),
        (
            utils_mlx.DEEPSEEK_V41_LOWER_MEMORY_MODEL_ID,
            utils_mlx.DEEPSEEK_V41_LOWER_MEMORY_REVISION,
        ),
    ],
)
def test_v41_derivative_load_injects_exact_checkpoint_profile(
    model_id: str,
    revision: str,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41"})
    )
    group = _Group()
    calls = []

    def fake_load_model(model_path: Path, **kwargs: object):
        calls.append((model_path, kwargs))
        return object(), {}

    monkeypatch.setattr(utils_mlx, "load_model", fake_load_model)

    utils_mlx.load_model_for_exo(
        tmp_path,
        group,  # type: ignore[arg-type]
        ModelId(model_id),
    )

    assert calls == [
        (
            tmp_path,
            {
                "lazy": True,
                "strict": True,
                "shard_group": group,
                "model_config": {
                    "checkpoint_profile": {
                        "repository": model_id,
                        "revision": revision,
                    }
                },
            },
        )
    ]


def test_v41_refuses_single_rank_loading(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    with pytest.raises(ValueError, match="distributed shard group"):
        utils_mlx.load_model_for_exo(tmp_path)


def test_v4_and_v41_encoding_identities_are_disjoint() -> None:
    v4 = SimpleNamespace(model="mlx-community/DeepSeek-V4-Flash")
    v41 = SimpleNamespace(model="deepseek-ai/DeepSeek-V4.1-Flash")
    v41_engram6 = SimpleNamespace(model=utils_mlx.DEEPSEEK_V41_ENGRAM6_MODEL_ID)
    v41_lower_memory = SimpleNamespace(
        model=utils_mlx.DEEPSEEK_V41_LOWER_MEMORY_MODEL_ID
    )

    assert utils_mlx._needs_v4_encoding(v4)  # type: ignore[arg-type]
    assert not utils_mlx._needs_v41_encoding(v4)  # type: ignore[arg-type]
    assert utils_mlx._needs_v41_encoding(v41)  # type: ignore[arg-type]
    assert not utils_mlx._needs_v4_encoding(v41)  # type: ignore[arg-type]
    assert utils_mlx._needs_v41_encoding(v41_engram6)  # type: ignore[arg-type]
    assert not utils_mlx._needs_v4_encoding(v41_engram6)  # type: ignore[arg-type]
    assert utils_mlx._needs_v41_encoding(v41_lower_memory)  # type: ignore[arg-type]
    assert not utils_mlx._needs_v4_encoding(v41_lower_memory)  # type: ignore[arg-type]
    assert v41.model != v41_engram6.model
    assert v41.model != v41_lower_memory.model
    assert v41_engram6.model != v41_lower_memory.model


def test_v41_render_uses_the_official_repository_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Encoder:
        @staticmethod
        def encode_messages(**kwargs: object) -> str:
            calls.append(kwargs)
            return "official-v41-prompt"

    import exo.worker.engines.mlx.vendor.deepseek_v41_encoding as encoding

    monkeypatch.setattr(encoding, "encode_messages", Encoder.encode_messages)
    params = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        reasoning_effort="high",
        enable_thinking=True,
        tools=None,
    )

    prompt = utils_mlx.render_chat_template(
        object(),  # type: ignore[arg-type]
        [{"role": "user", "content": "hello"}],
        params,  # type: ignore[arg-type]
    )

    assert prompt == "official-v41-prompt"
    assert calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_mode": "thinking",
            "reasoning_effort": 75,
        }
    ]


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("none", 50),
        ("minimal", 50),
        ("low", 50),
        ("medium", 75),
        ("high", 75),
        ("xhigh", 100),
    ],
)
def test_v41_reasoning_effort_uses_official_numeric_budgets(
    effort: str, budget: int
) -> None:
    params = SimpleNamespace(reasoning_effort=effort)
    assert utils_mlx._v41_reasoning_effort(params) == budget  # type: ignore[arg-type]


def test_v41_shared_attention_state_disables_prefix_cache() -> None:
    model_type = type("Model", (), {})
    model_type.__module__ = "mlx_lm.models.deepseek_v41"

    assert not supports_prefix_cache(model_type())  # type: ignore[arg-type]


def test_v41_integrated_vision_expands_the_official_image_span() -> None:
    image = Image.new("RGB", (4, 4), color=(32, 64, 96))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    class Tokenizer:
        def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
            assert "<｜deepseek_image｜>" in prompt
            assert not add_special_tokens
            return [10, 99, 11]

    class Runtime:
        captured_images = None

        def merge_image_embeddings(self, images, hidden):
            self.captured_images = images
            return hidden

    class FakeModel:
        def __init__(self):
            self.args = SimpleNamespace(
                model_type="deepseek_v41",
                image_token_id=99,
                vision_config=SimpleNamespace(
                    num_hidden_layers=1,
                    patch_size=2,
                    downsample_ratio=2,
                    max_image_tokens=16,
                    min_pixels=0,
                    max_wh_ratio=None,
                ),
            )
            self._runtime = Runtime()

        def embed(self, tokens: mx.array) -> mx.array:
            return mx.zeros((*tokens.shape, 8), dtype=mx.bfloat16)

    FakeModel.__module__ = "mlx_lm.models.deepseek_v41"
    model = FakeModel()
    processor = DeepseekV41VisionProcessor(model)  # type: ignore[arg-type]
    params = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        reasoning_effort="high",
        enable_thinking=True,
        tools=None,
    )
    encoded_image = base64.b64encode(image_bytes.getvalue()).decode()

    result = processor.process(
        images=[encoded_image],  # type: ignore[list-item]
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded_image},
                    },
                ],
            }
        ],
        tokenizer=Tokenizer(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        task_params=params,  # type: ignore[arg-type]
    )

    assert result.prompt_tokens.tolist() == [10, 99, 99, 99, 99, 11]
    assert result.token_types is not None
    assert result.token_types.tolist() == [-1, 0, 1, 2, 3, -1]
    image_input = model._runtime.captured_images[0][0]
    assert image_input.start == 1
    assert image_input.patches.shape == (4, 3, 2, 2)
    assert result.media_regions[0].start_pos == 1
    assert result.media_regions[0].end_pos == 5


def test_v41_prefill_receives_image_token_types_and_precomputed_embeddings() -> None:
    class Cache:
        offset = 0

    class FakeModel:
        layers = []

        def __init__(self):
            self.seen_token_types = None

        def embed(self, tokens: mx.array) -> mx.array:
            return mx.zeros((*tokens.shape, 2), dtype=mx.float32)

        def __call__(self, inputs: mx.array, cache=None, token_types=None):
            self.seen_token_types = token_types
            return self.embed(inputs)

    FakeModel.__module__ = "mlx_lm.models.deepseek_v41"
    model = FakeModel()
    embeddings = mx.array([[[3.0, 4.0], [5.0, 6.0]]])

    with patch_embed_tokens(
        model,  # type: ignore[arg-type]
        embeddings,
        token_count=2,
        token_types=mx.array([0, 1], dtype=mx.int32),
    ) as prefill_model:
        output = prefill_model(mx.array([[99, 99]]), cache=[Cache()])

    assert np.array_equal(np.asarray(output), np.asarray(embeddings))
    assert model.seen_token_types is not None
    assert model.seen_token_types.tolist() == [[0, 1]]


def test_v41_prefill_slices_token_types_from_the_cache_offset() -> None:
    class Cache:
        offset = 2

    class FakeModel:
        layers = []

        def __init__(self):
            self.seen_token_types = None

        def embed(self, tokens: mx.array) -> mx.array:
            return mx.zeros((*tokens.shape, 2), dtype=mx.float32)

        def __call__(self, inputs: mx.array, cache=None, token_types=None):
            self.seen_token_types = token_types
            return self.embed(inputs)

    FakeModel.__module__ = "mlx_lm.models.deepseek_v41"
    model = FakeModel()

    with patch_embed_tokens(
        model,  # type: ignore[arg-type]
        mx.zeros((1, 4, 2), dtype=mx.float32),
        start_offset=2,
        token_count=2,
        token_types=mx.array([-1, -1, 0, 1], dtype=mx.int32),
    ) as prefill_model:
        prefill_model(mx.array([[99, 99]]), cache=[Cache()])

    assert model.seen_token_types is not None
    assert model.seen_token_types.tolist() == [[0, 1]]
