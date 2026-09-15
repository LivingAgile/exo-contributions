import mlx.core as mx
import numpy as np
from mlx_vlm.models.muse_glimmer.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.muse_glimmer.muse_glimmer import Model
from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
    MuseGlimmerImageProcessor,
)
from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat
from PIL import Image


def _tiny_model() -> Model:
    text = TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=32,
    )
    vision = VisionConfig(
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=16,
    )
    return Model(
        ModelConfig(
            text_config=text,
            vision_config=vision,
            out_hidden_size=32,
            projector_hidden_size=32,
        )
    )


def test_muse_glimmer_package_constructs_language_and_vision_towers():
    model = _tiny_model()

    assert len(model.layers) == 2
    assert len(model.vision_tower.layers) == 2
    assert MODEL_CONFIG["muse_glimmer"] is MessageFormat.LIST_WITH_IMAGE_FIRST


def test_muse_glimmer_processor_and_checkpoint_key_sanitization():
    model = _tiny_model()
    processor = MuseGlimmerImageProcessor(max_image_tokens=16)
    processed = processor(
        images=Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8))
    )

    assert processed["pixel_values"].shape == (4, 1176)
    assert processed["image_grid_thw"].tolist() == [[1, 2, 2]]

    sanitized = model.sanitize(
        {
            "model.language_model.embed_tokens.weight": mx.zeros((2, 2)),
            "model.vision_tower.patch_embedder.patch_embedding.weight": mx.zeros(
                (2, 2)
            ),
            "model.vision_adapter.fc1.weight": mx.zeros((2, 2)),
            "model.vision_projection.weight": mx.zeros((2, 2)),
            "model.perception_emb_norm.weight": mx.zeros((2, 2)),
            "model.rotary_emb.inv_freq": mx.zeros((1,)),
        }
    )

    assert sorted(sanitized) == [
        "language_model.model.embed_tokens.weight",
        "perception_emb_norm.weight",
        "vision_adapter.fc1.weight",
        "vision_projection.weight",
        "vision_tower.patch_embedder.patch_embedding.weight",
    ]