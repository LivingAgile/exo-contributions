from exo.shared.models.model_cards import ConfigData


def test_muse_glimmer_config_supports_tensor_with_vision_metadata() -> None:
    config = ConfigData.model_validate(
        {
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 200092,
            "text_config": {
                "hidden_size": 6656,
                "num_hidden_layers": 52,
                "num_key_value_heads": 2,
                "max_position_embeddings": 131072,
            },
            "vision_config": {
                "model_type": "muse_glimmer_vision",
                "num_hidden_layers": 50,
            },
        },
        context={"model_id": "meta-models/Muse-Glimmer-30B"},
    )

    assert config.architectures == ["MuseGlimmerForConditionalGeneration"]
    assert config.layer_count == 52
    assert config.supports_tensor
    assert config.vision is not None
    assert config.vision.model_type == "muse_glimmer"
    assert config.vision.weights_repo == "meta-models/Muse-Glimmer-30B"