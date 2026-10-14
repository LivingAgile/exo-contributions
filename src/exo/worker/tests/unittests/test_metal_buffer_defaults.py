from exo.worker.runner.bootstrap import metal_buffer_defaults


def test_metal_buffer_defaults_apply_only_to_glm5_next_and_qwen4_exp() -> None:
    for model_type in ("glm5_next", "qwen4_exp"):
        assert metal_buffer_defaults(model_type) == {
            "MLX_MAX_MB_PER_BUFFER": "2048",
            "MLX_MAX_OPS_PER_BUFFER": "100000",
        }
    for other in (None, "", "gemma4", "qwen3_5", "qwen3_5_moe", "qwen3_next"):
        assert metal_buffer_defaults(other) == {}
