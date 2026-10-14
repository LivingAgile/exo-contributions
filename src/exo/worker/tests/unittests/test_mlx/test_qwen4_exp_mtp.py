# type: ignore
import copy
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map
from mlx_lm.generate import stream_generate
from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import ArraysCache
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
    Qwen4ExpMTPDraftModel,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import exo.worker.engines.mlx.qwen4_exp_mtp as mtp_module
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.generate import ban_token_ids, mlx_generate
from exo.worker.engines.mlx.qwen4_exp_mtp import (
    MTP_DRAFTER_ENV,
    Qwen4ExpMTPState,
    attach_qwen4_exp_mtp,
    get_qwen4_exp_mtp,
    maybe_attach_qwen4_exp_mtp,
    mtp_stream_generate,
)
from exo.worker.engines.mlx.types import Model as ExoModel
from exo.worker.tests.unittests.test_mlx.test_tp_bit_exact import MODEL_CONFIGS

VOCAB = 512
EOS = 1


def _text_config() -> dict[str, Any]:
    args = cast(dict[str, Any], MODEL_CONFIGS["qwen4_exp"]["args"])
    return copy.deepcopy(cast(dict[str, Any], args["text_config"]))


# Flash Next's real attention head shape. The fused attention kernel gives
# different bits for several query rows per call than for one at this shape,
# but not at the tiny default shape. The indexer head grows with it so that it
# still holds the rotary dimensions.
REAL_HEADS = {
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "indexer_head_dim": 128,
}


def _target(seed: int = 7, **text_overrides: Any) -> qwen4_exp.Model:
    mx.random.seed(seed)
    args = copy.deepcopy(cast(dict[str, Any], MODEL_CONFIGS["qwen4_exp"]["args"]))
    args["text_config"].update(text_overrides)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(**args))
    mx.eval(model.parameters())
    return model


def _drafter_config(**overrides: Any) -> Qwen4ExpMTPConfig:
    text = _text_config()
    text.update(
        model_type="qwen4_exp_text",
        max_position_embeddings=4096,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    )
    text.update(overrides)
    return Qwen4ExpMTPConfig.from_dict(
        {"model_type": "qwen4_exp_mtp", "text_config": text}
    )


class OracleDrafter(Qwen4ExpMTPDraftModel):
    """Real drafter whose proposals are overridden from a known continuation.

    `pattern[r]` is the number of correct leading drafts in round r (-1 means
    all correct); later rounds repeat the last entry. Position tracking uses
    the same accept calls the generator makes, so the drafter's own cache
    bookkeeping is exercised unchanged.
    """

    def __init__(
        self, config: Qwen4ExpMTPConfig, reference: list[int], pattern: list[int]
    ):
        super().__init__(config)
        self.reference = reference
        self.pattern = pattern
        self.position = 1
        self.round = 0

    def reset(self, target_model: Any) -> list[Any]:
        self.position = 1
        self.round = 0
        return super().reset(target_model)

    def draft_block(
        self,
        last_bonus: Any,
        hidden: mx.array,
        cache: Any,
        block_size: int,
        sampler: Any,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        real = super().draft_block(
            last_bonus, hidden, cache, block_size, sampler, token_dtype, greedy
        )
        correct = self.pattern[min(self.round, len(self.pattern) - 1)]
        drafts: list[int] = []
        for i in range(real.shape[1]):
            index = self.position + i
            truth = self.reference[index] if index < len(self.reference) else 2
            ok = correct < 0 or i < correct
            drafts.append(truth if ok else (truth + 1) % VOCAB)
        self.round += 1
        return mx.array([drafts], dtype=token_dtype)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: list[int],
        sampler: Any,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        self.position += accepted + 1
        super().accept_verified_tokens(
            verify_hidden,
            draft_tokens,
            accepted,
            new_tokens,
            sampler,
            token_dtype,
            greedy,
        )


def _tokenizer() -> TokenizerWrapper:
    vocab = {f"w{i}": i for i in range(VOCAB)}
    vocab["[UNK]"] = VOCAB
    native = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    wrapper = TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=native, unk_token="[UNK]"),
        eos_token_ids={EOS},
    )
    return wrapper


def _greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


PROMPT = [(7 * i + 3) % (VOCAB - 2) + 2 for i in range(40)]


def _plain(
    model: qwen4_exp.Model,
    max_tokens: int,
    processors: list[Any] | None = None,
    sampler: Any = _greedy,
) -> tuple[list[int], list[Any]]:
    cache = model.make_cache()
    prompt = mx.array(PROMPT)
    model(prompt[None, :-2], cache=cache)
    mx.eval([c.state for c in cache])
    tokens = [
        r.token
        for r in stream_generate(
            model,
            _tokenizer(),
            prompt[-2:],
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=cache,
            prefill_step_size=1,
        )
    ]
    return tokens, cache


def _mtp(
    model: qwen4_exp.Model,
    state: Qwen4ExpMTPState,
    max_tokens: int,
    processors: list[Any] | None = None,
    sampler: Any = _greedy,
) -> tuple[list[int], list[Any]]:
    cache = model.make_cache()
    prompt = mx.array(PROMPT)
    state.capture.begin_prefill()
    model(prompt[None, :-2], cache=cache)
    tail = state.capture.tail
    state.capture.mode = "off"
    assert tail is not None and tail.shape[1] == len(PROMPT) - 2
    mx.eval([c.state for c in cache])
    tokens = [
        r.token
        for r in mtp_stream_generate(
            model,
            _tokenizer(),
            prompt[-2:],
            state,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=cache,
            context_tokens=prompt,
            prefill_hidden=tail,
        )
    ]
    return tokens, cache


def _assert_cache_matches_sequential(
    model: qwen4_exp.Model, cache: list[Any], consumed: list[int]
) -> None:
    reference = model.make_cache()
    model(mx.array([PROMPT[:-2]]), cache=reference)
    for token in consumed:
        model(mx.array([[token]]), cache=reference)
    mx.eval([c.state for c in reference])
    for got, want in zip(cache, reference, strict=True):
        if isinstance(want, ArraysCache):
            for g, w in zip(got.cache, want.cache, strict=True):
                assert (g is None) == (w is None)
                if w is not None:
                    assert mx.allclose(g, w, atol=1e-4, rtol=1e-4).item()
        else:
            assert got.offset == want.offset
            n = want.offset
            assert mx.allclose(
                got.keys[..., :n, :], want.keys[..., :n, :], atol=1e-4
            ).item()
            assert mx.allclose(
                got.values[..., :n, :], want.values[..., :n, :], atol=1e-4
            ).item()
            gi, wi = got.indexer.keys, want.indexer.keys
            assert gi.shape == wi.shape
            assert mx.allclose(gi, wi, atol=1e-4).item()


def _attach(model: qwen4_exp.Model, drafter: Qwen4ExpMTPDraftModel) -> Qwen4ExpMTPState:
    state = attach_qwen4_exp_mtp(model, drafter)
    assert state is not None
    return state


@pytest.mark.parametrize(
    "pattern",
    [[-1], [0], [1], [2, 0, -1, 1]],
    ids=["full-accept", "full-reject", "partial", "mixed"],
)
def test_mtp_greedy_matches_plain_and_leaves_exact_cache(pattern: list[int]) -> None:
    model = _target()
    reference, _ = _plain(model, 24)
    assert len(reference) == 24
    drafter = OracleDrafter(_drafter_config(), reference, pattern)
    state = _attach(model, drafter)
    state.runtime_block_size = 4
    # Force the runtime depth immediately so every round drafts 3 tokens.
    state.configured_block_size = 4

    got, cache = _mtp(model, state, 24)

    assert got == reference
    assert sum(state.draft_lens) > 0
    if pattern == [-1]:
        assert all(
            a == d for a, d in zip(state.accept_lens, state.draft_lens, strict=True)
        )
    if pattern == [0]:
        assert all(a == 0 for a in state.accept_lens)
    consumed = [PROMPT[-2], PROMPT[-1], *got[:-1]]
    _assert_cache_matches_sequential(model, cache, consumed)


def test_mtp_with_untrained_drafter_matches_plain() -> None:
    model = _target(11)
    reference, _ = _plain(model, 40)
    mx.random.seed(3)
    drafter = Qwen4ExpMTPDraftModel(_drafter_config())
    mx.eval(drafter.parameters())
    state = _attach(model, drafter)
    got, _ = _mtp(model, state, 40)
    assert got == reference


def test_mtp_logits_processors_match_plain() -> None:
    model = _target()

    def follow_history(history: mx.array, logits: mx.array) -> mx.array:
        # Output depends on the exact history (last token and length), so a
        # verify row that sees the wrong history produces a different token.
        target = (int(history[-1].item()) + history.shape[0]) % VOCAB
        bonus = mx.zeros_like(logits)
        bonus[..., target] = 100.0
        return logits + bonus

    def processors() -> list[Any]:
        return [
            follow_history,
            ban_token_ids([EOS]),
            *make_logits_processors(repetition_penalty=1.3),
        ]

    reference, _ = _plain(model, 24, processors())
    drafter = OracleDrafter(_drafter_config(), reference, [1, -1, 0])
    state = _attach(model, drafter)
    state.configured_block_size = 4
    got, _ = _mtp(model, state, 24, processors())
    assert got == reference


def test_low_acceptance_falls_back_to_plain_decode() -> None:
    model = _target()
    reference, _ = _plain(model, 40)
    drafter = OracleDrafter(_drafter_config(), reference, [0])
    state = _attach(model, drafter)
    state.configured_block_size = 4
    got, _ = _mtp(model, state, 40)
    assert got == reference
    assert state.disabled
    assert sum(state.draft_lens) == 33  # 11 rounds of 3, then plain steps


def test_incompatible_drafter_is_refused() -> None:
    model = _target()
    drafter = Qwen4ExpMTPDraftModel(_drafter_config(rms_norm_eps=1e-5))
    assert attach_qwen4_exp_mtp(model, drafter) is None
    assert get_qwen4_exp_mtp(model) is None


def _served_like_target(seed: int, **text_overrides: Any) -> qwen4_exp.Model:
    """bf16 weights with 8-bit quantized projections, like the served checkpoint."""
    model = _target(seed, **text_overrides)
    model.update(
        tree_map(
            lambda p: p.astype(mx.bfloat16)
            if mx.issubdtype(p.dtype, mx.floating)
            else p,
            model.parameters(),
        )
    )
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda _p, m: hasattr(m, "to_quantized")
        and m.weight.shape[-1] % 32 == 0,
    )
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize(
    ("seed", "heads"),
    [(7, {}), (11, {}), (7, REAL_HEADS)],
    ids=["seed7", "seed11", "real-heads"],
)
def test_mtp_logprobs_are_bit_identical_to_plain(
    seed: int, heads: dict[str, int]
) -> None:
    # Plain decode scores one token per forward pass; MTP scores a block of
    # tokens per pass. Every emitted logprob row must still match bit for bit,
    # otherwise sampled and reasoning output drifts from plain decoding.
    model = _served_like_target(seed, **heads)
    tokenizer = _tokenizer()
    cache = model.make_cache()
    prompt = mx.array(PROMPT)
    model(prompt[None, :-2], cache=cache)
    mx.eval([c.state for c in cache])
    plain = list(
        stream_generate(
            model,
            tokenizer,
            prompt[-2:],
            max_tokens=24,
            sampler=_greedy,
            prompt_cache=cache,
            prefill_step_size=1,
        )
    )
    reference = [r.token for r in plain]

    model = _served_like_target(seed, **heads)
    drafter = OracleDrafter(_drafter_config(**heads), reference, [-1, 1, -1, 0])
    if heads:
        # Like the served drafter. Metal has no float32 attention kernel at
        # head size 256.
        drafter.update(
            tree_map(
                lambda p: p.astype(mx.bfloat16)
                if mx.issubdtype(p.dtype, mx.floating)
                else p,
                drafter.parameters(),
            )
        )
    state = _attach(model, drafter)
    state.configured_block_size = 4
    cache = model.make_cache()
    state.capture.begin_prefill()
    model(prompt[None, :-2], cache=cache)
    tail = state.capture.tail
    state.capture.mode = "off"
    mx.eval([c.state for c in cache])
    got = list(
        mtp_stream_generate(
            model,
            tokenizer,
            prompt[-2:],
            state,
            max_tokens=24,
            sampler=_greedy,
            logits_processors=None,
            prompt_cache=cache,
            context_tokens=prompt,
            prefill_hidden=tail,
        )
    )

    assert [r.token for r in got] == reference
    assert sum(state.accept_lens) > 0
    for want, have in zip(plain, got, strict=True):
        assert mx.array_equal(want.logprobs, have.logprobs).item()


@pytest.mark.parametrize("kv_len", [256, 1500])
def test_verify_attention_matches_one_row_decode_at_real_heads(kv_len: int) -> None:
    # Each verify row must give the same bits as a one-token decode step that
    # sees only the keys up to its own position.
    _attach(_target(), Qwen4ExpMTPDraftModel(_drafter_config()))
    rows, h, kvh, d = 5, 24, 2, 256
    mx.random.seed(1)
    q = mx.random.normal((1, h, rows, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, kvh, kv_len, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, kvh, kv_len, d)).astype(mx.bfloat16)
    causal = mx.arange(kv_len - rows, kv_len)[:, None] >= mx.arange(kv_len)
    keep = causal & (mx.random.uniform(shape=(1, 1, rows, kv_len)) > 0.3)
    for mask in ("causal", keep):
        token = mtp_module._stepwise_recurrence.set(True)
        try:
            block = qwen4_exp.scaled_dot_product_attention(
                q, k, v, cache=None, scale=d**-0.5, mask=mask
            )
        finally:
            mtp_module._stepwise_recurrence.reset(token)
        for t in range(rows):
            end = kv_len - rows + t + 1
            row_mask = None if isinstance(mask, str) else mask[..., t : t + 1, :end]
            want = mx.fast.scaled_dot_product_attention(
                q[:, :, t : t + 1],
                k[:, :, :end],
                v[:, :, :end],
                scale=d**-0.5,
                mask=row_mask,
            )
            assert mx.array_equal(block[:, :, t : t + 1], want).item()


def test_attach_requires_env_and_qwen4_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _target()
    loaded: list[str] = []

    def fake_load(path: str) -> Qwen4ExpMTPDraftModel:
        loaded.append(path)
        return Qwen4ExpMTPDraftModel(_drafter_config())

    monkeypatch.setattr(mtp_module, "_load_drafter", fake_load)
    monkeypatch.delenv(MTP_DRAFTER_ENV, raising=False)
    assert maybe_attach_qwen4_exp_mtp(model) is None
    monkeypatch.setenv(MTP_DRAFTER_ENV, "/models/drafter")
    assert maybe_attach_qwen4_exp_mtp(object()) is None
    assert loaded == []
    assert maybe_attach_qwen4_exp_mtp(model) is not None
    assert loaded == ["/models/drafter"]


def test_mlx_generate_with_mtp_matches_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _tokenizer()
    prompt = " ".join(f"w{t}" for t in PROMPT)
    task = TextGenerationTaskParams(
        model=ModelId("test"),
        input=[],
        max_output_tokens=30,
        temperature=0.0,
        bench=True,
    )

    def run(model: qwen4_exp.Model) -> list[int]:
        return [
            r.token
            for r in mlx_generate(
                model=cast(ExoModel, cast(object, model)),
                tokenizer=tokenizer,
                task=task,
                prompt=prompt,
                kv_prefix_cache=None,
                group=None,
            )
        ]

    plain_model = _target(5)
    reference = run(plain_model)
    model = _target(5)
    drafter = OracleDrafter(_drafter_config(), reference, [1, -1, 0, 2])
    state = _attach(model, drafter)
    state.configured_block_size = 4
    tails: list[int] = []
    end_prefill = state.capture.end_prefill

    def observe_end_prefill() -> mx.array | None:
        tail = end_prefill()
        tails.append(-1 if tail is None else tail.shape[1])
        return tail

    monkeypatch.setattr(state.capture, "end_prefill", observe_end_prefill)
    got = run(model)
    assert got == reference
    assert sum(state.accept_lens) > 0
    # Hidden states cover exactly the prompt positions decode does not re-run.
    assert tails == [len(PROMPT) - 2]
