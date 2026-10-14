"""Multi-token prediction (MTP) decoding for Qwen 3.8 Flash Next (`qwen4_exp`).

The checkpoint's MTP head is shipped as a separate drafter
(`Qwen4ExpMTPDraftModel` from mlx-vlm). The drafter proposes tokens from the
target's final pre-mixer hidden state; the target verifies a block of them in
one forward pass. Every emitted token is the target's own sample at its
position, so greedy output matches plain decoding and sampled output keeps the
target's distribution.

This module is model-specific: it attaches only to `mlx_lm.models.qwen4_exp`
models when `EXO_QWEN4_EXP_MTP_DRAFTER` names a drafter checkpoint, and only
on the sequential (non-batched, non-pipeline) generation path.
"""

import os
import time
from collections.abc import Callable, Collection, Generator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import mlx.core as mx
from mlx import nn
from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen4_exp import Model as Qwen4ExpModel
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger

MTP_DRAFTER_ENV = "EXO_QWEN4_EXP_MTP_DRAFTER"
RUNTIME_BLOCK_SIZE = 4
PREFILL_CONTEXT_WINDOW = 1024
FALLBACK_WINDOW_DRAFTED = 32
FALLBACK_MIN_ACCEPTANCE = 0.5
_STATE_ATTR = "_exo_qwen4_exp_mtp"
_CAPTURE_ATTR = "_exo_mtp_capture"
_COMPAT_FIELDS = (
    "hidden_size",
    "hc_count",
    "vocab_size",
    "num_attention_heads",
    "head_dim",
    "rms_norm_eps",
)

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


class Drafter(Protocol):
    """The parts of mlx-vlm's `Qwen4ExpMTPDraftModel` this module uses."""

    config: object

    def bind(self, target_model: nn.Module) -> object: ...

    def reset(self, target_model: nn.Module) -> object: ...

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus: int,
        sampler: None,
        *,
        greedy: bool,
    ) -> object: ...

    def draft_block(
        self,
        last_bonus: int,
        hidden: mx.array,
        cache: None,
        block_size: int,
        sampler: None,
        *,
        greedy: bool,
    ) -> mx.array: ...

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: list[int],
        sampler: None,
        *,
        greedy: bool,
    ) -> object: ...

    def draft_eval_state(self) -> list[mx.array]: ...


class HiddenCapture:
    """Records the target's pre-mixer hidden state on every forward pass.

    In prefill mode it keeps a rolling, evaluated tail of positions so the
    drafter can be primed with recent context without retaining the whole
    prefill graph. In decode mode it keeps the latest (lazy) hidden state.
    """

    def __init__(self) -> None:
        self.mode = "off"
        self.window = PREFILL_CONTEXT_WINDOW
        self.tail: mx.array | None = None
        self.last: mx.array | None = None

    def begin_prefill(self, window: int = PREFILL_CONTEXT_WINDOW) -> None:
        self.mode = "prefill"
        self.window = window
        self.tail = None
        self.last = None

    def end_prefill(self) -> mx.array | None:
        """Return hidden states for the prefilled prompt positions.

        EXO's prefill runs one extra position past its input and the input
        excludes the final prompt token, so the last two captured positions
        are re-run by decode and are dropped here.
        """
        tail = self.tail
        self.mode = "off"
        self.tail = None
        if tail is None or tail.shape[1] <= 2:
            return None
        return tail[:, :-2]

    def record(self, h: mx.array) -> None:
        if self.mode == "prefill":
            tail = h if self.tail is None else mx.concatenate([self.tail, h], axis=1)
            tail = tail[:, -(self.window + 2) :]
            mx.eval(tail)
            self.tail = tail
        elif self.mode == "decode":
            self.last = h


def _install_capture(mixer: nn.Module, capture: HiddenCapture) -> None:
    base = type(mixer)
    if getattr(base, "_exo_mtp_capture_class", False):
        object.__setattr__(mixer, _CAPTURE_ATTR, capture)
        return

    def __call__(  # noqa: N807
        self: nn.Module, h: mx.array, *args: object, **kwargs: object
    ) -> mx.array:
        cast(HiddenCapture, object.__getattribute__(self, _CAPTURE_ATTR)).record(h)
        return base.__call__(self, h, *args, **kwargs)

    capturing = type(
        f"MTPCapture{base.__name__}",
        (base,),
        {"__call__": __call__, "_exo_mtp_capture_class": True},
    )
    object.__setattr__(mixer, _CAPTURE_ATTR, capture)
    mixer.__class__ = capturing


@dataclass
class Qwen4ExpMTPState:
    drafter: Drafter
    capture: HiddenCapture
    configured_block_size: int
    runtime_block_size: int = RUNTIME_BLOCK_SIZE
    accept_lens: list[int] = field(default_factory=list)
    draft_lens: list[int] = field(default_factory=list)
    disabled: bool = False

    def reset_request(self) -> None:
        self.accept_lens = []
        self.draft_lens = []
        self.disabled = False


def get_qwen4_exp_mtp(model: object) -> Qwen4ExpMTPState | None:
    return cast(Qwen4ExpMTPState | None, getattr(model, _STATE_ATTR, None))


def _drafter_incompatibility(model: nn.Module, drafter: Drafter) -> str | None:
    text_config: object = getattr(drafter.config, "text_config", None)
    if text_config is None:
        return "drafter has no text_config"
    args = cast(object, getattr(model, "args"))  # noqa: B009 - untyped mlx_lm model
    target = cast(object, getattr(args, "text"))  # noqa: B009
    for name in _COMPAT_FIELDS:
        expected: object = getattr(target, name, None)
        actual: object = getattr(text_config, name, None)
        if expected != actual:
            return f"{name}: target={expected!r} drafter={actual!r}"
    return None


def _is_pipeline(model: nn.Module) -> bool:
    from exo.worker.engines.mlx.auto_parallel import (
        PipelineFirstLayer,
        PipelineLastLayer,
    )

    return any(
        isinstance(layer, (PipelineFirstLayer, PipelineLastLayer))
        for layer in cast(Sequence[nn.Module], getattr(model, "layers", []))
    )


def attach_qwen4_exp_mtp(model: object, drafter: Drafter) -> Qwen4ExpMTPState | None:
    """Bind a loaded drafter to a qwen4_exp target. Returns None when refused."""
    if not isinstance(model, Qwen4ExpModel):
        return None
    if _is_pipeline(model):
        logger.warning("Qwen4Exp MTP refused: pipeline-parallel placement")
        return None
    problem = _drafter_incompatibility(model, drafter)
    if problem is not None:
        logger.warning(f"Qwen4Exp MTP refused: incompatible drafter ({problem})")
        return None
    capture = HiddenCapture()
    inner = cast(nn.Module, getattr(model, "model"))  # noqa: B009 - untyped mlx_lm model
    mixer = cast(nn.Module, getattr(inner, "hyper_connection_mixer"))  # noqa: B009
    _install_capture(mixer, capture)
    drafter.bind(model)
    block_size: int = getattr(drafter.config, "block_size", 2)
    state = Qwen4ExpMTPState(
        drafter=drafter,
        capture=capture,
        configured_block_size=int(block_size),
    )
    object.__setattr__(model, _STATE_ATTR, state)
    logger.info(
        f"Qwen4Exp MTP attached (configured block {state.configured_block_size}, "
        f"runtime block {state.runtime_block_size})"
    )
    return state


def _load_drafter(path: str) -> Drafter:
    from mlx_vlm.speculative.drafters import (
        load_drafter,  # pyright: ignore[reportUnknownVariableType]
    )

    loaded = cast(tuple[Drafter, str], load_drafter(path))
    drafter, kind = loaded
    if kind != "mtp":
        raise ValueError(f"expected an MTP drafter, got kind {kind!r}")
    return drafter


def maybe_attach_qwen4_exp_mtp(model: object) -> Qwen4ExpMTPState | None:
    """Attach MTP when enabled by environment. Never raises: MTP is optional."""
    path = os.environ.get(MTP_DRAFTER_ENV)
    if not path or not isinstance(model, Qwen4ExpModel):
        return None
    existing = get_qwen4_exp_mtp(model)
    if existing is not None:
        return existing
    try:
        drafter = _load_drafter(path)
    except Exception:  # noqa: BLE001 - MTP is optional; keep plain decode
        logger.opt(exception=True).warning(
            f"Qwen4Exp MTP disabled: could not load drafter from {path}"
        )
        return None
    return attach_qwen4_exp_mtp(model, drafter)


def _effective_block_size(
    requested: int, configured: int, accept_lens: list[int], remaining: int
) -> int:
    # Port of mlx_vlm.speculative.mtp._effective_mtp_block_size at 83c3f54a.
    block_total = min(requested, remaining)
    configured = min(configured, block_total)
    if block_total <= configured or configured <= 1:
        return block_total
    if len(accept_lens) < 8:
        return configured
    recent = accept_lens[-32:]
    hits = sum(1 for accepted in recent if accepted >= configured - 1)
    if hits / len(recent) < 0.65:
        return configured
    return block_total


def _acceptance_too_low(accept_lens: list[int], draft_lens: list[int]) -> bool:
    accepted = drafted = 0
    for a, d in zip(reversed(accept_lens), reversed(draft_lens), strict=True):
        accepted += a
        drafted += d
        if drafted >= FALLBACK_WINDOW_DRAFTED:
            return accepted / drafted < FALLBACK_MIN_ACCEPTANCE
    return False


# Per layer: a copy of the ArraysCache slot list, or the KVCache offset.
_Snapshot = list[list[mx.array | None] | int]


def _snapshot(cache: KVCacheType) -> _Snapshot:
    snap: _Snapshot = []
    for c in cache:
        if isinstance(c, ArraysCache):
            slots = cast(list[mx.array | None], getattr(c, "cache"))  # noqa: B009
            snap.append(list(slots))
        elif isinstance(c, KVCache):
            snap.append(c.offset)
        else:
            raise TypeError(f"Qwen4Exp MTP cannot roll back {type(c).__name__}")
    return snap


def _restore(cache: KVCacheType, snap: _Snapshot) -> None:
    for c, saved in zip(cache, snap, strict=True):
        if isinstance(saved, list):
            setattr(c, "cache", list(saved))  # noqa: B010 - not in the mlx_lm stub
        else:
            kv = cast(KVCache, c)
            kv.trim(kv.offset - saved)


def _cache_state(cache: KVCacheType) -> list[mx.array]:
    # Same evaluation as generate.py's `mx.eval([c.state for c in cache])`.
    return cast(list[mx.array], [c.state for c in cache])


def _mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    state: Qwen4ExpMTPState,
    *,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[LogitsProcessor] | None,
    prompt_cache: KVCacheType,
    context_tokens: mx.array,
    prefill_hidden: mx.array | None,
) -> Generator[tuple[int, mx.array, bool]]:
    drafter = state.drafter
    capture = state.capture
    processors = logits_processors or []
    state.reset_request()
    capture.mode = "decode"
    history: mx.array | None = None

    def forward(tokens: mx.array) -> tuple[mx.array, mx.array]:
        logits = model(tokens[None], cache=prompt_cache)
        assert capture.last is not None
        return logits, capture.last

    def sample_rows(
        logits: mx.array, inputs: mx.array
    ) -> tuple[list[mx.array], list[mx.array]]:
        sampled: list[mx.array] = []
        logprobs: list[mx.array] = []
        for i in range(inputs.shape[0]):
            row = logits[:, i, :]
            if processors:
                consumed = inputs[: i + 1]
                tokens = (
                    consumed if history is None else mx.concatenate([history, consumed])
                )
                for processor in processors:
                    row = processor(tokens, row)
            row_logprobs = row - mx.logsumexp(row, keepdims=True)
            sampled.append(sampler(row_logprobs))
            logprobs.append(row_logprobs.squeeze(0))
        return sampled, logprobs

    def consume(inputs: mx.array) -> None:
        nonlocal history
        if processors:
            history = inputs if history is None else mx.concatenate([history, inputs])

    try:
        with mx.stream(generation_stream):
            hiddens: list[mx.array] = []
            if prefill_hidden is not None:
                hiddens.append(prefill_hidden)
            for i in range(prompt.shape[0] - 1):
                _, h = forward(prompt[i : i + 1])
                hiddens.append(h)
                mx.eval(*_cache_state(prompt_cache))
                mx.clear_cache()
            last_input = prompt[-1:]
            logits, h = forward(last_input)
            hiddens.append(h)
            sampled, logprobs = sample_rows(logits, last_input)
            consume(last_input)
            hidden_all = mx.concatenate(hiddens, axis=1)[:, -PREFILL_CONTEXT_WINDOW:]
            context_ids = context_tokens[-hidden_all.shape[1] :]
            y = cast(int, sampled[0].item())
            drafter.reset(model)
            drafter.prefill_from_target_hidden(
                context_ids[None], hidden_all, y, None, greedy=True
            )
            last_hidden = h
            mx.eval(*drafter.draft_eval_state(), logprobs[0])

        yield y, logprobs[0], False
        emitted = 1
        bonus = y

        while emitted < max_tokens:
            with mx.stream(generation_stream):
                remaining = max_tokens - emitted
                block = (
                    1
                    if state.disabled
                    else _effective_block_size(
                        state.runtime_block_size,
                        state.configured_block_size,
                        state.accept_lens,
                        remaining,
                    )
                )
                if block <= 1:
                    inputs = mx.array([bonus], dtype=mx.int32)
                    logits, last_hidden = forward(inputs)
                    sampled, logprobs = sample_rows(logits, inputs)
                    consume(inputs)
                    bonus = cast(int, sampled[0].item())
                    emitted_tokens = [(bonus, logprobs[0], False)]
                else:
                    draft = drafter.draft_block(
                        bonus, last_hidden, None, block, None, greedy=True
                    )
                    inputs = mx.concatenate(
                        [mx.array([bonus], dtype=mx.int32), draft[0].astype(mx.int32)]
                    )
                    snap = _snapshot(prompt_cache)
                    logits, verify_hidden = forward(inputs)
                    sampled, logprobs = sample_rows(logits, inputs)
                    targets = mx.concatenate(sampled)
                    mx.eval(targets, draft)
                    target_list = cast(list[int], targets.tolist())
                    draft_list = cast(list[int], draft[0].tolist())
                    accepted = 0
                    while (
                        accepted < len(draft_list)
                        and draft_list[accepted] == target_list[accepted]
                    ):
                        accepted += 1
                    new_tokens = draft_list[:accepted] + [target_list[accepted]]
                    if accepted < len(draft_list):
                        _restore(prompt_cache, snap)
                        forward(inputs[: accepted + 1])
                    consume(inputs[: accepted + 1])
                    drafter.accept_verified_tokens(
                        verify_hidden, draft, accepted, new_tokens, None, greedy=True
                    )
                    state.accept_lens.append(accepted)
                    state.draft_lens.append(len(draft_list))
                    if _acceptance_too_low(state.accept_lens, state.draft_lens):
                        state.disabled = True
                        logger.info(
                            "Qwen4Exp MTP: acceptance below floor, drafting off"
                        )
                    last_hidden = verify_hidden[:, accepted : accepted + 1]
                    bonus = new_tokens[-1]
                    emitted_tokens = [
                        (token, logprobs[i], i < accepted)
                        for i, token in enumerate(new_tokens)
                    ]
                mx.eval(*_cache_state(prompt_cache), *[t[1] for t in emitted_tokens])
            for token, token_logprobs, from_draft in emitted_tokens:
                if emitted >= max_tokens:
                    break
                previous = emitted
                emitted += 1
                yield token, token_logprobs, from_draft
                if previous // 256 != emitted // 256:
                    mx.clear_cache()
    finally:
        capture.mode = "off"
        capture.last = None
        if state.draft_lens:
            drafted = sum(state.draft_lens)
            logger.info(
                f"Qwen4Exp MTP: accepted {sum(state.accept_lens)}/{drafted} drafted "
                f"tokens over {len(state.draft_lens)} rounds"
                + (" (fell back to plain decode)" if state.disabled else "")
            )


def mtp_stream_generate(
    model: object,
    tokenizer: TokenizerWrapper,
    prompt: mx.array,
    state: Qwen4ExpMTPState,
    *,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[LogitsProcessor] | None,
    prompt_cache: KVCacheType,
    context_tokens: mx.array,
    prefill_hidden: mx.array | None = None,
) -> Generator[GenerationResponse]:
    """Drop-in for `mlx_lm.generate.stream_generate` on the MTP path.

    Mirrors stream_generate's detokenizing wrapper so callers see the same
    response sequence (including the EOS and max-token handling).
    """
    target = cast(nn.Module, model)
    detokenizer = cast(StreamingDetokenizer, tokenizer.detokenizer)
    eos_token_ids: Collection[int] = tokenizer.eos_token_ids or ()
    token_generator = _mtp_generate_step(
        prompt,
        target,
        state,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=prompt_cache,
        context_tokens=context_tokens,
        prefill_hidden=prefill_hidden,
    )
    with wired_limit(target, [generation_stream]):
        tic = time.perf_counter()
        prompt_tps = 0.0
        n = 0
        token = 0
        logprobs = mx.array([])
        from_draft = False
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )
        token_generator.close()

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in eos_token_ids else "length",
        )
