"""Neural text compressor with context mixing.

Lossless compression using SmolLM2-135M + ensemble of adaptive models:

  1. N-gram model   – fast local pattern prediction (order 1-4)
  2. LZP model      – long-range exact match prediction (order 4-8)
  3. Context mixer   – adaptive linear blending of all models
  4. Adaptive head   – online bias correction on LLM logits
  5. Confidence skip – bypass the LLM when n-gram is confident enough

The compressor and decompressor maintain identical model states by
processing tokens in the same order with the same updates, ensuring
lossless symmetry.

This module provides the core NeuralCompressor class used as workers
by ParallelNeuralCompressor (NC05/NC06 formats).
"""

import gc
import struct
import sys

import numpy as np

from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder
from model_wrapper import ModelWrapper
from utils import probs_to_cdf, CdfConverter
from ngram_model import NgramModel
from lzp_model import LZPModel
from context_mixer import ContextMixer
from adaptive_head import AdaptiveHead

# ---- CDF precision ----

# Enhanced CDF: 2^24 instead of the original 2^16.
# With vocab_size=49152, 2^16 wastes 75% of the CDF range on MIN_PROB
# floors, adding ~2 bits overhead per token.  2^24 wastes only 0.3%,
# cutting overhead to ~0.004 bits/token.
# Safe with 32-bit arithmetic coder (min symbol width = 64).
CDF_TOTAL = 1 << 24

# Config flags (stored in file header for decompressor)
FLAG_NGRAM = 0x01
FLAG_LZP = 0x02
FLAG_ADAPTIVE_HEAD = 0x04
FLAG_CONFIDENCE_SKIP = 0x08

# ---- Segmentation constants ----

CHUNK_TYPE_TEXT = 0x54     # 'T'
CHUNK_TYPE_BINARY = 0x42   # 'B'
MIN_TEXT_RUN = 64
MAX_BRIDGE_GAP = 8
MIN_BINARY_CHUNK = 64

# Binary blob compression methods
BLOB_GZIP = 0x47   # 'G'
BLOB_LZMA = 0x4C   # 'L'
BLOB_RAW = 0x52     # 'R'
LZMA_THRESHOLD = 4096

# Bytes considered "text-like": printable ASCII (32-126) + tab/LF/CR
TEXT_BYTES = frozenset(range(32, 127)) | {9, 10, 13}

# Bytes that the SmolLM2 tokenizer silently drops during encode→decode.
# A binary chunk containing any of these must NEVER be absorbed into a
# text chunk, or the roundtrip will lose data.
TOKENIZER_LOSSY_BYTES = frozenset({0x04, 0x06, 0x13, 0x14, 0x16, 0x1D})

# ---- Default hyperparameters ----

DEFAULT_NGRAM_ORDER = 4
DEFAULT_LZP_MAX_ORDER = 8
DEFAULT_LZP_MIN_ORDER = 4
DEFAULT_MIXER_LR = 0.5
DEFAULT_ADAPTIVE_LR = 0.001
DEFAULT_SKIP_THRESHOLD = 1.5  # bits; skip LLM only when n-gram is VERY confident
DEFAULT_WARMUP = 100  # tokens — use LLM alone while secondary models accumulate data
DEFAULT_TEMPERATURE = 1.0  # softmax temperature; <1 sharpens, >1 softens

# When LLM is skipped and both n-gram and LZP are active,
# blend them with these fixed weights.
SKIP_NGRAM_WEIGHT = 0.7
SKIP_LZP_WEIGHT = 0.3


def _segment_chunks(data: bytes) -> list[tuple[int, int, int]]:
    """Segment data into text and binary chunks.

    Returns list of (chunk_type, offset, length) tuples where chunk_type
    is CHUNK_TYPE_TEXT or CHUNK_TYPE_BINARY.
    """
    if not data:
        return []

    # Step 1: classify each byte and collect contiguous runs
    runs = []  # list of (type, offset, length)
    current_type = CHUNK_TYPE_TEXT if data[0] in TEXT_BYTES else CHUNK_TYPE_BINARY
    run_start = 0

    for i in range(1, len(data)):
        byte_type = CHUNK_TYPE_TEXT if data[i] in TEXT_BYTES else CHUNK_TYPE_BINARY
        if byte_type != current_type:
            runs.append((current_type, run_start, i - run_start))
            current_type = byte_type
            run_start = i
    runs.append((current_type, run_start, len(data) - run_start))

    # Step 2: demote short text runs to binary
    runs = [
        (CHUNK_TYPE_BINARY if t == CHUNK_TYPE_TEXT and length < MIN_TEXT_RUN else t,
         off, length)
        for t, off, length in runs
    ]

    # Step 3: merge adjacent same-type runs (after demotion)
    merged = [runs[0]]
    for t, off, length in runs[1:]:
        if t == merged[-1][0]:
            prev_t, prev_off, prev_len = merged[-1]
            merged[-1] = (prev_t, prev_off, prev_len + length)
        else:
            merged.append((t, off, length))
    runs = merged

    # Step 4: bridge small binary gaps between text runs
    if len(runs) >= 3:
        bridged = [runs[0]]
        i = 1
        while i < len(runs) - 1:
            prev_t = bridged[-1][0]
            curr_t, curr_off, curr_len = runs[i]
            next_t = runs[i + 1][0]

            if (prev_t == CHUNK_TYPE_TEXT and curr_t == CHUNK_TYPE_BINARY
                    and next_t == CHUNK_TYPE_TEXT and curr_len <= MAX_BRIDGE_GAP):
                # Bridge: merge prev + gap + next into one text chunk
                prev_t2, prev_off, prev_len = bridged[-1]
                next_t2, next_off, next_len = runs[i + 1]
                bridged[-1] = (CHUNK_TYPE_TEXT, prev_off,
                               prev_len + curr_len + next_len)
                i += 2
            else:
                bridged.append((curr_t, curr_off, curr_len))
                i += 1
        if i < len(runs):
            bridged.append(runs[i])
        runs = bridged

    # Step 5: final merge of adjacent same-type runs
    merged = [runs[0]]
    for t, off, length in runs[1:]:
        if t == merged[-1][0]:
            prev_t, prev_off, prev_len = merged[-1]
            merged[-1] = (prev_t, prev_off, prev_len + length)
        else:
            merged.append((t, off, length))
    runs = merged

    # Step 6: absorb small binary chunks into adjacent text chunks,
    # but only if the chunk contains no tokenizer-lossy bytes.
    if len(runs) >= 2:
        absorbed = []
        i = 0
        while i < len(runs):
            t, off, length = runs[i]
            if (t == CHUNK_TYPE_BINARY and length < MIN_BINARY_CHUNK
                    and not TOKENIZER_LOSSY_BYTES.intersection(
                        data[off:off + length])):
                left_text = (absorbed and absorbed[-1][0] == CHUNK_TYPE_TEXT)
                right_text = (i + 1 < len(runs)
                              and runs[i + 1][0] == CHUNK_TYPE_TEXT)
                if left_text and right_text:
                    # Merge left + this + right into one text chunk
                    prev_t, prev_off, prev_len = absorbed[-1]
                    _next_t, _next_off, next_len = runs[i + 1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length + next_len)
                    i += 2
                    continue
                elif left_text:
                    prev_t, prev_off, prev_len = absorbed[-1]
                    absorbed[-1] = (CHUNK_TYPE_TEXT, prev_off,
                                    prev_len + length)
                    i += 1
                    continue
                elif right_text:
                    # Convert to text; will merge with next text chunk
                    absorbed.append((CHUNK_TYPE_TEXT, off, length))
                    i += 1
                    continue
            absorbed.append((t, off, length))
            i += 1
        runs = absorbed

        # Final merge after absorption
        merged = [runs[0]]
        for t, off, length in runs[1:]:
            if t == merged[-1][0]:
                prev_t, prev_off, prev_len = merged[-1]
                merged[-1] = (prev_t, prev_off, prev_len + length)
            else:
                merged.append((t, off, length))
        runs = merged

    return runs


def _entropy(probs: np.ndarray, buf: np.ndarray = None) -> float:
    """Compute Shannon entropy in bits.

    Args:
        probs: Probability distribution.
        buf: Optional pre-allocated buffer (same shape as probs) to
             avoid 768 KB of temporary allocations per call.
    """
    if buf is not None:
        np.add(probs, 1e-10, out=buf)
        np.log2(buf, out=buf)
        buf *= probs
        return -float(buf.sum())
    log_p = np.log2(probs + 1e-10)
    return -float((probs * log_p).sum())


class NeuralCompressor:
    """Lossless neural compressor with ensemble prediction."""

    def __init__(
        self,
        model: ModelWrapper = None,
        verbose: bool = True,
        *,
        use_ngram: bool = True,
        use_lzp: bool = True,
        use_adaptive_head: bool = True,
        use_confidence_skip: bool = True,
        ngram_order: int = DEFAULT_NGRAM_ORDER,
        lzp_max_order: int = DEFAULT_LZP_MAX_ORDER,
        lzp_min_order: int = DEFAULT_LZP_MIN_ORDER,
        mixer_lr: float = DEFAULT_MIXER_LR,
        adaptive_lr: float = DEFAULT_ADAPTIVE_LR,
        skip_threshold: float = DEFAULT_SKIP_THRESHOLD,
        warmup: int = DEFAULT_WARMUP,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.verbose = verbose
        self.model = model or ModelWrapper(verbose=verbose)
        self.vocab_size = self.model.vocab_size

        # Progress counters (read by ParallelNeuralCompressor monitor)
        self._progress = 0
        self._progress_total = 0

        # Feature flags
        self.use_ngram = use_ngram
        self.use_lzp = use_lzp
        self.use_adaptive_head = use_adaptive_head
        # Confidence skip requires n-gram to compute entropy
        self.use_confidence_skip = use_confidence_skip and use_ngram
        self.skip_threshold = skip_threshold
        self.warmup = warmup
        self.temperature = temperature

        # Secondary models
        self.ngram = NgramModel(
            max_order=ngram_order, vocab_size=self.vocab_size
        ) if use_ngram else None

        self.lzp = LZPModel(
            max_order=lzp_max_order, min_order=lzp_min_order,
            vocab_size=self.vocab_size,
        ) if use_lzp else None

        self.adaptive_head = AdaptiveHead(
            vocab_size=self.vocab_size, lr=adaptive_lr,
        ) if use_adaptive_head else None

        # Context mixer: combines LLM + active secondary models
        num_mix_models = 1  # LLM always present
        if use_ngram:
            num_mix_models += 1
        if use_lzp:
            num_mix_models += 1
        self.mixer = ContextMixer(
            num_models=num_mix_models, lr=mixer_lr,
            vocab_size=self.vocab_size,
        ) if num_mix_models > 1 else None

        # Pre-allocated buffers to avoid per-token numpy temporaries.
        # These eliminate ~5 MB of malloc/free per token across 8 workers.
        self._entropy_buf = np.zeros(self.vocab_size, dtype=np.float64)
        self._temp_buf = np.zeros(self.vocab_size, dtype=np.float64)
        self._cdf_converter = CdfConverter(self.vocab_size)

    def _config_flags(self) -> int:
        """Encode active features as a bitmask."""
        flags = 0
        if self.use_ngram:
            flags |= FLAG_NGRAM
        if self.use_lzp:
            flags |= FLAG_LZP
        if self.use_adaptive_head:
            flags |= FLAG_ADAPTIVE_HEAD
        if self.use_confidence_skip:
            flags |= FLAG_CONFIDENCE_SKIP
        return flags

    def _reset_secondary_models(self):
        """Reset all secondary models for a new sequence."""
        if self.ngram:
            self.ngram.reset()
        if self.lzp:
            self.lzp.reset()
        if self.mixer:
            self.mixer.reset()
        if self.adaptive_head:
            self.adaptive_head.reset()

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Sharpen or soften model probabilities via temperature scaling.

        Uses pre-allocated buffer to avoid ~1.5 MB of temporaries per call.
        """
        if self.temperature == 1.0:
            return probs
        buf = self._temp_buf
        np.add(probs, 1e-10, out=buf)
        np.log(buf, out=buf)
        buf /= self.temperature
        buf -= buf.max()
        np.exp(buf, out=buf)
        buf /= buf.sum()
        return buf

    def _get_probs(
        self, context: list[int], token_index: int,
    ) -> tuple[np.ndarray, bool, "list[np.ndarray] | None"]:
        """Compute blended prediction for the next token.

        All secondary models and mixing operate on numpy arrays.
        The LLM's torch tensor is converted to numpy at the boundary.

        Args:
            context: Token IDs seen so far.
            token_index: Position in the sequence (for warmup check).

        Returns:
            (final_probs, skipped_llm, model_probs_for_mixer_update)
            All probability arrays are numpy float64.
        """
        in_warmup = (token_index < self.warmup)

        # Secondary model predictions (always computed for learning)
        ngram_probs = self.ngram.predict(context) if self.ngram else None
        lzp_probs = self.lzp.predict(context) if self.lzp else None

        # During warmup: LLM only, no mixing, no skip
        if in_warmup:
            llm_probs = self.model.get_probs(context).numpy()
            llm_probs = self._apply_temperature(llm_probs)
            return llm_probs, False, None

        # Confidence-based LLM skip (post-warmup only)
        skip_llm = False
        if self.use_confidence_skip and ngram_probs is not None:
            ent = _entropy(ngram_probs, self._entropy_buf)
            skip_llm = (ent < self.skip_threshold)

        if skip_llm:
            if ngram_probs is not None and lzp_probs is not None:
                probs = (SKIP_NGRAM_WEIGHT * ngram_probs
                         + SKIP_LZP_WEIGHT * lzp_probs)
            elif ngram_probs is not None:
                probs = ngram_probs
            else:
                probs = np.full(
                    self.vocab_size, 1.0 / self.vocab_size,
                    dtype=np.float64,
                )
            return probs, True, None

        # LLM prediction (torch → numpy at boundary)
        llm_probs = self.model.get_probs(context).numpy()
        llm_probs = self._apply_temperature(llm_probs)

        if self.adaptive_head:
            llm_probs = self.adaptive_head.adjust(llm_probs)

        # Mixing
        if self.mixer is not None:
            model_probs = [llm_probs]
            if ngram_probs is not None:
                model_probs.append(ngram_probs)
            if lzp_probs is not None:
                model_probs.append(lzp_probs)
            probs = self.mixer.mix(model_probs)
            return probs, False, model_probs

        return llm_probs, False, None

    def _update_models(
        self,
        context: list[int],
        actual_token: int,
        skipped_llm: bool,
        model_probs: "list[np.ndarray] | None",
        llm_adjusted_probs: "np.ndarray | None",
    ):
        """Update all models after observing a token."""
        if self.ngram:
            self.ngram.update(context, actual_token)
        if self.lzp:
            self.lzp.update(context, actual_token)

        if not skipped_llm:
            if self.mixer and model_probs is not None:
                self.mixer.update(actual_token, model_probs)
            if self.adaptive_head and llm_adjusted_probs is not None:
                self.adaptive_head.update(actual_token, llm_adjusted_probs)

    # ------------------------------------------------------------------
    # Text stream compression (used by parallel workers)
    # ------------------------------------------------------------------

    def _compress_text_to_stream(
        self, text: str, *,
        bytes_done: int = 0, bytes_total: int = 0, chunk_size: int = 0,
    ) -> tuple[int, int, bytes]:
        """Arithmetic-code a text string using the ensemble.

        Returns:
            (token_count, bit_count, stream_bytes)
        """
        token_ids = self.model.tokenizer.encode(text)
        num_tokens = len(token_ids)

        if self.verbose:
            print(f"Tokens: {num_tokens}", file=sys.stderr)

        keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK

        encoder = ArithmeticEncoder()
        context: list[int] = []
        skipped_count = 0
        self._progress_total = num_tokens
        self._progress = 0

        # Disable cyclic GC during the hot loop. The N-gram/LZP tables
        # create millions of small dicts that are never cyclic (int→int).
        # Without this, Python's GC periodically scans ALL tracked objects,
        # causing growing pauses as table size increases.
        gc.disable()

        for i, token_id in enumerate(token_ids):
            self._progress = i
            if self.verbose and (i % 500 == 0 or i == num_tokens - 1):
                line = (
                    f"\rEncoding: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)"
                )
                if bytes_total > 0:
                    frac = (i + 1) / num_tokens if num_tokens else 1
                    overall = (bytes_done + chunk_size * frac) / bytes_total
                    line += f"  [total: {100*overall:.1f}%]"
                if self.use_confidence_skip:
                    line += f"  [skipped: {skipped_count}]"
                print(line, end="", file=sys.stderr)

            probs, skipped_llm, model_probs = self._get_probs(context, i)
            if skipped_llm:
                skipped_count += 1

            # Extract LLM adjusted probs for adaptive head update
            llm_adjusted = None
            if not skipped_llm and model_probs is not None:
                llm_adjusted = model_probs[0]  # first model is always LLM
            elif not skipped_llm and self.adaptive_head:
                llm_adjusted = probs  # probs IS the adjusted LLM output

            # Encode (zero-alloc CDF conversion)
            cdf = self._cdf_converter.convert(probs, CDF_TOTAL)
            encoder.encode_symbol(cdf, token_id)

            # Update models
            self._update_models(
                context, token_id, skipped_llm, model_probs, llm_adjusted,
            )

            # Maintain context window
            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                context = context[-keep:]

        gc.enable()

        if self.verbose:
            print(file=sys.stderr)
            warmup_used = min(self.warmup, num_tokens)
            if warmup_used > 0 and self.mixer:
                print(
                    f"Warmup: {warmup_used} tokens (LLM only)",
                    file=sys.stderr,
                )
            if self.use_confidence_skip:
                pct = 100 * skipped_count / num_tokens if num_tokens else 0
                print(
                    f"LLM skipped: {skipped_count}/{num_tokens} "
                    f"({pct:.1f}%)",
                    file=sys.stderr,
                )
            if self.mixer:
                print(
                    f"Final mixer weights: "
                    f"{[f'{w:.3f}' for w in self.mixer.get_weights()]}",
                    file=sys.stderr,
                )

        compressed_bits = encoder.get_bit_count()
        stream = encoder.finish()
        return num_tokens, compressed_bits, stream

    def _decompress_text_stream(self, stream: bytes, num_tokens: int) -> str:
        """Decode an arithmetic-coded stream back to text."""
        decoder = ArithmeticDecoder(stream)
        context: list[int] = []
        token_ids: list[int] = []
        self._progress_total = num_tokens
        self._progress = 0

        gc.disable()

        for i in range(num_tokens):
            self._progress = i
            if self.verbose and (i % 100 == 0 or i == num_tokens - 1):
                print(
                    f"\rDecompressing: {i+1}/{num_tokens} "
                    f"({100*(i+1)/num_tokens:.1f}%)",
                    end="", file=sys.stderr,
                )

            probs, skipped_llm, model_probs = self._get_probs(context, i)

            llm_adjusted = None
            if not skipped_llm and model_probs is not None:
                llm_adjusted = model_probs[0]
            elif not skipped_llm and self.adaptive_head:
                llm_adjusted = probs

            cdf = self._cdf_converter.convert(probs, CDF_TOTAL)
            token_id = decoder.decode_symbol(cdf)
            token_ids.append(token_id)

            self._update_models(
                context, token_id, skipped_llm, model_probs, llm_adjusted,
            )

            context.append(token_id)
            if len(context) > self.model.MAX_CONTEXT:
                keep = self.model.MAX_CONTEXT - self.model.SLIDE_CHUNK
                context = context[-keep:]

        gc.enable()

        if self.verbose:
            print(file=sys.stderr)

        return self.model.tokenizer.decode(token_ids)

    def _apply_flags(self, flags: int):
        """Configure features from stored flags (for decompression)."""
        want_ngram = bool(flags & FLAG_NGRAM)
        want_lzp = bool(flags & FLAG_LZP)
        want_adaptive = bool(flags & FLAG_ADAPTIVE_HEAD)
        want_skip = bool(flags & FLAG_CONFIDENCE_SKIP)

        if want_ngram and self.ngram is None:
            self.ngram = NgramModel(
                max_order=DEFAULT_NGRAM_ORDER, vocab_size=self.vocab_size,
            )
        self.use_ngram = want_ngram

        if want_lzp and self.lzp is None:
            self.lzp = LZPModel(
                max_order=DEFAULT_LZP_MAX_ORDER,
                min_order=DEFAULT_LZP_MIN_ORDER,
                vocab_size=self.vocab_size,
            )
        self.use_lzp = want_lzp

        if want_adaptive and self.adaptive_head is None:
            self.adaptive_head = AdaptiveHead(
                vocab_size=self.vocab_size, lr=DEFAULT_ADAPTIVE_LR,
            )
        self.use_adaptive_head = want_adaptive

        self.use_confidence_skip = want_skip and self.use_ngram

        # Rebuild mixer for the correct number of models.
        num_mix = 1
        if self.use_ngram:
            num_mix += 1
        if self.use_lzp:
            num_mix += 1
        self.mixer = ContextMixer(
            num_models=num_mix, lr=DEFAULT_MIXER_LR,
        ) if num_mix > 1 else None

