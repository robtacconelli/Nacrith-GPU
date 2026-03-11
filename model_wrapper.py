"""
Model wrapper for SmolLM2-135M language model using llama.cpp.

Loads a GGUF model from the local directory and the tokenizer from
the local tokenizer/ subdirectory.  Requires llama-cpp-python and
the tokenizers library (HuggingFace Rust tokenizer, no torch needed).
"""

import json
import os
import sys

import numpy as np
from llama_cpp import Llama
from tokenizers import Tokenizer as _HFTokenizer


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _softmax_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class _LocalTokenizer:
    """Thin wrapper around tokenizers.Tokenizer loaded from a local directory."""

    def __init__(self, tokenizer_dir: str):
        tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
        cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        self._tok = _HFTokenizer.from_file(tok_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        bos = cfg.get("bos_token")
        self.bos_token_id = self._tok.token_to_id(bos) if bos else None

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class ModelWrapper:
    """Wraps SmolLM2-135M for next-token probability prediction.

    Uses llama.cpp (via llama-cpp-python) for fast single-token
    incremental decode with KV-cache.  Requires a local GGUF file
    and a local tokenizer/ directory.
    """

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    MAX_CONTEXT = 2048
    # When context exceeds MAX_CONTEXT, drop this many old tokens at once.
    SLIDE_CHUNK = 512

    _GGUF_NAMES = [
        "smollm2-135m-f32.gguf",
        "smollm2-135m-f16.gguf",
        "smollm2-135m.gguf",
    ]

    def __init__(self, model_name: str = None, gguf_path: str = None,
                 verbose: bool = True):
        self.model_name = model_name or self.MODEL_NAME
        self.verbose = verbose
        self._cache_len = 0
        self.device = "cpu"

        gguf = gguf_path or self._find_gguf()
        if gguf is None:
            raise FileNotFoundError(
                "No GGUF model found. Place a .gguf file next to this script."
            )
        self._init_llama_cpp(gguf)

        if self.verbose:
            print(
                f"Backend: llama.cpp, device: {self.device}, "
                f"vocab_size: {self.vocab_size}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # GGUF discovery
    # ------------------------------------------------------------------

    def _find_gguf(self) -> str | None:
        """Search for a GGUF model file next to this script."""
        base = os.path.dirname(os.path.abspath(__file__))
        for name in self._GGUF_NAMES:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
        return None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_llama_cpp(self, gguf_path: str):
        if self.verbose:
            print(f"Loading GGUF: {gguf_path}", file=sys.stderr)

        self._llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=self.MAX_CONTEXT,
            seed=42,
            logits_all=True,
            verbose=self.verbose,
        )

        self.vocab_size = self._llm.n_vocab()

        # Load tokenizer from local tokenizer/ directory.
        # Using the tokenizers Rust library directly avoids the 47-token
        # detokenize bug present in llama.cpp's built-in tokenizer.
        base = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(base, "tokenizer")
        self.tokenizer = _LocalTokenizer(tokenizer_dir)

        self.model = self._llm   # tests check `model is not None`
        self._n_tokens = 0
        self._n_batch = self._llm.n_batch
        # Partial cold-start bookkeeping: track the end of the last
        # sequence of full n_batch-sized batches from the most recent
        # cold start.  KV entries before this position are bit-identical
        # to what any future cold start would produce (causal attention
        # never modifies past KV entries), so they can be reused.
        self._cold_batch_end = 0
        self._cold_valid = False

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self):
        """Clear the KV-cache. Call when starting a new sequence."""
        self._cache_len = 0
        self._llm.reset()
        self._n_tokens = 0
        self._cold_batch_end = 0
        self._cold_valid = False

    def _slide_kv_cache(self, keep: int):
        """Shift llama.cpp KV cache: drop oldest tokens, shift positions.

        Instead of reset() + eval(all_kept_tokens), removes the oldest
        SLIDE_CHUNK tokens from the KV cache and shifts remaining positions
        down.  The last position is left for _forward_llama_cpp to
        re-evaluate incrementally.
        """
        drop = self._n_tokens - keep
        if drop <= 0:
            return
        self._llm._ctx.kv_cache_seq_rm(0, 0, drop)
        self._llm._ctx.kv_cache_seq_shift(0, drop, -1, -drop)
        self._n_tokens = keep - 1
        self._cache_len = keep - 1
        self._llm.n_tokens = keep - 1
        self._cold_valid = False

    # ------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------

    def get_probs(self, token_ids: list[int]) -> np.ndarray:
        """Get next-token probability distribution given context.

        Uses KV-cache for incremental inference.  On the first call (or
        after reset_cache), processes the full context.  On subsequent
        calls where only one token was appended, processes just that
        single new token using the cached states.

        Args:
            token_ids: List of token IDs as context.  Can be empty,
                       in which case a BOS/default context is used.

        Returns:
            numpy array of shape (vocab_size,) with probabilities summing to 1.
        """
        if len(token_ids) == 0:
            bos = self.tokenizer.bos_token_id
            token_ids = [bos if bos is not None else 0]

        if len(token_ids) > self.MAX_CONTEXT:
            keep = self.MAX_CONTEXT - self.SLIDE_CHUNK
            token_ids = token_ids[-keep:]
            self._slide_kv_cache(keep)

        ctx_len = len(token_ids)
        logits = self._forward_llama_cpp(token_ids, ctx_len)
        self._cache_len = ctx_len
        return _softmax(logits)

    def _forward_llama_cpp(
        self, token_ids: list[int], ctx_len: int,
    ) -> np.ndarray:
        """llama.cpp forward pass with incremental KV-cache.

        Three cases:
        1. ctx_len == _n_tokens + 1  → append one token (common path)
        2. 0 < ctx_len <= _n_tokens  → context was trimmed from the front
           (sliding window); shift KV cache and eval just the last token
        3. otherwise                 → cold start (full or partial)
        """
        if self._n_tokens > 0 and ctx_len == self._n_tokens + 1:
            self._llm.eval([token_ids[-1]])
        elif self._n_tokens > 0 and 0 < ctx_len <= self._n_tokens:
            drop = self._n_tokens - (ctx_len - 1)
            self._llm._ctx.kv_cache_seq_rm(0, 0, drop)
            self._llm._ctx.kv_cache_seq_shift(0, drop, -1, -drop)
            self._llm.n_tokens = ctx_len - 1
            self._n_tokens = ctx_len - 1
            self._llm.eval([token_ids[-1]])
            self._cold_valid = False
        else:
            # Cold start — reuse KV entries from previous full batches
            # when possible.  KV entries produced by full n_batch-sized
            # eval() calls are bit-identical across cold starts (same
            # tokens, same batch size, causal attention).
            reuse = (self._cold_batch_end
                     if self._cold_valid and self._cold_batch_end > 0
                     else 0)
            if reuse > 0 and reuse < ctx_len:
                # Partial cold start: keep KV 0..reuse-1, reprocess
                # from reuse onward with the same batch alignment as
                # a full cold start.
                self._llm.n_tokens = reuse
                self._llm.eval(token_ids[reuse:])
            else:
                self._llm.reset()
                self._llm.eval(token_ids)
            self._cold_batch_end = (ctx_len // self._n_batch) * self._n_batch
            self._cold_valid = True

        self._n_tokens = ctx_len
        return self._llm.scores[ctx_len - 1].copy()

    # ------------------------------------------------------------------
    # Batch forward (stateless)
    # ------------------------------------------------------------------

    def forward_window(self, token_ids: list[int]) -> np.ndarray:
        """Run a single forward pass on a token window.

        Returns softmax probabilities for ALL positions in the window.
        Does NOT use or update the KV-cache — purely stateless.

        Note: invalidates the incremental cache.  Call reset_cache()
        before resuming get_probs().

        Args:
            token_ids: Up to MAX_CONTEXT token IDs.

        Returns:
            numpy array of shape (len(token_ids), vocab_size).
            result[j] = P(next_token | token_ids[0], ..., token_ids[j]).
        """
        self._llm.reset()
        self._llm.eval(token_ids)
        logits = self._llm.scores[:len(token_ids)].copy()
        self._n_tokens = 0
        self._cache_len = 0
        self._llm.reset()
        return _softmax_2d(logits)
