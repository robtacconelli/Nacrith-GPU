"""
Model wrapper for SmolLM2-135M language model.

Primary backend: llama.cpp (via llama-cpp-python) for fast inference
by eliminating Python/PyTorch dispatch overhead (~7x faster than
PyTorch + CUDA Graphs for single-token incremental decode).

Fallback: PyTorch with StaticCache + CUDA Graphs on CUDA,
DynamicCache on CPU.  Used automatically when llama-cpp-python
is not installed or no GGUF model file is found.
"""

import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    _HAS_LLAMA_CPP = False


class ModelWrapper:
    """Wraps SmolLM2-135M for next-token probability prediction.

    Primary backend: llama.cpp — ~7x faster than PyTorch for
    single-token incremental decode on CUDA.

    Fallback: PyTorch with StaticCache + CUDA Graphs (CUDA) or
    DynamicCache (CPU).
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

    # PyTorch CUDA graph settings (fallback only)
    _GRAPH_WARMUP = 3

    def __init__(self, model_name: str = None, gguf_path: str = None,
                 verbose: bool = True):
        self.model_name = model_name or self.MODEL_NAME
        self.verbose = verbose
        self._cache_len = 0

        # Try llama.cpp first, fall back to PyTorch
        gguf = gguf_path or self._find_gguf()
        if _HAS_LLAMA_CPP and gguf:
            self._backend = "llama.cpp"
            self._init_llama_cpp(gguf)
        else:
            self._backend = "pytorch"
            self._init_pytorch()

        if self.verbose:
            print(
                f"Backend: {self._backend}, device: {self.device}, "
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
    # llama.cpp backend
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
        # Use HuggingFace tokenizer for encode/decode — llama.cpp's
        # detokenize drops content for 47 long whitespace/repeat tokens.
        # Token IDs are identical between both tokenizers.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._llm          # tests check `model is not None`
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._n_tokens = 0

    # ------------------------------------------------------------------
    # PyTorch backend (fallback)
    # ------------------------------------------------------------------

    def _init_pytorch(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

        self.device = self._select_device()
        if self.verbose:
            print(f"Using device: {self.device}", file=sys.stderr)
            print(f"Loading model {self.model_name}...", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

        self._use_cuda_graph = (self.device == "cuda")
        if self._use_cuda_graph:
            self._setup_cuda_graph()
        else:
            self._past_key_values = None

        if self.verbose:
            print(
                f"Model loaded: vocab_size={self.vocab_size}, "
                f"device={self.device}",
                file=sys.stderr,
            )

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            try:
                t = torch.tensor([1.0], device="cuda")
                del t
                return "cuda"
            except Exception:
                pass
        return "cpu"

    def _setup_cuda_graph(self):
        from transformers import StaticCache

        self._static_cache = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=self.MAX_CONTEXT,
            device=self.device,
            dtype=torch.float32,
        )
        self._graph = None
        self._graph_input = torch.zeros(
            1, 1, dtype=torch.long, device=self.device,
        )
        self._graph_position = torch.zeros(
            1, dtype=torch.long, device=self.device,
        )
        self._graph_output = None

        if self.verbose:
            print("CUDA graph acceleration enabled", file=sys.stderr)

    def _capture_graph(self):
        self._graph_position.fill_(self._cache_len)
        self._graph_input.fill_(0)

        for _ in range(self._GRAPH_WARMUP):
            self.model(
                self._graph_input,
                past_key_values=self._static_cache,
                cache_position=self._graph_position,
                use_cache=True,
            )
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_output = self.model(
                self._graph_input,
                past_key_values=self._static_cache,
                cache_position=self._graph_position,
                use_cache=True,
            )
        torch.cuda.synchronize()

        if self.verbose:
            print("CUDA graph captured", file=sys.stderr)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self):
        """Clear the KV-cache. Call when starting a new sequence."""
        self._cache_len = 0

        if self._backend == "llama.cpp":
            self._llm.reset()
            self._n_tokens = 0
        elif self._use_cuda_graph:
            self._static_cache.reset()
            self._graph = None
        else:
            self._past_key_values = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _slide_kv_cache(self, keep: int):
        """Shift llama.cpp KV cache: drop oldest tokens, shift positions.

        Instead of reset() + eval(all_kept_tokens), this removes the
        oldest SLIDE_CHUNK tokens from the KV cache and shifts the
        remaining positions down.  The last position is left for
        _forward_llama_cpp to re-evaluate incrementally (1 token eval
        instead of re-processing all *keep* tokens from scratch).
        """
        drop = self._n_tokens - keep
        if drop <= 0:
            return
        # Remove positions [0, drop) from KV cache
        self._llm._ctx.kv_cache_seq_rm(0, 0, drop)
        # Shift remaining positions [drop, n_tokens) down by -drop
        self._llm._ctx.kv_cache_seq_shift(0, drop, -1, -drop)
        # Set to keep-1 so _forward_llama_cpp sees ctx_len == _n_tokens+1,
        # triggering incremental eval of just the last token.  Also sync
        # llama.cpp's internal counter so eval() places it at the right pos.
        self._n_tokens = keep - 1
        self._cache_len = keep - 1
        self._llm.n_tokens = keep - 1

    # ------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_probs(self, token_ids: list[int]) -> torch.Tensor:
        """Get next-token probability distribution given context.

        Uses KV-cache for incremental inference. On the first call (or
        after reset_cache), processes the full context. On subsequent
        calls where only one token was appended, processes just that
        single new token using the cached states.

        Args:
            token_ids: List of token IDs as context. Can be empty,
                       in which case a BOS/default context is used.

        Returns:
            Tensor of shape (vocab_size,) with probabilities summing to 1.
        """
        if len(token_ids) == 0:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            token_ids = [bos if bos is not None else 0]

        if len(token_ids) > self.MAX_CONTEXT:
            keep = self.MAX_CONTEXT - self.SLIDE_CHUNK
            token_ids = token_ids[-keep:]
            if self._backend == "llama.cpp":
                self._slide_kv_cache(keep)
            else:
                self.reset_cache()

        ctx_len = len(token_ids)

        if self._backend == "llama.cpp":
            logits = self._forward_llama_cpp(token_ids, ctx_len)
        elif self._use_cuda_graph:
            logits = self._forward_cuda_graph(token_ids, ctx_len)
        else:
            logits = self._forward_eager_cpu(token_ids, ctx_len)

        self._cache_len = ctx_len
        probs = torch.softmax(logits, dim=0)
        return probs.cpu()

    # ------------------------------------------------------------------
    # llama.cpp forward
    # ------------------------------------------------------------------

    def _forward_llama_cpp(
        self, token_ids: list[int], ctx_len: int,
    ) -> torch.Tensor:
        """llama.cpp forward pass with incremental KV-cache.

        Three cases:
        1. ctx_len == _n_tokens + 1  → append one token (common path)
        2. 0 < ctx_len <= _n_tokens  → context was trimmed from the front
           (sliding window); shift KV cache and eval just the last token
        3. otherwise                 → cold start; reset and eval everything
        """
        if self._n_tokens > 0 and ctx_len == self._n_tokens + 1:
            # Case 1: incremental — eval just the new token
            self._llm.eval([token_ids[-1]])
        elif self._n_tokens > 0 and 0 < ctx_len <= self._n_tokens:
            # Case 2: context trimmed from front (sliding window).
            # The caller dropped tokens from the beginning of the context.
            # Shift the KV cache instead of re-processing everything.
            drop = self._n_tokens - (ctx_len - 1)
            self._llm._ctx.kv_cache_seq_rm(0, 0, drop)
            self._llm._ctx.kv_cache_seq_shift(0, drop, -1, -drop)
            self._llm.n_tokens = ctx_len - 1
            self._n_tokens = ctx_len - 1
            # Now eval just the last token (the one appended after the trim)
            self._llm.eval([token_ids[-1]])
        else:
            # Case 3: cold start — reset and eval everything
            self._llm.reset()
            self._llm.eval(token_ids)

        self._n_tokens = ctx_len
        logits = self._llm.scores[ctx_len - 1]
        return torch.from_numpy(logits.copy())

    # ------------------------------------------------------------------
    # PyTorch CUDA Graph forward (fallback)
    # ------------------------------------------------------------------

    def _forward_cuda_graph(
        self, token_ids: list[int], ctx_len: int,
    ) -> torch.Tensor:
        if self._cache_len > 0 and ctx_len == self._cache_len + 1:
            if self._graph is None:
                self._capture_graph()
            self._graph_input.fill_(token_ids[-1])
            self._graph_position.fill_(ctx_len - 1)
            self._graph.replay()
            return self._graph_output.logits[0, -1, :]
        else:
            self._static_cache.reset()
            self._graph = None
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device,
            )
            cache_position = torch.arange(
                ctx_len, device=self.device, dtype=torch.long,
            )
            outputs = self.model(
                input_ids,
                past_key_values=self._static_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            return outputs.logits[0, -1, :]

    # ------------------------------------------------------------------
    # PyTorch CPU eager forward (fallback)
    # ------------------------------------------------------------------

    def _forward_eager_cpu(
        self, token_ids: list[int], ctx_len: int,
    ) -> torch.Tensor:
        if (self._past_key_values is not None
                and ctx_len == self._cache_len + 1):
            new_ids = torch.tensor(
                [[token_ids[-1]]], dtype=torch.long, device=self.device,
            )
            outputs = self.model(
                new_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        else:
            self._past_key_values = None
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device,
            )
            outputs = self.model(
                input_ids,
                use_cache=True,
            )

        self._past_key_values = outputs.past_key_values
        return outputs.logits[0, -1, :]

    # ------------------------------------------------------------------
    # Batch forward (stateless)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_window(self, token_ids: list[int]) -> torch.Tensor:
        """Run a single forward pass on a token window.

        Returns softmax probabilities for ALL positions in the window.
        Does NOT use or update the KV-cache — purely stateless.

        Note: on llama.cpp backend this invalidates the incremental
        cache. Call reset_cache() before resuming get_probs().

        Args:
            token_ids: Up to MAX_CONTEXT token IDs.

        Returns:
            Tensor of shape (len(token_ids), vocab_size) on CPU.
            result[j] = P(next_token | token_ids[0], ..., token_ids[j]).
        """
        if self._backend == "llama.cpp":
            self._llm.reset()
            self._llm.eval(token_ids)
            logits = self._llm.scores[:len(token_ids)].copy()
            self._n_tokens = 0
            self._cache_len = 0
            self._llm.reset()
            return torch.softmax(torch.from_numpy(logits), dim=-1)

        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device,
        )
        outputs = self.model(input_ids, use_cache=False)
        return torch.softmax(outputs.logits[0], dim=-1).cpu()
