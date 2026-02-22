<p align="center">
  <img src="assets/banner_gpu.png" alt="Nacrith">
</p>

<p align="center">
  <a href="https://nacrith.com">Website</a> · <a href="#">Technical Paper (PDF) <i>Soon</i></a> · <a href="https://huggingface.co/spaces/robtacconelli/Nacrith-GPU">Try on Hugging Face</a> · <a href="https://github.com/st4ck/Nacrith-CPU">CPU Version</a>
</p>

### Nacrith GPU - Neural Arithmetic Compression — A Game-Changer in Text Encoding

**Nacrith GPU** is a **lossless** text compression system that delivers exceptional results by combining the predictive power of a neural language model with the mathematical precision of arithmetic coding. Where traditional compressors see bytes, Nacrith *understands language* — achieving compression ratios **far below the classical Shannon entropy limits** and **3-4x better than gzip, xz, and zip**.

At its core, Nacrith GPU pairs a small but capable LLM ([SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), 135M parameters) with an arithmetic encoder. The LLM reads text token by token and, at each step, predicts *how likely every possible next token is*. These probability predictions are fed directly into the arithmetic coder, which assigns shorter bit sequences to likely tokens and longer ones to surprises. Because the LLM captures grammar, semantics, and world knowledge — not just local byte patterns — it predicts with far higher confidence than dictionary-based methods, resulting in dramatically fewer bits per token. Both compressor and decompressor run the **exact same model**, so predictions are identical on both sides, guaranteeing **perfect lossless reconstruction**.

---

## What's New in This Version

### Parallel Compression (NC05/NC06)

The compressor now splits work across **multiple GPU worker threads**, each running its own llama.cpp model instance. This delivers near-linear speedup with worker count:

- **Automatic worker detection**: queries VRAM via `nvidia-smi` and calculates the optimal number of instances (first instance ~1.2 GB, each additional ~660 MB)
- **Up to 8 workers** (configurable with `--workers`)
- Text is split at newline boundaries into equal chunks, one per worker
- Binary files (NC06) parallelize all embedded text chunks across workers
- Each worker maintains independent state (KV cache, secondary models, mixer) — no contention

### llama.cpp Inference Backend

Switched from PyTorch to **llama.cpp** (via `llama-cpp-python`) as the primary inference backend:

- **~7x faster** than PyTorch + CUDA Graphs for single-token incremental decode
- Eliminates Python/PyTorch dispatch overhead — all GPU work happens in C/C++
- Requires a GGUF-format model file (e.g., `smollm2-135m-f32.gguf`)
- Falls back to PyTorch automatically if llama-cpp-python is not installed or no GGUF file is found

### Dual Tokenizer Architecture

- **llama.cpp** handles encoding (inference) on the GPU
- **HuggingFace tokenizer** handles text tokenization and detokenization — llama.cpp's built-in detokenizer drops content for 47 long whitespace/repeat tokens, while HuggingFace handles them correctly
- Token IDs are identical between both tokenizers, so they can be used interchangeably

### Context Mixing Ensemble

Four components work together to maximize prediction accuracy:

| Component | Purpose |
|-----------|---------|
| **N-gram model** (order 1-4) | Fast local pattern prediction with interpolated backoff |
| **Context mixer** | Adaptive linear blending — models that predict well gain weight |
| **Adaptive head** | Online bias correction in log-probability space |
| **Confidence skip** | Bypasses LLM when N-gram entropy is low enough |

All features are stored as flags in the file header, so the decompressor auto-configures itself.

### KV Cache Sliding Window

When the 2048-token context window fills up, the compressor now uses **native KV cache manipulation** instead of resetting and re-processing:

- `kv_cache_seq_rm` removes the oldest tokens from the cache
- `kv_cache_seq_shift` shifts remaining positions down
- Only the last token is re-evaluated (1 token eval instead of 1536)
- **37x faster** per context slide (~19ms vs ~693ms on GTX 1050 Ti)

### Performance Optimizations

- **Integer context hashing**: N-gram model uses a 64-bit rolling hash instead of tuple keys, eliminating ~54M tuple allocations per worker and reducing GC pressure
- **Inner dict cap**: N-gram continuation dicts are capped at 64 entries per context, preventing progressive slowdown as common contexts accumulate hundreds of unique continuations
- **Enhanced CDF precision**: 2^24 instead of 2^16, reducing per-token overhead from ~2 bits to ~0.004 bits

### Format Cleanup

Old formats (NC01-NC04) have been removed. The compressor now uses only:

- **NC05** — parallel text compression
- **NC06** — parallel hybrid binary/text compression

---

## Benchmark Results

Tested on English prose of varying sizes. GPU: NVIDIA GTX 1050 Ti.

| Sample | Original | gzip | xz | zip | Nacrith GPU |
|--------|----------|------|----|-----|---------|
| small | 3.0 KB | 1.5 KB (49.3%) | 1.6 KB (52.1%) | 1.6 KB (54.0%) | **263 B (8.6%)** |
| medium | 50.0 KB | 18.9 KB (37.8%) | 17.6 KB (35.3%) | 19.1 KB (38.1%) | **4.0 KB (7.9%)** |
| large | 100.5 KB | 39.0 KB (38.8%) | 35.5 KB (35.3%) | 39.2 KB (39.0%) | **9.6 KB (9.5%)** |
| alice29 | 148.5 KB | 52.9 KB (35.6%) | 47.4 KB (31.9%) | 53.1 KB (35.7%) | **17.0 KB (11.5%)** |
| asyoulik | 122.2 KB | 48.8 KB (39.9%) | 45.4 KB (37.1%) | 49.0 KB (40.1%) | **19.9 KB (16.3%)** |
| **enwik8** | **95.4 MB** | **34.9 MB (36.5%)** | **26.4 MB (27.7%)** | **34.8 MB (36.4%)** | **11.2 MB (11.7%)** |

*Percentages = compressed / original (lower is better).*

### Comparison with Neural & State-of-the-Art Compressors

On standard benchmarks, Nacrith GPU outperforms all compared neural and statistical compressors:

| System | Model | alice29.txt (bpb) | enwik8 (bpb) |
|--------|-------|-------------------|--------------|
| gzip -9 | — | 2.851 | 2.916 |
| CMIX v21 | LSTM + 2,000+ models | 1.635 | 1.17 |
| NNCP v3 | Transformer-XL (online) | 3.96 | ~1.19 |
| PAQ8px -8L | Context mixing | 1.728 | ~1.27 |
| ts_zip | RWKV-169M | ~1.142 | ~1.11 |
| FineZip | LLaMA-3-8B (fine-tuned) | — | 1.024 |
| **Nacrith GPU** | **SmolLM2-135M** | **0.918** | **0.9389** |

### alice29.txt — All Compressors (bits per byte)

![alice29.txt Compression Comparison](assets/alice29_comparison.png)

### enwik8 — All Compressors (bits per byte)

![enwik8 Compression Comparison](assets/enwik8_comparison.png)

### Compression Ratio (% of original size)

![Compression Ratio Comparison](assets/compression_ratio.png)

### Compressed Size

![Compressed Size Comparison](assets/compressed_size.png)

### Space Savings

![Space Savings](assets/space_savings.png)

### Key Observations

- **Nacrith GPU achieves ~8-12% ratio** on English text — roughly **3.1x better than gzip** and **2.5x better than bzip2**, even at the 100KB scale.
- On **enwik8** (100 MB Wikipedia), Nacrith GPU achieves **0.9389 bpb** — the best result among all compared systems — outperforming ts_zip (~1.11 bpb) by 15%, FineZip (1.024 bpb) by 8% with a 60x smaller model and no fine-tuning, and CMIX v21 (1.17 bpb) by 20%.
- On **alice29.txt** (Canterbury Corpus), Nacrith GPU achieves **0.918 bpb** — 44% better than CMIX v21 and 20% better than ts_zip.
- Nacrith GPU saves **88-92% of space** consistently across all tested sizes, while gzip saves ~50-64% and xz saves ~48-68%.
- Trade-off: compression speed is slower than traditional compressors since each token requires a neural network forward pass. Parallel compression with multiple workers substantially improves throughput.
- The model uses ~1.2 GB of VRAM for the first instance, plus ~660 MB per additional worker. Any CUDA-capable GPU with at least 2 GB of VRAM will work. Falls back to CPU if no GPU is available.
- All results are **fully lossless** — decompressed output matches the original byte-for-byte.

### Beyond the Shannon Entropy Limit

Nacrith GPU doesn't just beat traditional compressors — it compresses **well below the classical Shannon entropy bounds** of the source data. On the 100KB benchmark file:

| Method | Size | bits/byte |
|--------|------|-----------|
| Original | 100.5 KB | 8.0000 |
| Shannon 0th-order limit | 59.5 KB | 4.7398 |
| Shannon 1st-order limit | 44.2 KB | 3.5213 |
| Shannon 2nd-order limit | 34.4 KB | 2.7373 |
| gzip -9 | 39.0 KB | 3.1065 |
| xz -9 | 35.5 KB | 2.8263 |
| **Nacrith GPU** | **9.6 KB** | **0.7635** |

Nacrith GPU achieves **0.76 bits/byte** — **84% below** the 0th-order Shannon limit, **78% below** the 1st-order (bigram) limit, and **72% below** even the 2nd-order (trigram) limit. This is the best compression performance among evaluated systems: it proves the neural model captures deep linguistic structure — grammar, semantics, long-range context — that no frequency-based or dictionary-based method can exploit. For comparison, gzip and xz both hover *above* the 2nd-order Shannon limit, unable to break through the statistical ceiling that Nacrith GPU shatters.

---

## How It Works

Nacrith GPU exploits the deep connection between **prediction** and **compression** (Shannon, 1948): a good predictor of text can be turned into a good compressor.

### The LLM + Arithmetic Coding Pipeline

**The LLM** (SmolLM2-135M) is a transformer neural network trained on large amounts of text. Given a sequence of tokens, it outputs a probability distribution over the entire vocabulary (~49K tokens) for what comes next. It captures grammar, common phrases, semantic relationships, and world knowledge — far beyond simple byte-pattern matching.

**Arithmetic coding** is a mathematically optimal encoding scheme that maps a sequence of symbols to a single number in the interval `[0, 1)`. For each symbol, it narrows the interval proportionally to that symbol's probability. High-probability symbols barely shrink the interval (costing almost zero bits), while unlikely symbols shrink it a lot (costing many bits). The final interval width determines the total compressed size.

**Together**: the LLM provides the probabilities, the arithmetic coder turns them into bits. The better the LLM predicts, the fewer bits are needed. A token predicted at 99% confidence costs only ~0.014 bits. A token at 50% costs 1 bit. Only truly surprising tokens are expensive.

### Compression

```
Input text --> Split into N chunks (one per worker)
              --> Each worker independently:
                    Tokenize --> For each token:
                                    1. Ensemble predicts P(next token | context)
                                    2. Arithmetic encoder narrows interval
                                 --> Compressed stream
              --> Combine streams into NC05/NC06 file
```

### Decompression

```
NC05/NC06 file --> Split streams to N workers
                   --> Each worker:
                         For each position:
                            1. Same ensemble predicts same P(next token | context)
                            2. Arithmetic decoder recovers token
                            3. Recovered token feeds back as context
                         --> Tokens
                   --> Concatenate + Detokenize --> Original text
```

Both sides run the **exact same model with the exact same weights**, producing identical probability distributions. This symmetry guarantees perfect lossless reconstruction.

### Why Nacrith GPU Beats Traditional Compressors

- **gzip/xz/zip** use pattern matching on raw bytes within a sliding window. They only exploit local, literal repetitions.
- **Nacrith GPU** captures semantic and syntactic structure. It "knows" that after *"The President of the United"*, the word *"States"* is extremely likely — even if that exact phrase never appeared recently. This deep understanding of language produces far better predictions, which directly translates to fewer bits.

---

## Binary File Compression (NC06)

Nacrith GPU also supports compressing **binary files** such as PDFs, executables, and other non-UTF-8 data using a hybrid chunked approach. Binary mode is activated automatically when the input file is not valid UTF-8.

### How It Works

Binary files are rarely pure binary — they often contain significant amounts of embedded text (strings, metadata, markup, code). Nacrith GPU exploits this by **segmenting** the input into text and binary chunks, then compressing each with the best method for its type. Text chunks are compressed **in parallel** across GPU workers.

**1. Byte classification and segmentation**

Every byte is classified as text-like (printable ASCII 32-126, plus tab/LF/CR) or binary. Contiguous runs of the same type are grouped, then refined through several passes:

- Short text runs (< 64 bytes) are demoted to binary — too small to benefit from neural compression.
- Small binary gaps (< 8 bytes) between text runs are bridged — keeping the text chunk contiguous.
- Small binary chunks (< 64 bytes) adjacent to text are absorbed — avoiding fragment overhead.

The result is a clean sequence of alternating text and binary chunks.

**2. Binary blob compression**

All binary chunks are merged into a single blob and compressed with **lzma** (for blobs >= 4 KB) or **gzip** (for smaller blobs). If neither reduces the size, the raw bytes are stored as-is.

**3. Neural text compression (parallel)**

Each text chunk is split across workers and compressed using the full LLM + ensemble pipeline. Workers operate concurrently on their sub-chunks.

**4. NC06 file format**

| Section | Description |
|---------|-------------|
| Header (15 bytes) | Magic `NC06` + flags + temperature + version + entry count |
| Entry table | Per chunk: type byte (`T`/`B`) + original length (uint32) |
| Binary section | Compression method (`G`/`L`/`R`) + compressed length + data |
| Text streams (parallel) | Per text chunk: sub-chunk count + per-sub-chunk header + streams |

### Compression effectiveness on binary files

The compression ratio on binary files depends heavily on the **proportion of meaningful text** in the file. Files with large text regions (e.g., PDFs with embedded text content, HTML, XML, source code archives) will see significant compression gains on those regions. Files that are mostly opaque binary data (images, video, already-compressed archives) will see little to no improvement over gzip/lzma, since the neural model cannot predict non-text byte patterns.

| File type | Expected result |
|-----------|----------------|
| Text-heavy PDFs, HTML, XML | Good — large text chunks benefit from neural compression |
| Source code archives | Good — code is highly predictable for the LLM |
| Compressed archives (zip, gz) | Poor — already compressed, no text to exploit |
| Images, video, audio | Poor — almost entirely binary, falls back to gzip/lzma |

---

## Architecture

```
Nacrith-GPU-Parallel/
├── cli.py                 # Command-line interface (compress/decompress/benchmark)
├── parallel/
│   ├── __init__.py
│   └── compressor.py      # ParallelNeuralCompressor (NC05/NC06 formats)
├── compressor.py           # NeuralCompressor core (per-worker compression engine)
├── model_wrapper.py        # SmolLM2-135M wrapper (llama.cpp primary, PyTorch fallback)
├── arithmetic_coder.py     # Arithmetic encoder/decoder (32-bit precision)
├── ngram_model.py          # Token-level N-gram model (order 1-4)
├── context_mixer.py        # Adaptive linear context mixer
├── adaptive_head.py        # Online bias correction layer
├── utils.py                # CDF conversion, formatting helpers
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test configuration
├── benchmark/              # Benchmark text files
└── tests/
    ├── conftest.py         # Shared fixtures (model loading)
    ├── test_arithmetic.py  # Arithmetic coder tests (fast, no model)
    ├── test_chunking.py    # Binary segmentation tests (fast, no model)
    ├── test_model.py       # Model wrapper tests (slow)
    └── test_parallel.py    # Parallel roundtrip tests (slow)
```

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- ~500 MB disk space for the GGUF model

### Setup

```bash
# Clone the repository
git clone https://github.com/st4ck/Nacrith-GPU.git
cd Nacrith-GPU

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8 GPUs
pip install transformers accelerate numpy
pip install llama-cpp-python  # Primary inference backend

# Install test dependencies
pip install pytest
```

### GGUF Model

The llama.cpp backend requires a GGUF-format model file. Place it in the project root directory. The compressor looks for these filenames (in order):

1. `smollm2-135m-f32.gguf` (full precision, best compression)
2. `smollm2-135m-f16.gguf` (half precision)
3. `smollm2-135m.gguf`

To convert from HuggingFace format, use `convert_hf_to_gguf.py` from the [llama.cpp repository](https://github.com/ggerganov/llama.cpp).

> **Note on GPU compatibility:** If you have an older NVIDIA GPU (e.g., GTX 1050 Ti with CUDA capability 6.1), use PyTorch with CUDA 11.8 (`cu118`). Newer PyTorch builds with CUDA 12.x may not support older GPU architectures. The system will automatically fall back to CPU if GPU is unavailable.

## Usage

### Compress a text file

```bash
python cli.py compress input.txt output.nc
```

### Compress a binary file

Binary mode is detected automatically when the file is not valid UTF-8:

```bash
python cli.py compress document.pdf output.nc
```

### Decompress a file

```bash
python cli.py decompress output.nc restored.txt
```

### Set worker count

```bash
# Auto-detect from VRAM (default)
python cli.py compress input.txt output.nc

# Force 4 workers
python cli.py compress input.txt output.nc --workers 4

# Single worker (no parallelism)
python cli.py compress input.txt output.nc --workers 1
```

### Tune features

```bash
# Disable specific features
python cli.py compress input.txt output.nc --no-ngram

# Adjust hyperparameters
python cli.py compress input.txt output.nc --ngram-order 6 --temperature 0.8

# All options
python cli.py compress --help
```

### Benchmark against gzip

```bash
python cli.py benchmark input.txt
```

### Example session

```
$ python cli.py compress alice.txt alice.nc
Original: 148.5 KB
Mode: text (parallel, NC05)
Features: ngram(order=4), adaptive(lr=0.001), skip(threshold=1.5), warmup=100, workers=auto
Loading 4 llama.cpp instances...
  4 workers ready in 8.2s
Compressing: 100.0%  [4 workers]
Compressed: 17.0 KB
Ratio: 0.1145 (11.5%)
Time: 42.3s

$ python cli.py decompress alice.nc restored.txt
Decompressing: 100.0%
Decompressed: 148.5 KB
Time: 39.1s

$ diff alice.txt restored.txt && echo "Files match!"
Files match!
```

## Running Tests

```bash
# Fast tests (arithmetic coder + chunking, no model download needed)
python -m pytest tests/test_arithmetic.py tests/test_chunking.py -v

# All tests including model-dependent ones
python -m pytest -v -m slow

# All tests
python -m pytest -v
```

## Compressed File Formats

Files use the `.nc` extension (Nacrith Compressed). Two formats:

### NC05 — Parallel Text

Used for valid UTF-8 text files. The file is split into N chunks compressed in parallel.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 bytes | Magic | `NC05` |
| 4 | 1 byte | Flags | Feature flags (ngram, adaptive, skip) |
| 5 | 2 bytes | Temperature | uint16, fixed-point x10000 |
| 7 | 2 bytes | Chunk count | uint16, number of parallel chunks |
| 9 | 12 × N bytes | Chunk table | Per chunk: token count (4B) + bit count (4B) + stream length (4B) |
| variable | variable | Streams | Concatenated arithmetic-coded bitstreams |

### NC06 — Parallel Hybrid Binary

Used for non-UTF-8 files. See [Binary File Compression](#binary-file-compression-nc06) for the format details.

## Limitations

- **Speed**: Compression/decompression requires one neural network forward pass per token. Parallel workers improve throughput, but it's still slower than traditional compressors.
- **Model overhead**: The GGUF model weights (~500 MB for f32) must be available on both the compressor and decompressor side. This is amortized over many files but makes this impractical for compressing small individual files in isolation.
- **Binary files are not always compressible**: Neural compression excels on text. Binary files only benefit in proportion to their embedded text content. Files that are already compressed (zip, gz, jpeg) or mostly opaque binary data will not compress well — in some cases the output may be larger than the input.
- **Context window**: Nacrith uses a 2048-token context window (the model supports up to 8192). A sliding window with KV cache shifting is used for longer texts, which may slightly reduce compression efficiency at window boundaries.

## Theory

This project implements the insight from Shannon's information theory: **compression is equivalent to prediction**. The better a model predicts the next symbol, the fewer bits are needed to encode it.

The theoretical lower bound for lossless compression is the **entropy** of the source:

```
H(X) = -Σ P(x) log₂ P(x)
```

Arithmetic coding achieves compression rates within a fraction of a bit of the entropy. By using a neural language model that assigns high probabilities to likely tokens, the effective entropy (cross-entropy between the model and the true text distribution) is much lower than what byte-level pattern matching can achieve.

For further reading:
- Shannon, C. E. (1948). *A Mathematical Theory of Communication*
- [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (DeepMind, ICLR 2024)
- [LLMZip: Lossless Text Compression using Large Language Models](https://arxiv.org/abs/2306.04050)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
