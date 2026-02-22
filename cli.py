#!/usr/bin/env python3
"""Command-line interface for the parallel neural compressor.

Uses ParallelNeuralCompressor for compression (NC05 text, NC06 binary)
with automatic GPU worker detection.

Decompression automatically reads the config from the file header.
"""

import argparse
import gzip
import sys
import time

from parallel import ParallelNeuralCompressor
from utils import format_size


def _make_parallel_compressor(args) -> ParallelNeuralCompressor:
    """Build a ParallelNeuralCompressor from CLI arguments."""
    return ParallelNeuralCompressor(
        n_workers=args.workers,
        verbose=True,
        use_ngram=not args.no_ngram,
        use_lzp=not args.no_lzp,
        use_adaptive_head=not args.no_adaptive,
        use_confidence_skip=not args.no_skip,
        ngram_order=args.ngram_order,
        lzp_max_order=args.lzp_order,
        mixer_lr=args.mixer_lr,
        adaptive_lr=args.adaptive_lr,
        skip_threshold=args.skip_threshold,
        warmup=args.warmup,
        temperature=args.temperature,
    )


def cmd_compress(args):
    binary_mode = False
    try:
        with open(args.input, 'r', encoding='utf-8', newline='') as f:
            text = f.read()
        raw = text.encode('utf-8')
    except UnicodeDecodeError:
        binary_mode = True
        with open(args.input, 'rb') as f:
            raw = f.read()

    original_size = len(raw)
    print(f"Original: {format_size(original_size)}")

    if binary_mode:
        print("Mode: binary (hybrid, parallel, NC06)")

        # Show active features
        features = []
        if not args.no_ngram:
            features.append(f"ngram(order={args.ngram_order})")
        if not args.no_lzp:
            features.append(f"lzp(order={args.lzp_order})")
        if not args.no_adaptive:
            features.append(f"adaptive(lr={args.adaptive_lr})")
        if not args.no_skip:
            features.append(f"skip(threshold={args.skip_threshold})")
        features.append(f"warmup={args.warmup}")
        if args.temperature != 1.0:
            features.append(f"temp={args.temperature}")
        workers_str = "auto" if args.workers == 0 else str(args.workers)
        features.append(f"workers={workers_str}")
        print(f"Features: {', '.join(features)}")

        pc = _make_parallel_compressor(args)
        start = time.time()
        compressed = pc.compress_bytes(raw)
        elapsed = time.time() - start
    else:
        print(f"Mode: text (parallel, NC05)")

        # Show active features
        features = []
        if not args.no_ngram:
            features.append(f"ngram(order={args.ngram_order})")
        if not args.no_lzp:
            features.append(f"lzp(order={args.lzp_order})")
        if not args.no_adaptive:
            features.append(f"adaptive(lr={args.adaptive_lr})")
        if not args.no_skip:
            features.append(f"skip(threshold={args.skip_threshold})")
        features.append(f"warmup={args.warmup}")
        if args.temperature != 1.0:
            features.append(f"temp={args.temperature}")
        workers_str = f"auto" if args.workers == 0 else str(args.workers)
        features.append(f"workers={workers_str}")
        print(f"Features: {', '.join(features)}")

        pc = _make_parallel_compressor(args)
        start = time.time()
        compressed = pc.compress(text)
        elapsed = time.time() - start

    with open(args.output, 'wb') as f:
        f.write(compressed)

    comp_size = len(compressed)
    ratio = comp_size / original_size if original_size > 0 else 0
    print(f"Compressed: {format_size(comp_size)}")
    print(f"Ratio: {ratio:.4f} ({100*ratio:.1f}%)")
    print(f"Time: {elapsed:.1f}s")


def cmd_decompress(args):
    with open(args.input, 'rb') as f:
        data = f.read()

    magic = data[:4]
    if magic not in (b"NC05", b"NC06"):
        raise ValueError(
            f"Unsupported format: {magic!r} (expected NC05 or NC06)"
        )

    pc = ParallelNeuralCompressor(
        n_workers=args.workers, verbose=True,
    )
    start = time.time()
    result = pc.decompress(data)
    elapsed = time.time() - start

    if isinstance(result, bytes):
        with open(args.output, 'wb') as f:
            f.write(result)
        print(f"Decompressed: {format_size(len(result))} (binary)")
    else:
        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            f.write(result)
        print(f"Decompressed: {format_size(len(result.encode('utf-8')))}")
    print(f"Time: {elapsed:.1f}s")


def cmd_benchmark(args):
    binary_mode = False
    try:
        with open(args.input, 'r', encoding='utf-8', newline='') as f:
            text = f.read()
        raw = text.encode('utf-8')
    except UnicodeDecodeError:
        binary_mode = True
        with open(args.input, 'rb') as f:
            raw = f.read()

    original_size = len(raw)
    print(f"Original: {format_size(original_size)}")
    if binary_mode:
        print("Mode: binary (hybrid)")
    print()

    # gzip baseline
    gzip_data = gzip.compress(raw, compresslevel=9)
    gzip_size = len(gzip_data)
    print(f"gzip:      {format_size(gzip_size):>10s}  "
          f"(ratio: {gzip_size/original_size:.4f})")

    if binary_mode:
        # Binary: use parallel compressor (NC06)
        pc = _make_parallel_compressor(args)
        start = time.time()
        compressed = pc.compress_bytes(raw)
        elapsed = time.time() - start
    else:
        # Text: use parallel compressor
        pc = _make_parallel_compressor(args)
        start = time.time()
        compressed = pc.compress(text)
        elapsed = time.time() - start

    nc_size = len(compressed)
    print(f"Nacrith:   {format_size(nc_size):>10s}  "
          f"(ratio: {nc_size/original_size:.4f})  "
          f"[{elapsed:.1f}s]")
    print()

    if nc_size < gzip_size:
        improvement = (1 - nc_size / gzip_size) * 100
        print(f"Nacrith is {improvement:.1f}% smaller than gzip")
    elif nc_size > gzip_size:
        overhead = (nc_size / gzip_size - 1) * 100
        print(f"Nacrith is {overhead:.1f}% larger than gzip")
    else:
        print("Nacrith and gzip produce identical size")


def _add_feature_args(parser):
    """Add feature toggle arguments to a parser."""
    g = parser.add_argument_group("features")
    g.add_argument(
        "--no-ngram", action="store_true",
        help="Disable N-gram model",
    )
    g.add_argument(
        "--no-lzp", action="store_true", default=True,
        help="Disable LZP (match) model",
    )
    g.add_argument(
        "--lzp", action="store_false", dest="no_lzp",
        help="Enable LZP (match) model",
    )
    g.add_argument(
        "--no-adaptive", action="store_true",
        help="Disable adaptive head (online bias learning)",
    )
    g.add_argument(
        "--no-skip", action="store_true",
        help="Disable confidence-based LLM skipping",
    )

    h = parser.add_argument_group("hyperparameters")
    h.add_argument(
        "--ngram-order", type=int, default=4,
        help="N-gram max order (default: 4)",
    )
    h.add_argument(
        "--lzp-order", type=int, default=8,
        help="LZP max order (default: 8)",
    )
    h.add_argument(
        "--mixer-lr", type=float, default=0.5,
        help="Mixer learning rate (default: 0.5)",
    )
    h.add_argument(
        "--adaptive-lr", type=float, default=0.001,
        help="Adaptive head learning rate (default: 0.001)",
    )
    h.add_argument(
        "--skip-threshold", type=float, default=1.5,
        help="Entropy threshold for LLM skipping in bits (default: 1.5)",
    )
    h.add_argument(
        "--warmup", type=int, default=100,
        help="Tokens of LLM-only warmup before mixing starts (default: 100)",
    )
    h.add_argument(
        "--temperature", type=float, default=1.0,
        help="Softmax temperature: <1.0 sharpens predictions (try 0.7-0.9), "
             ">1.0 softens (default: 1.0)",
    )
    h.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel GPU workers (default: 0 = auto-detect from VRAM)",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Nacrith GPU Parallel — neural compressor with "
            "context mixing and multi-instance GPU parallelism. "
            "Uses SmolLM2-135M + N-gram + LZP + adaptive mixer."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_comp = sub.add_parser(
        "compress",
        help="Compress a file using parallel neural ensemble (NC05)",
    )
    p_comp.add_argument("input", help="Input file")
    p_comp.add_argument("output", help="Output compressed file (.nc5)")
    _add_feature_args(p_comp)
    p_comp.set_defaults(func=cmd_compress)

    # decompress
    p_decomp = sub.add_parser(
        "decompress",
        help="Decompress a compressed file (NC05/NC06)",
    )
    p_decomp.add_argument("input", help="Input compressed file")
    p_decomp.add_argument("output", help="Output file")
    p_decomp.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel GPU workers for NC05/NC06 (default: 0 = auto)",
    )
    p_decomp.set_defaults(func=cmd_decompress)

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Benchmark Nacrith vs gzip",
    )
    p_bench.add_argument("input", help="Input file")
    _add_feature_args(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
