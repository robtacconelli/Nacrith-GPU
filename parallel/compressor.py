"""Parallel neural text compressor.

Splits text into chunks and compresses them concurrently using
multiple llama.cpp model instances on the GPU.  Threading works
because llama-cpp-python releases the GIL during C-level inference.

Format: NC05 (parallel text streams)
    b'NC05'                         4 bytes  magic
    flags                           1 byte   feature flags
    temperature                     2 bytes  uint16, fixed-point × 10000
    n_chunks                        2 bytes  uint16
    per-chunk table (n_chunks ×):
        num_tokens                  4 bytes  uint32
        compressed_bits             4 bytes  uint32
        stream_length               4 bytes  uint32
    streams                         concatenated

Format: NC06 (parallel hybrid binary/text)
    Header (15 bytes):
        b'NC06' | flags | temperature | version=1 | num_entries
    Entry table:
        Per entry: chunk_type (1B) | original_length (4B)
    Binary blob:
        method (1B) | compressed_length (4B) | compressed_data
    Text chunks — PARALLEL:
        Per text entry:
            n_sub_chunks (2B, uint16)
            Sub-chunk table (n_sub_chunks × 12B):
                num_tokens (4B) | compressed_bits (4B) | stream_length (4B)
            Streams (concatenated)
"""

import gzip
import lzma
import os
import struct
import sys
import threading
import time

import numpy as np

from llama_cpp import Llama
from transformers import AutoTokenizer

from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder
from compressor import (
    NeuralCompressor,
    _segment_chunks, CHUNK_TYPE_TEXT, CHUNK_TYPE_BINARY,
    BLOB_GZIP, BLOB_LZMA, BLOB_RAW, LZMA_THRESHOLD,
)
from model_wrapper import ModelWrapper
from utils import probs_to_cdf

MAGIC_NC05 = b"NC05"
MAGIC_NC06 = b"NC06"
NC06_VERSION = 1
HEADER_BASE = 4 + 1 + 2 + 2  # magic + flags + temp + n_chunks
ENTRY_SIZE = 4 + 4 + 4        # tokens + bits + stream_len

# Minimum tokens per chunk — below this, splitting isn't worth it.
MIN_CHUNK_TOKENS = 200

# VRAM budget for auto-calculating worker count.
_VRAM_FIRST_INSTANCE_MB = 1169   # model weights + KV cache + compute
_VRAM_EXTRA_INSTANCE_MB = 660    # KV cache + compute (weights shared)
_VRAM_RESERVE_MB = 512           # keep free for OS / other apps
# Beyond 8 workers, GPU contention causes throughput to drop.
_MAX_USEFUL_WORKERS = 8


def _auto_worker_count() -> int:
    """Calculate optimal worker count from available VRAM."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        free_mb = int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 1  # no GPU or nvidia-smi failed

    usable = free_mb - _VRAM_RESERVE_MB
    if usable < _VRAM_FIRST_INSTANCE_MB:
        return 1

    # First instance costs more (includes model weights)
    remaining = usable - _VRAM_FIRST_INSTANCE_MB
    extra = max(0, remaining // _VRAM_EXTRA_INSTANCE_MB)
    n = 1 + extra

    return min(n, _MAX_USEFUL_WORKERS)


def _split_text(text: str, n_chunks: int) -> list[str]:
    """Split text into roughly equal chunks at newline boundaries."""
    if n_chunks <= 1 or not text:
        return [text]

    target_size = len(text) // n_chunks
    chunks = []
    start = 0

    for _ in range(n_chunks - 1):
        target = start + target_size
        # Search for a newline near the target split point
        split = text.find("\n", target)
        if split == -1 or split > target + target_size // 2:
            # No good newline; try before the target
            split = text.rfind("\n", start, target)
        if split == -1 or split <= start:
            # No newline at all — split at target
            split = target
        else:
            split += 1  # include the newline in this chunk

        chunks.append(text[start:split])
        start = split

    chunks.append(text[start:])

    # Remove empty trailing chunks
    while len(chunks) > 1 and not chunks[-1]:
        chunks.pop()

    return chunks


class ParallelNeuralCompressor:
    """Compresses text using multiple llama.cpp instances in parallel.

    Each worker thread owns its own model instance and NeuralCompressor,
    so all state (KV cache, secondary models, mixer) is independent.
    """

    def __init__(
        self,
        n_workers: int = 0,
        gguf_path: str = None,
        model_name: str = None,
        verbose: bool = True,
        **compressor_kwargs,
    ):
        if n_workers <= 0:
            n_workers = _auto_worker_count()
        self.n_workers = n_workers
        self.verbose = verbose
        self._compressor_kwargs = compressor_kwargs

        model_name = model_name or ModelWrapper.MODEL_NAME

        # Find GGUF path
        if gguf_path is None:
            tmp = ModelWrapper(verbose=False)
            gguf_path = tmp._find_gguf()
            del tmp
            if gguf_path is None:
                raise FileNotFoundError(
                    "No GGUF model found. Convert with convert_hf_to_gguf.py"
                )

        self._gguf_path = gguf_path
        self._model_name = model_name

        # Load shared tokenizer (used for token-count estimation only)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Spin up worker models + compressors
        if self.verbose:
            print(
                f"Loading {n_workers} llama.cpp instances...",
                file=sys.stderr,
            )
        t0 = time.perf_counter()
        self._workers = []
        for _ in range(n_workers):
            m = ModelWrapper(
                model_name=model_name, gguf_path=gguf_path, verbose=False,
            )
            c = NeuralCompressor(model=m, verbose=False, **compressor_kwargs)
            self._workers.append((m, c))
        elapsed = time.perf_counter() - t0
        if self.verbose:
            print(
                f"  {n_workers} workers ready in {elapsed:.1f}s",
                file=sys.stderr,
            )

        self._monitor_stop = threading.Event()

        # Grab config flags and temperature from first compressor
        self._flags = self._workers[0][1]._config_flags()
        self._temperature = self._workers[0][1].temperature

    # ------------------------------------------------------------------
    # Progress monitor
    # ------------------------------------------------------------------

    def _start_monitor(self, label: str, total_tokens: int,
                        active_indices: list[int],
                        tokens_done_offset: int = 0):
        """Start a background thread that prints aggregate progress.

        Args:
            label: Prefix for the progress line (e.g. "Compressing").
            total_tokens: Grand total tokens across all chunks.
            active_indices: Chunk indices currently being processed.
            tokens_done_offset: Tokens already finished in prior batches.
        """
        if not self.verbose:
            return None
        self._monitor_stop.clear()
        workers = self._workers
        n_w = len(workers)

        def _monitor():
            prev_done = tokens_done_offset
            prev_time = self._monitor_t0
            while not self._monitor_stop.wait(0.5):
                batch_done = 0
                for idx in active_indices:
                    _, comp = workers[idx % n_w]
                    batch_done += comp._progress
                done = tokens_done_offset + batch_done
                if total_tokens > 0:
                    now = time.perf_counter()
                    elapsed = now - self._monitor_t0
                    avg_tps = done / elapsed if elapsed > 0 else 0
                    dt = now - prev_time
                    inst_tps = (done - prev_done) / dt if dt > 0 else 0
                    prev_done = done
                    prev_time = now
                    print(
                        f"\r{label}: {done}/{total_tokens} tokens "
                        f"({100 * done / total_tokens:.1f}%) "
                        f"[{inst_tps:.0f} now, {avg_tps:.0f} avg tok/s]",
                        end="", file=sys.stderr,
                    )

        self._monitor_t0 = time.perf_counter()
        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return t

    def _stop_monitor(self, monitor_thread):
        """Stop the progress monitor and clear the line."""
        if monitor_thread is None:
            return
        self._monitor_stop.set()
        monitor_thread.join()
        if self.verbose:
            print("\r" + " " * 80 + "\r", end="", file=sys.stderr)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, text: str) -> bytes:
        """Compress text using parallel workers (NC05 format)."""
        if not text:
            return self._pack_header(0) + b""

        chunks = _split_text(text, self.n_workers)
        n_chunks = len(chunks)

        est_tokens = [
            len(self._tokenizer.encode(c)) for c in chunks
        ]
        est_total = sum(est_tokens)

        if self.verbose:
            print(
                f"Split into {n_chunks} chunks: "
                + ", ".join(f"{t} tok" for t in est_tokens)
                + f"  (total {est_total})",
                file=sys.stderr,
            )

        # Compress chunks in parallel
        results = [None] * n_chunks
        errors = [None] * n_chunks

        def _compress_worker(idx):
            try:
                model, comp = self._workers[idx % len(self._workers)]
                model.reset_cache()
                comp._reset_secondary_models()
                results[idx] = comp._compress_text_to_stream(chunks[idx])
            except Exception as e:
                errors[idx] = e

        t0 = time.perf_counter()
        monitor = self._start_monitor(
            "Compressing", est_total, list(range(n_chunks)),
        )
        threads = []
        for i in range(n_chunks):
            t = threading.Thread(target=_compress_worker, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        self._stop_monitor(monitor)

        # Check for errors
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"Worker {i} failed: {err}") from err

        elapsed = time.perf_counter() - t0
        total_tokens = sum(r[0] for r in results)

        if self.verbose:
            print(
                f"Compressed {total_tokens} tokens in {elapsed:.2f}s "
                f"({total_tokens / elapsed:.0f} tok/s)",
                file=sys.stderr,
            )

        return self._pack(n_chunks, results)

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def decompress(self, data: bytes) -> "str | bytes":
        """Decompress NC05 or NC06 parallel format."""
        magic = data[:4]
        if magic == MAGIC_NC06:
            return self._decompress_nc06(data)
        if magic != MAGIC_NC05:
            raise ValueError(f"Expected NC05/NC06 magic, got {magic!r}")

        flags = data[4]
        temp_encoded, n_chunks = struct.unpack(">HH", data[5:9])

        if n_chunks == 0:
            return ""

        # Parse entry table
        entries = []
        offset = HEADER_BASE
        for _ in range(n_chunks):
            num_tokens, comp_bits, stream_len = struct.unpack(
                ">III", data[offset : offset + ENTRY_SIZE],
            )
            entries.append((num_tokens, comp_bits, stream_len))
            offset += ENTRY_SIZE

        # Extract streams
        streams = []
        for num_tokens, comp_bits, stream_len in entries:
            streams.append(data[offset : offset + stream_len])
            offset += stream_len

        # Configure workers for decompression flags
        temperature = temp_encoded / 10000.0
        for _, comp in self._workers:
            comp.temperature = temperature
            comp._apply_flags(flags)

        # Decompress in parallel, processing at most n_workers chunks
        # concurrently so each worker is used by exactly one thread.
        # (n_chunks can exceed n_workers when the file was compressed
        # with more workers than are available for decompression.)
        texts = [None] * n_chunks
        errors = [None] * n_chunks
        n_workers = len(self._workers)

        def _decompress_worker(idx):
            try:
                worker_idx = idx % n_workers
                model, comp = self._workers[worker_idx]
                model.reset_cache()
                comp._reset_secondary_models()
                num_tokens = entries[idx][0]
                texts[idx] = comp._decompress_text_stream(
                    streams[idx], num_tokens,
                )
            except Exception as e:
                errors[idx] = e

        total_tokens = sum(e[0] for e in entries)
        tokens_done = 0

        t0 = time.perf_counter()
        self._monitor_t0 = time.perf_counter()
        for batch_start in range(0, n_chunks, n_workers):
            batch_end = min(batch_start + n_workers, n_chunks)
            batch_indices = list(range(batch_start, batch_end))

            monitor = self._start_monitor(
                "Decompressing", total_tokens,
                batch_indices, tokens_done,
            )

            threads = []
            for i in batch_indices:
                t = threading.Thread(target=_decompress_worker, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            self._stop_monitor(monitor)

            for i in batch_indices:
                if errors[i] is not None:
                    raise RuntimeError(
                        f"Worker {i} failed: {errors[i]}"
                    ) from errors[i]
                tokens_done += entries[i][0]

        elapsed = time.perf_counter() - t0

        if self.verbose:
            print(
                f"Decompressed {total_tokens} tokens in {elapsed:.2f}s "
                f"({total_tokens / elapsed:.0f} tok/s)",
                file=sys.stderr,
            )

        return "".join(texts)

    # ------------------------------------------------------------------
    # Hybrid binary compression (NC06)
    # ------------------------------------------------------------------

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes using parallel hybrid format (NC06).

        Combines binary/text segmentation with NC05-style parallel text
        compression across multiple workers.
        """
        chunks = _segment_chunks(data)
        num_entries = len(chunks)

        flags = self._flags
        temp_encoded = int(round(self._temperature * 10000))

        file_header = MAGIC_NC06 + struct.pack(
            '>BHII', flags, temp_encoded, NC06_VERSION, num_entries,
        )

        if num_entries == 0:
            return file_header

        # Build entry table and classify chunks
        entry_table = []
        binary_parts = []
        text_indices = []
        total_binary = 0

        for ci, (chunk_type, offset, length) in enumerate(chunks):
            entry_table.append(struct.pack('>BI', chunk_type, length))
            if chunk_type == CHUNK_TYPE_BINARY:
                binary_parts.append(data[offset:offset + length])
                total_binary += length
            else:
                text_indices.append(ci)

        # Compress merged binary blob
        if total_binary > 0:
            binary_blob = b''.join(binary_parts)

            if self.verbose:
                print(f"Binary blob: {total_binary} bytes", file=sys.stderr)

            if total_binary >= LZMA_THRESHOLD:
                compressed = lzma.compress(binary_blob)
                method = BLOB_LZMA
            else:
                compressed = gzip.compress(binary_blob, compresslevel=9)
                method = BLOB_GZIP

            if len(compressed) >= total_binary:
                compressed = binary_blob
                method = BLOB_RAW

            if self.verbose:
                labels = {
                    BLOB_GZIP: "gzip", BLOB_LZMA: "lzma", BLOB_RAW: "raw",
                }
                print(
                    f"Binary blob compressed: {len(compressed)} bytes "
                    f"({labels[method]})",
                    file=sys.stderr,
                )

            binary_section = (
                struct.pack('>BI', method, len(compressed)) + compressed
            )
        else:
            binary_section = b''

        # Compress text chunks in parallel (NC05-style sub-chunks)
        n_workers = self.n_workers
        text_sections = []

        # Estimate total tokens across all text chunks for progress
        all_sub_chunks = []  # list of (text_idx, sub_chunks_list)
        grand_total_tokens = 0
        for ti, ci in enumerate(text_indices):
            chunk_type, offset, length = chunks[ci]
            chunk_data = data[offset:offset + length]
            text = chunk_data.decode('latin-1')

            sub_chunks = _split_text(text, n_workers)
            est_tokens = [
                len(self._tokenizer.encode(sc)) for sc in sub_chunks
            ]
            grand_total_tokens += sum(est_tokens)
            all_sub_chunks.append((ci, sub_chunks, est_tokens))

        if self.verbose:
            print(
                f"Text chunks: {len(text_indices)}, "
                f"total sub-chunks: "
                f"{sum(len(sc) for _, sc, _ in all_sub_chunks)}, "
                f"estimated tokens: {grand_total_tokens}",
                file=sys.stderr,
            )

        tokens_done_offset = 0
        t0 = time.perf_counter()
        self._monitor_t0 = time.perf_counter()

        for ti, (ci, sub_chunks, est_tokens) in enumerate(all_sub_chunks):
            n_sub = len(sub_chunks)

            if self.verbose:
                print(
                    f"Text chunk {ti+1}/{len(text_indices)}: "
                    f"{n_sub} sub-chunks, "
                    f"~{sum(est_tokens)} tokens",
                    file=sys.stderr,
                )

            # Compress sub-chunks in parallel
            results = [None] * n_sub
            errors = [None] * n_sub

            def _compress_worker(idx, _sub_chunks=sub_chunks):
                try:
                    model, comp = self._workers[idx % n_workers]
                    model.reset_cache()
                    comp._reset_secondary_models()
                    results[idx] = comp._compress_text_to_stream(
                        _sub_chunks[idx],
                    )
                except Exception as e:
                    errors[idx] = e

            monitor = self._start_monitor(
                "Compressing", grand_total_tokens,
                list(range(n_sub)), tokens_done_offset,
            )

            threads = []
            for i in range(n_sub):
                t = threading.Thread(target=_compress_worker, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            self._stop_monitor(monitor)

            # Check for errors
            for i, err in enumerate(errors):
                if err is not None:
                    raise RuntimeError(
                        f"Worker {i} failed on text chunk {ti}: {err}"
                    ) from err

            # Pack this text entry: n_sub_chunks + sub-chunk table + streams
            sub_entry_table = b""
            sub_streams = b""
            for num_tokens, comp_bits, stream in results:
                sub_entry_table += struct.pack(
                    ">III", num_tokens, comp_bits, len(stream),
                )
                sub_streams += stream
                tokens_done_offset += num_tokens

            text_sections.append(
                struct.pack(">H", n_sub) + sub_entry_table + sub_streams
            )

        elapsed = time.perf_counter() - t0
        if self.verbose:
            total_tokens = tokens_done_offset
            print(
                f"Compressed {total_tokens} tokens in {elapsed:.2f}s "
                f"({total_tokens / elapsed:.0f} tok/s)",
                file=sys.stderr,
            )

        return (
            file_header
            + b''.join(entry_table)
            + binary_section
            + b''.join(text_sections)
        )

    def _decompress_nc06(self, data: bytes) -> bytes:
        """Decompress NC06 (parallel hybrid binary) format."""
        flags = data[4]
        temp_encoded, _version, num_entries = struct.unpack(
            '>HII', data[5:15],
        )
        temperature = temp_encoded / 10000.0

        if num_entries == 0:
            return b""

        # Configure workers
        for _, comp in self._workers:
            comp.temperature = temperature
            comp._apply_flags(flags)

        # Parse entry table
        pos = 15
        entries = []
        total_binary = 0
        for _ in range(num_entries):
            etype, elen = struct.unpack('>BI', data[pos:pos + 5])
            entries.append((etype, elen))
            if etype == CHUNK_TYPE_BINARY:
                total_binary += elen
            pos += 5

        # Decompress binary blob
        binary_data = b''
        if total_binary > 0:
            method, comp_len = struct.unpack('>BI', data[pos:pos + 5])
            pos += 5
            compressed = data[pos:pos + comp_len]
            pos += comp_len

            if method == BLOB_RAW:
                binary_data = compressed
            elif method == BLOB_GZIP:
                binary_data = gzip.decompress(compressed)
            elif method == BLOB_LZMA:
                binary_data = lzma.decompress(compressed)

            if self.verbose:
                labels = {
                    BLOB_GZIP: "gzip", BLOB_LZMA: "lzma", BLOB_RAW: "raw",
                }
                print(
                    f"Binary blob: {total_binary} bytes "
                    f"({labels.get(method, '?')})",
                    file=sys.stderr,
                )

        # First pass: parse all text sub-chunk headers to compute total tokens
        text_entry_data = []  # (n_sub, sub_entries, sub_streams, entry_pos)
        grand_total_tokens = 0
        scan_pos = pos

        for ci, (etype, elen) in enumerate(entries):
            if etype != CHUNK_TYPE_TEXT:
                continue

            n_sub = struct.unpack(">H", data[scan_pos:scan_pos + 2])[0]
            scan_pos += 2

            sub_entries = []
            for _ in range(n_sub):
                num_tokens, comp_bits, stream_len = struct.unpack(
                    ">III", data[scan_pos:scan_pos + 12],
                )
                sub_entries.append((num_tokens, comp_bits, stream_len))
                grand_total_tokens += num_tokens
                scan_pos += 12

            sub_streams = []
            for num_tokens, comp_bits, stream_len in sub_entries:
                sub_streams.append(data[scan_pos:scan_pos + stream_len])
                scan_pos += stream_len

            text_entry_data.append((ci, n_sub, sub_entries, sub_streams))

        # Decompress text entries in parallel (batched by workers)
        n_workers = len(self._workers)
        text_results = {}  # ci -> decoded text string
        tokens_done = 0
        t0 = time.perf_counter()
        self._monitor_t0 = time.perf_counter()

        for ti, (ci, n_sub, sub_entries, sub_streams) in enumerate(
            text_entry_data
        ):
            if self.verbose:
                chunk_tokens = sum(e[0] for e in sub_entries)
                print(
                    f"Text chunk {ti+1}/{len(text_entry_data)}: "
                    f"{n_sub} sub-chunks, {chunk_tokens} tokens",
                    file=sys.stderr,
                )

            texts = [None] * n_sub
            errors = [None] * n_sub

            def _decompress_worker(
                idx, _sub_entries=sub_entries, _sub_streams=sub_streams,
            ):
                try:
                    worker_idx = idx % n_workers
                    model, comp = self._workers[worker_idx]
                    model.reset_cache()
                    comp._reset_secondary_models()
                    num_tokens = _sub_entries[idx][0]
                    texts[idx] = comp._decompress_text_stream(
                        _sub_streams[idx], num_tokens,
                    )
                except Exception as e:
                    errors[idx] = e

            # Process in batches of n_workers
            for batch_start in range(0, n_sub, n_workers):
                batch_end = min(batch_start + n_workers, n_sub)
                batch_indices = list(range(batch_start, batch_end))

                monitor = self._start_monitor(
                    "Decompressing", grand_total_tokens,
                    batch_indices, tokens_done,
                )

                threads = []
                for i in batch_indices:
                    t = threading.Thread(
                        target=_decompress_worker, args=(i,),
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                self._stop_monitor(monitor)

                for i in batch_indices:
                    if errors[i] is not None:
                        raise RuntimeError(
                            f"Worker {i} failed on text chunk {ti}: "
                            f"{errors[i]}"
                        ) from errors[i]
                    tokens_done += sub_entries[i][0]

            text_results[ci] = "".join(texts)

        elapsed = time.perf_counter() - t0
        if self.verbose and grand_total_tokens > 0:
            print(
                f"Decompressed {grand_total_tokens} tokens in {elapsed:.2f}s "
                f"({grand_total_tokens / elapsed:.0f} tok/s)",
                file=sys.stderr,
            )

        # Reassemble output in entry order
        binary_offset = 0
        output_parts = []

        for ci, (etype, elen) in enumerate(entries):
            if etype == CHUNK_TYPE_BINARY:
                output_parts.append(
                    binary_data[binary_offset:binary_offset + elen]
                )
                binary_offset += elen
            else:
                output_parts.append(
                    text_results[ci].encode('latin-1')
                )

        return b''.join(output_parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _pack_header(self, n_chunks: int) -> bytes:
        temp_encoded = int(round(self._temperature * 10000))
        return MAGIC_NC05 + struct.pack(">BHH", self._flags, temp_encoded, n_chunks)

    def _pack(self, n_chunks: int, results: list) -> bytes:
        header = self._pack_header(n_chunks)

        entry_table = b""
        stream_data = b""
        for num_tokens, comp_bits, stream in results:
            entry_table += struct.pack(">III", num_tokens, comp_bits, len(stream))
            stream_data += stream

        return header + entry_table + stream_data
