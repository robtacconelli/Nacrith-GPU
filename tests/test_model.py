"""Tests for model wrapper — require model download (slow)."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.slow
def test_model_loads(model_wrapper):
    assert model_wrapper is not None
    assert model_wrapper.model is not None
    assert model_wrapper.tokenizer is not None


@pytest.mark.slow
def test_device_type(model_wrapper):
    assert model_wrapper.device in ("cuda", "cpu")


@pytest.mark.slow
def test_get_probs_shape(model_wrapper):
    probs = model_wrapper.get_probs([1, 2, 3])
    assert isinstance(probs, torch.Tensor)
    assert probs.ndim == 1
    assert probs.shape[0] == model_wrapper.vocab_size


@pytest.mark.slow
def test_get_probs_valid_distribution(model_wrapper):
    probs = model_wrapper.get_probs([1, 2, 3])
    assert torch.all(probs >= 0).item()
    assert abs(probs.sum().item() - 1.0) < 1e-3


@pytest.mark.slow
def test_get_probs_deterministic(model_wrapper):
    probs1 = model_wrapper.get_probs([5, 10, 15, 20])
    probs2 = model_wrapper.get_probs([5, 10, 15, 20])
    assert torch.equal(probs1, probs2)


@pytest.mark.slow
@pytest.mark.parametrize("context_len", [1, 10, 50])
def test_different_context_lengths(model_wrapper, context_len):
    context = list(range(1, context_len + 1))
    probs = model_wrapper.get_probs(context)
    assert torch.all(probs >= 0).item()
    assert abs(probs.sum().item() - 1.0) < 1e-3


@pytest.mark.slow
def test_tokenizer_roundtrip(model_wrapper):
    text = "Hello, World! This is a test."
    tokens = model_wrapper.tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    decoded = model_wrapper.tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
