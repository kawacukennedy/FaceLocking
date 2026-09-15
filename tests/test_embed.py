"""Tests for the ArcFace ONNX embedder."""

from pathlib import Path

import numpy as np
import pytest

from src.embed import ArcFaceEmbedderONNX, DEFAULT_MODEL_PATH

MODEL = Path(DEFAULT_MODEL_PATH)

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and MODEL.stat().st_size > 1_000_000),
    reason="ArcFace ONNX model not present; run scripts/download_models.sh",
)


@pytest.fixture(scope="module")
def embedder():
    return ArcFaceEmbedderONNX(model_path=str(MODEL))


def test_embedding_shape_and_normalization(embedder):
    aligned = np.random.default_rng(0).integers(0, 256, size=(112, 112, 3)).astype(np.uint8)
    result = embedder.embed(aligned)
    assert result.embedding.shape == (512,)
    assert result.dim == 512
    norm = np.linalg.norm(result.embedding)
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_embedding_preprocess_input_layout():
    embedder = ArcFaceEmbedderONNX(model_path=str(MODEL))
    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    tensor = embedder._preprocess(aligned)
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32


def test_same_input_same_embedding(embedder):
    rng = np.random.default_rng(7)
    aligned = rng.integers(0, 256, size=(112, 112, 3)).astype(np.uint8)
    a = embedder.embed(aligned).embedding
    b = embedder.embed(aligned.copy()).embedding
    assert np.allclose(a, b, atol=1e-6)


def test_different_inputs_differ(embedder):
    rng = np.random.default_rng(1)
    a = embedder.embed(rng.integers(0, 256, size=(112, 112, 3)).astype(np.uint8)).embedding
    b = embedder.embed(rng.integers(0, 256, size=(112, 112, 3)).astype(np.uint8)).embedding
    assert not np.allclose(a, b, atol=1e-3)