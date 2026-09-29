"""Recognition cache tests.

The cache exists so live recognition stays smooth: an expensive ArcFace pass is
repeated only when a face moved or its entry went stale. These tests use a fake
embedder, so no model is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recognize import FaceDBMatcher, RecognitionCache, match_expression
from src.expressions import FaceExpression


class FakeFace:
    """Minimal stand-in for a detected face."""

    def __init__(self, x1, y1, x2, y2, kps=None):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.kps = np.zeros((5, 2), dtype=np.float32) if kps is None else kps


class FakeEmbedder:
    """Counts embeddings and returns a deterministic vector per call."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, aligned_bgr):
        self.calls += 1
        vector = np.zeros(4, dtype=np.float32)
        vector[0] = 1.0  # identical to the stored template
        from src.embed import EmbeddingResult

        return EmbeddingResult(embedding=vector, norm_before=1.0, dim=4)


@pytest.fixture
def matcher() -> FaceDBMatcher:
    return FaceDBMatcher({"Kennedy": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)})


def test_first_sight_embeds(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher)
    out = cache.identify(np.zeros((480, 640, 3), np.uint8), [FakeFace(100, 100, 200, 200)], now=0.0)
    assert len(out) == 1
    assert out[0].label == "Kennedy"
    assert out[0].reused is False
    assert cache.embed_calls == 1


def test_still_face_reuses_the_decision(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher, refresh_s=0.4, move_px=18.0)
    frame = np.zeros((480, 640, 3), np.uint8)
    cache.identify(frame, [FakeFace(100, 100, 200, 200)], now=0.0)
    # Same face, 1 px of detector jitter, well inside both limits.
    out = cache.identify(frame, [FakeFace(101, 100, 201, 200)], now=0.10)
    assert cache.embed_calls == 1
    assert out[0].reused is True
    assert out[0].label == "Kennedy"


def test_moved_face_is_re_embedded(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher, refresh_s=0.4, move_px=18.0)
    frame = np.zeros((480, 640, 3), np.uint8)
    cache.identify(frame, [FakeFace(100, 100, 200, 200)], now=0.0)
    cache.identify(frame, [FakeFace(180, 100, 280, 200)], now=0.10)  # 80 px jump
    assert cache.embed_calls == 2


def test_stale_entry_is_refreshed_even_if_still(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher, refresh_s=0.4, move_px=18.0)
    frame = np.zeros((480, 640, 3), np.uint8)
    face = [FakeFace(100, 100, 200, 200)]
    cache.identify(frame, face, now=0.0)
    cache.identify(frame, face, now=0.5)  # same box, but older than refresh_s
    assert cache.embed_calls == 2


def test_lost_face_forgets_its_entry(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher)
    frame = np.zeros((480, 640, 3), np.uint8)
    cache.identify(frame, [FakeFace(100, 100, 200, 200)], now=0.0)
    cache.identify(frame, [], now=0.1)  # face left the frame
    assert cache._entries == []
    cache.identify(frame, [FakeFace(100, 100, 200, 200)], now=0.2)
    assert cache.embed_calls == 2  # a returning face is embedded again


def test_two_faces_keep_separate_identities(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher, refresh_s=0.4, move_px=18.0)
    frame = np.zeros((480, 640, 3), np.uint8)
    left = FakeFace(40, 100, 140, 200)
    right = FakeFace(420, 100, 520, 200)
    out = cache.identify(frame, [left, right], now=0.0)
    assert len(out) == 2 and cache.embed_calls == 2
    # Jitter both faces a little: each must map to its own cached entry.
    out = cache.identify(
        frame, [FakeFace(42, 101, 142, 201), FakeFace(418, 99, 518, 199)], now=0.1
    )
    assert cache.embed_calls == 2
    assert all(not r.reused for r in out) is False  # both were reused
    assert len(cache._entries) == 2


def test_reset_forces_a_fresh_decision(matcher):
    embedder = FakeEmbedder()
    cache = RecognitionCache(embedder, matcher)
    frame = np.zeros((480, 640, 3), np.uint8)
    face = [FakeFace(100, 100, 200, 200)]
    cache.identify(frame, face, now=0.0)
    cache.reset()
    cache.identify(frame, face, now=0.1)
    assert cache.embed_calls == 2


def test_unknown_face_is_labelled_unknown(matcher):
    class Impostor(FakeEmbedder):
        def embed(self, aligned_bgr):
            self.calls += 1
            from src.embed import EmbeddingResult

            vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cosine 0
            return EmbeddingResult(embedding=vector, norm_before=1.0, dim=4)

    cache = RecognitionCache(Impostor(), matcher)
    out = cache.identify(np.zeros((480, 640, 3), np.uint8), [FakeFace(100, 100, 200, 200)], now=0.0)
    assert out[0].label == "Unknown"
    assert out[0].result.accepted is False


def test_expression_pairing_is_independent_of_identity_cache():
    state = FaceExpression(0, 0.3, 0.3, 0.8, box=(100, 100, 200, 200), smiling=True)
    assert match_expression([state], (105, 105, 195, 195)) is state
