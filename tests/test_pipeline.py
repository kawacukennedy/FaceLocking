"""End-to-end pipeline test on a real face sample image.

Flow under test: face detection -> 5-point landmarks -> alignment -> embedding.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.embed import ArcFaceEmbedderONNX
from src.haar_5pt import Haar5ptDetector, align_face_5pt
from tests import SAMPLE_FACE

SAMPLE = Path(SAMPLE_FACE)

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason="sample face image not present (tests/data/sample_face.jpg)",
)


def test_detector_finds_a_face_and_five_landmarks():
    detector = Haar5ptDetector(min_size=(50, 50))
    frame = cv2.imread(str(SAMPLE))
    assert frame is not None
    faces = detector.detect(frame, max_faces=1)
    assert faces, "expected at least one detection on the sample image"
    face = faces[0]
    assert face.kps.shape == (5, 2)
    # landmark ordering contract: left eye is to the viewer's left of right eye
    assert face.kps[0, 0] < face.kps[1, 0]


def test_alignment_produces_canonical_crop():
    detector = Haar5ptDetector(min_size=(50, 50))
    frame = cv2.imread(str(SAMPLE))
    face = detector.detect(frame, max_faces=1)[0]
    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
    assert aligned.shape == (112, 112, 3)
    assert float(aligned.mean()) > 1.0


@pytest.mark.skipif(
    not (Path("models/embedder_arcface.onnx").exists()
         and Path("models/embedder_arcface.onnx").stat().st_size > 1_000_000),
    reason="ArcFace ONNX model not present; run scripts/download_models.sh",
)
def test_full_pipeline_embedding_is_stable():
    detector = Haar5ptDetector(min_size=(50, 50))
    embedder = ArcFaceEmbedderONNX()
    frame = cv2.imread(str(SAMPLE))
    faces = detector.detect(frame, max_faces=1)
    assert faces
    aligned, _ = align_face_5pt(frame, faces[0].kps, out_size=(112, 112))
    first = embedder.embed(aligned).embedding
    second = embedder.embed(aligned).embedding
    sim = float(np.dot(first, second))
    assert sim > 0.99