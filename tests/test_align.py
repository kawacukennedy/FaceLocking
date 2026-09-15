"""Tests for the ArcFace 5-point affine alignment math."""

import numpy as np
import pytest

from src.haar_5pt import (
    ARCFACE_TEMPLATE_112,
    align_face_5pt,
    estimate_norm_5pt,
    bbox_from_5pt,
)


def _synthetic_landmarks(seed: int = 0, scale: float = 1.0, angle_deg: float = 0.0):
    """Generate landmarks from the template with rotation and scale applied."""
    rng = np.random.default_rng(seed)
    kps = ARCFACE_TEMPLATE_112.copy().astype(np.float32) / 112.0
    kps -= kps.mean(axis=0)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    kps = kps @ rotation.T * scale
    kps += np.array([0.5, 0.5], dtype=np.float32)
    kps += rng.normal(0, 0.002, kps.shape).astype(np.float32)
    return kps.astype(np.float32)


def test_estimate_norm_5pt_maps_back_to_template_default_size():
    kps = _synthetic_landmarks(seed=1, scale=1.1, angle_deg=15)
    matrix = estimate_norm_5pt(kps, out_size=(112, 112))
    warped = cv2_transform_points(kps, matrix)
    error = np.abs(warped - ARCFACE_TEMPLATE_112).max()
    assert error < 2.0


def test_estimate_norm_5pt_scales_to_other_output_sizes():
    kps = _synthetic_landmarks(seed=2)
    matrix = estimate_norm_5pt(kps, out_size=(224, 224))
    warped = cv2_transform_points(kps, matrix)
    expected = ARCFACE_TEMPLATE_112 * 2.0
    error = np.abs(warped - expected).max()
    assert error < 3.0


def test_align_face_5pt_output_shape_and_dtype():
    frame = np.full((200, 300, 3), 120, dtype=np.uint8)
    kps = _synthetic_landmarks(seed=3, scale=0.3, angle_deg=-10)
    kps *= np.array([300.0, 200.0], dtype=np.float32)
    kps += np.array([0.0, 0.0], dtype=np.float32)
    kps[:, 1] *= 112.0 / 112.0  # keep coordinates within reason for the warp
    aligned, matrix = align_face_5pt(frame, kps, out_size=(112, 112))
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32


def test_bbox_from_5pt_is_valid_xyxy():
    kps = _synthetic_landmarks(seed=4)
    box = bbox_from_5pt(kps)
    assert box[0] < box[2]
    assert box[1] < box[3]


def cv2_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    import cv2

    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ matrix.T).astype(np.float32)