"""Smile and blink logic tests.

The expression detector is driven with synthetic FaceMesh landmarks so the
geometry, calibration, blink debouncing and smile scoring are all testable
without a camera or MediaPipe inference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from src.expressions import (
    ExpressionDetector,
    eye_aspect_ratio,
    IOD_LEFT,
    IOD_RIGHT,
    INNER_LOWER_LIP,
    INNER_UPPER_LIP,
    LEFT_EYE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    RIGHT_EYE,
)

LANDMARK_COUNT = 468
FRAME_SHAPE = (480, 640)


class Point:
    """Minimal stand-in for a MediaPipe normalized landmark."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


def make_landmarks(
    ear: float = 0.30,
    mouth_width: float = 1.05,
    mouth_open: float = 0.04,
    corner_lift: float = 0.02,
    iod_px: float = 100.0,
    center: Sequence[float] = (0.5, 0.5),
) -> list:
    """Build a synthetic 468-point mesh with the requested metrics.

    Eye points are placed on a canonical rectangle so
    ``eye_aspect_ratio`` returns exactly ``ear``; the mouth points reproduce
    the requested width, gap and corner lift in units of ``iod_px``.
    """
    pts = [Point(center[0], center[1]) for _ in range(LANDMARK_COUNT)]
    height, width = FRAME_SHAPE
    cx, cy = center[0] * width, center[1] * height

    def place(index: int, x: float, y: float) -> None:
        pts[index] = Point(x / width, y / height)

    # Inter-ocular corners define the scale unit.
    place(IOD_LEFT, cx - iod_px / 2.0, cy)
    place(IOD_RIGHT, cx + iod_px / 2.0, cy)

    # One eye: p1 (outer) .. p4 (inner) horizontal span, with p2/p6 and p3/p5
    # vertically aligned so the EAR is exactly ``ear``: (2 * 2 * half) / (2 * span).
    for eye, eye_cx in ((RIGHT_EYE, cx - iod_px / 2.0), (LEFT_EYE, cx + iod_px / 2.0)):
        p1, p2, p3, p4, p5, p6 = eye
        span = 0.32 * iod_px
        half = ear * span / 2.0
        x1 = eye_cx - 0.5 * span
        x4 = eye_cx + 0.5 * span
        y_mid = cy - 0.10 * iod_px
        place(p1, x1, y_mid)
        place(p2, x1, y_mid - half)
        place(p3, x4, y_mid - half)
        place(p4, x4, y_mid)
        place(p5, x4, y_mid + half)
        place(p6, x1, y_mid + half)

    # Mouth: corners lifted relative to the lip centre (smile raises corners).
    mw = mouth_width * iod_px
    gap = mouth_open * iod_px
    lip_cy = cy + 0.75 * iod_px
    place(MOUTH_LEFT, cx - mw / 2.0, lip_cy - corner_lift * iod_px)
    place(MOUTH_RIGHT, cx + mw / 2.0, lip_cy - corner_lift * iod_px)
    place(INNER_UPPER_LIP, cx, lip_cy - gap / 2.0)
    place(INNER_LOWER_LIP, cx, lip_cy + gap / 2.0)
    return pts


def feed(
    detector: ExpressionDetector,
    landmarks,
    count: int = 1,
    t0: float = 0.0,
    dt: float = 0.05,
    boxes=None,
):
    """Run ``count`` identical frames through the detector, newest result last."""
    result = []
    t = t0
    for _ in range(count):
        result = detector.update_from_landmarks(
            [landmarks], FRAME_SHAPE, face_boxes=boxes, timestamp=t
        )
        t += dt
    return result[0] if result else None


# --------------------------------------------------------------------- EAR
def test_eye_aspect_ratio_geometry():
    # Square-ish eye: two vertical halves over a 2 * span denominator.
    eye = np.array(
        [
            [0.0, 0.0],  # p1 outer
            [0.25, -0.1],  # p2 upper
            [0.75, -0.1],  # p3 upper
            [1.0, 0.0],  # p4 inner
            [0.75, 0.1],  # p5 lower
            [0.25, 0.1],  # p6 lower
        ]
    )
    assert eye_aspect_ratio(eye) == pytest.approx(0.2, abs=1e-6)


def test_ear_is_low_when_eyes_close():
    open_ear = ExpressionDetector._face_metrics(make_landmarks(ear=0.30))
    shut_ear = ExpressionDetector._face_metrics(make_landmarks(ear=0.05))
    assert open_ear[0] > 0.25
    assert shut_ear[0] < 0.10


def test_ear_is_scale_invariant():
    near = ExpressionDetector._face_metrics(make_landmarks(iod_px=100.0))
    far = ExpressionDetector._face_metrics(make_landmarks(iod_px=40.0))
    assert near[0] == pytest.approx(far[0], rel=0.05)


# ------------------------------------------------------------------ blink
def test_single_blink_is_counted_once():
    det = ExpressionDetector(calibrate_frames=5, blink_consecutive=2, smooth_alpha=1.0)
    feed(det, make_landmarks(ear=0.30), count=6)  # neutral calibration
    assert det.calibrated

    state = feed(det, make_landmarks(ear=0.30), count=1, t0=1.0)
    assert state is not None and not state.blinking

    closed = feed(det, make_landmarks(ear=0.05), count=2, t0=1.2)
    assert closed.blinking
    assert det.total_blinks == 1

    # Eyes open again; a lingering close inside the refractory window is not
    # counted as a second blink.
    feed(det, make_landmarks(ear=0.30), count=3, t0=1.4)
    feed(det, make_landmarks(ear=0.05), count=2, t0=1.52)
    assert det.total_blinks == 1

    # A later, separate blink is counted.
    feed(det, make_landmarks(ear=0.30), count=2, t0=2.2)
    feed(det, make_landmarks(ear=0.05), count=2, t0=2.6)
    assert det.total_blinks == 2


def test_blink_refractory_suppresses_rapid_double_count():
    det = ExpressionDetector(calibrate_frames=3, blink_consecutive=1, smooth_alpha=1.0)
    feed(det, make_landmarks(ear=0.30), count=4)
    feed(det, make_landmarks(ear=0.05), count=1, t0=1.0)
    feed(det, make_landmarks(ear=0.30), count=1, t0=1.05)
    # Second close only 50 ms later: inside the refractory window.
    feed(det, make_landmarks(ear=0.05), count=1, t0=1.10)
    assert det.total_blinks == 1


def test_one_closed_frame_does_not_count_as_blink():
    det = ExpressionDetector(calibrate_frames=3, blink_consecutive=2, smooth_alpha=1.0)
    feed(det, make_landmarks(ear=0.30), count=4)
    feed(det, make_landmarks(ear=0.05), count=1, t0=1.0)
    assert det.total_blinks == 0


def test_blink_threshold_scales_with_neutral_ear():
    det = ExpressionDetector(calibrate_frames=4, smooth_alpha=1.0)
    feed(det, make_landmarks(ear=0.32), count=5)
    neutral = det._neutral["ear"]
    threshold = det._blink_threshold(neutral)
    assert threshold == pytest.approx(neutral * 0.68, rel=0.05)
    # An eye ratio that is 20% of neutral is unambiguously shut.
    assert neutral * 0.2 < threshold


# ------------------------------------------------------------------ smile
def test_smile_rises_above_neutral():
    det = ExpressionDetector(calibrate_frames=8, smooth_alpha=1.0)
    feed(det, make_landmarks(mouth_width=1.00, mouth_open=0.05, corner_lift=0.01), count=9)
    assert det.calibrated
    neutral_state = feed(det, make_landmarks(mouth_width=1.00, mouth_open=0.05, corner_lift=0.01), count=1, t0=1.0)
    smile_state = feed(det, make_landmarks(mouth_width=1.30, mouth_open=0.02, corner_lift=0.10), count=2, t0=1.2)
    assert smile_state.smile_score > neutral_state.smile_score
    assert smile_state.smiling
    assert not neutral_state.smiling
    assert det.total_smiles >= 1


def test_open_mouth_is_not_a_smile():
    det = ExpressionDetector(calibrate_frames=8, smooth_alpha=1.0)
    feed(det, make_landmarks(mouth_width=1.00, mouth_open=0.05, corner_lift=0.01), count=9)
    talking = feed(det, make_landmarks(mouth_width=1.25, mouth_open=0.35, corner_lift=0.05), count=3, t0=1.0)
    assert talking.smile_score < 0.55
    assert not talking.smiling


def test_smile_score_is_bounded():
    det = ExpressionDetector(calibrate_frames=3, smooth_alpha=1.0)
    feed(det, make_landmarks(), count=4)
    state = feed(
        det,
        make_landmarks(mouth_width=2.0, mouth_open=0.001, corner_lift=0.5),
        count=2,
        t0=1.0,
    )
    assert 0.0 <= state.smile_score <= 1.0


# ------------------------------------------------------- session behaviour
def test_reset_recalibrates_and_clears_counts():
    det = ExpressionDetector(calibrate_frames=3, blink_consecutive=1, smooth_alpha=1.0)
    feed(det, make_landmarks(ear=0.30), count=4)
    feed(det, make_landmarks(ear=0.05), count=1, t0=1.0)
    assert det.total_blinks == 1
    det.reset()
    assert not det.calibrated
    assert det.total_blinks == 0
    assert det.calibration_progress == (0, 3)


def test_face_id_is_stable_across_frames_with_boxes():
    det = ExpressionDetector(calibrate_frames=3, smooth_alpha=1.0)
    # A detection box around the synthetic face (see FRAME_SHAPE).
    boxes = np.array([[250, 220, 390, 320]], dtype=np.float32)
    first = feed(det, make_landmarks(), count=1, boxes=boxes)
    assert first.box == (250, 220, 390, 320)
    again = None
    for t in (0.05, 0.10, 0.15):
        # Small jitter, as a real detector box would have.
        jittered = boxes + np.array([2, -1, 2, -1], dtype=np.float32)
        again = feed(det, make_landmarks(), count=1, t0=t, boxes=jittered)
    assert first.face_index == again.face_index
    assert again.box == (252, 219, 392, 319)


def test_box_derived_from_landmarks_without_detector_boxes():
    det = ExpressionDetector(calibrate_frames=2, smooth_alpha=1.0)
    state = feed(det, make_landmarks(iod_px=100.0), count=1)
    x1, y1, x2, y2 = state.box
    assert 0 <= x1 < x2 <= FRAME_SHAPE[1]
    assert 0 <= y1 < y2 <= FRAME_SHAPE[0]
    # The mesh is centred, so the box must straddle the frame centre.
    assert x1 < FRAME_SHAPE[1] / 2 < x2


def test_no_landmarks_yields_no_states():
    det = ExpressionDetector()
    assert det.update_from_landmarks([], FRAME_SHAPE) == []


# ------------------------------------------- pairing with detector boxes
def test_recognize_pairs_states_with_detections_by_overlap():
    from src.expressions import FaceExpression
    from src.recognize import match_expression

    left = FaceExpression(0, 0.30, 0.30, 0.20, box=(100, 100, 200, 200))
    right = FaceExpression(1, 0.30, 0.30, 0.90, box=(400, 300, 500, 400), smiling=True)
    states = [left, right]

    assert match_expression(states, (110, 110, 190, 190)) is left
    assert match_expression(states, (405, 305, 495, 395)) is right
    # A face that is not in the expression result has no state to pair with.
    assert match_expression(states, (10, 10, 60, 60)) is None
    assert match_expression([], (110, 110, 190, 190)) is None
