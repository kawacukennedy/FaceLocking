"""Smile and blink detection from MediaPipe FaceMesh landmarks.

Two classic, well documented signals are used, both scale invariant so they do
not depend on how far away the person sits:

* **Blink** — the *Eye Aspect Ratio* (EAR) of Soukupova et al.:

      EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

  The eyes are open while the ratio stays high and it collapses towards zero
  when the eyelids close. A blink is reported when the ratio stays below the
  threshold for a couple of consecutive frames, with a short refractory period
  so one blink is not counted twice.

* **Smile** — a weighted combination of two mouth features, normalized by the
  inter-ocular distance: mouth width (primary) and corner lift, since smiles
  raise the mouth corners (secondary). Lip aperture gates the result, so an
  open mouth (talking, yawning) is not scored as a smile.

The first ``calibrate_frames`` frames with a visible face define the person's
neutral baseline, so the same thresholds work for any user, skin tone or
lighting. Press ``c`` in the demo to recalibrate.

Run::

    python -m src.expressions
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe FaceMesh indices (468-point topology).
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
INNER_UPPER_LIP, INNER_LOWER_LIP = 13, 14
IOD_LEFT, IOD_RIGHT = 33, 263  # inter-ocular corners, used as the scale unit

# Blink: a fraction of the calibrated neutral EAR, with an absolute floor.
DEFAULT_BLINK_THRESHOLD = 0.20
BLINK_THRESHOLD_RATIO = 0.68

# Smile: bounded gains measured against the neutral mouth, in inter-ocular
# units. Width is the primary cue, corner lift the secondary one, and lip
# aperture gates the result so an open mouth is not scored as a smile.
WIDTH_GAIN_SPAN = 0.12  # +12% mouth width saturates the width gain
LIFT_GAIN_SPAN = 0.05  # +0.05 iod corner lift saturates the lift gain
OPEN_GATE_SPAN = 1.5  # lip gap this many times neutral fully suppresses
WIDTH_WEIGHT = 0.55
LIFT_WEIGHT = 0.45
NEUTRAL_SMILE_FLOOR = 0.55  # score above which a smile is reported

# Used until the person's neutral face has been measured (values in iod units).
FALLBACK_NEUTRAL = {"ear": 0.30, "width": 1.05, "open": 0.05, "lift": 0.01}


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Eye Aspect Ratio for one eye given six ``(x, y)`` landmarks in order."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal <= 1e-6:
        return 1.0
    return vertical / horizontal


@dataclass
class FaceExpression:
    """Per-face expression state for one video frame."""

    face_index: int
    ear: float
    neutral_ear: float
    smile_score: float
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    smiling: bool = False
    blinking: bool = False
    blink_event: bool = False
    blink_count: int = 0
    smile_count: int = 0
    calibrated: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "ear": self.ear,
            "smile_score": self.smile_score,
            "smiling": float(self.smiling),
            "blinking": float(self.blinking),
            "blink_count": float(self.blink_count),
        }


@dataclass
class _FaceHistory:
    """Per-face temporal state (a face keeps its id between frames)."""

    center: Tuple[float, float] = (0.5, 0.5)  # normalized landmark centre
    blink_frames: int = 0
    last_blink_at: float = 0.0
    blink_count: int = 0
    smile_count: int = 0
    smiling: bool = False
    ear_smooth: Optional[float] = None
    smile_smooth: Optional[float] = None
    last_seen: float = 0.0


class ExpressionDetector:
    """Detects smiles and blinks for every face in a frame.

    Args:
        blink_threshold: Absolute EAR threshold; ignored once calibrated.
        smile_threshold: Smile score in ``[0, 1]`` above which a smile is set.
        calibrate_frames: Frames used to learn the neutral face.
        blink_consecutive: Consecutive closed-eye frames required for a blink.
        blink_refractory_s: Minimum seconds between two counted blinks.
        smooth_alpha: EMA factor for the reported EAR / smile score.
        max_num_faces: Maximum faces evaluated per frame.
        match_radius: How far (in normalized frame units) a face may move
            between frames and still keep its per-face counters.
    """

    def __init__(
        self,
        blink_threshold: float = DEFAULT_BLINK_THRESHOLD,
        smile_threshold: float = NEUTRAL_SMILE_FLOOR,
        calibrate_frames: int = 45,
        blink_consecutive: int = 2,
        blink_refractory_s: float = 0.35,
        smooth_alpha: float = 0.55,
        max_num_faces: int = 5,
        static_image_mode: bool = False,
        mesh: Optional[Any] = None,
        match_radius: float = 0.15,
    ) -> None:
        self.blink_threshold = float(blink_threshold)
        self.smile_threshold = float(smile_threshold)
        self.calibrate_frames = int(calibrate_frames)
        self.blink_consecutive = max(1, int(blink_consecutive))
        self.blink_refractory_s = float(blink_refractory_s)
        self.smooth_alpha = float(smooth_alpha)
        self.max_num_faces = int(max_num_faces)
        self.static_image_mode = bool(static_image_mode)
        self.match_radius = float(match_radius)
        # The FaceMesh is created on first use so the scoring logic can also be
        # driven from landmarks produced elsewhere (or from tests).
        self._mesh = mesh
        self._closed = False

        self._histories: Dict[int, _FaceHistory] = {}
        self._next_id = 0
        self._calib_target: Optional[int] = calibrate_frames
        self._calib_samples: List[np.ndarray] = []
        self._neutral: Optional[Dict[str, float]] = None
        self.total_blinks = 0
        self.total_smiles = 0
        self.last_blink_at: float = 0.0
        self.last_smile_at: float = 0.0

    # ------------------------------------------------------------------ setup
    @property
    def calibrated(self) -> bool:
        return self._neutral is not None

    @property
    def calibration_progress(self) -> Tuple[int, int]:
        """``(frames_collected, frames_required)`` for the neutral baseline."""
        target = self._calib_target if self._calib_target is not None else 0
        return len(self._calib_samples), target

    def _get_mesh(self):
        """Return the FaceMesh, building it on first use."""
        if self._mesh is None:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                refine_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        return self._mesh

    def reset(self) -> None:
        """Forget the neutral baseline and per-face history."""
        self._calib_samples.clear()
        self._calib_target = self.calibrate_frames
        self._neutral = None
        self._histories.clear()
        self.total_blinks = 0
        self.total_smiles = 0

    def close(self) -> None:
        if not self._closed and self._mesh is not None:
            self._mesh.close()
        self._closed = True

    def __enter__(self) -> "ExpressionDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- internals
    def _ema(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return value
        return previous + self.smooth_alpha * (value - previous)

    def _match_faces(self, centers: List[Tuple[float, float]]) -> List[int]:
        """Assign each face centre to an existing face id (or a new one).

        Matching uses normalized landmark centres rather than detector boxes: the
        landmarks are the measurement itself, so the association is correct
        whether or not the caller supplied detection boxes, and it survives a
        face moving smoothly between frames.
        """
        assignments: List[int] = []
        used: set = set()
        for cx, cy in centers:
            best_id, best_dist = None, self.match_radius
            for face_id, hist in self._histories.items():
                if face_id in used:
                    continue
                dist = math.hypot(cx - hist.center[0], cy - hist.center[1])
                if dist < best_dist:
                    best_id, best_dist = face_id, dist
            if best_id is None:
                best_id = self._new_history()
            used.add(best_id)
            assignments.append(best_id)
        return assignments

    def _new_history(self, now: float = 0.0) -> int:
        face_id = self._next_id
        self._next_id += 1
        self._histories[face_id] = _FaceHistory(last_seen=now)
        return face_id

    # ---------------------------------------------------------------- metrics
    @staticmethod
    def _face_metrics(lm: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        """Return ``(ear, mouth_width, mouth_open, corner_lift, iod)``."""
        pts = np.asarray([(p.x, p.y) for p in lm], dtype=np.float32)
        iod = _dist(pts[IOD_LEFT], pts[IOD_RIGHT])
        if iod <= 1e-6:
            return None
        ear = 0.5 * (
            eye_aspect_ratio(pts[list(LEFT_EYE)]) + eye_aspect_ratio(pts[list(RIGHT_EYE)])
        )
        width = _dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / iod
        lip_gap = _dist(pts[INNER_UPPER_LIP], pts[INNER_LOWER_LIP]) / iod
        corner_y = 0.5 * (pts[MOUTH_LEFT][1] + pts[MOUTH_RIGHT][1])
        lip_y = 0.5 * (pts[INNER_UPPER_LIP][1] + pts[INNER_LOWER_LIP][1])
        lift = (corner_y - lip_y) / iod
        return ear, width, lip_gap, lift, iod

    @staticmethod
    def _landmark_center(lm: Sequence[Any]) -> Tuple[float, float]:
        """Normalized centre of a face, taken from its core landmarks."""
        core = (IOD_LEFT, IOD_RIGHT, INNER_UPPER_LIP, INNER_LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)
        xs = [lm[i].x for i in core]
        ys = [lm[i].y for i in core]
        return 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))

    @staticmethod
    def _landmark_box(lm: Sequence[Any], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Pixel bounding box covering a face's full landmark set."""
        height, width = frame_shape
        xs = [p.x * width for p in lm]
        ys = [p.y * height for p in lm]
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )

    def _boxes_for(
        self,
        faces: Sequence[Tuple[int, Sequence[Any], Tuple[float, ...]]],
        frame_shape: Tuple[int, int],
        face_boxes: Optional[np.ndarray],
    ) -> List[Tuple[int, int, int, int]]:
        """Pixel box per face: the caller's detection box if it matches, else the
        landmark extent.

        Detection boxes give a snug box for drawing, while the landmark extent is
        always available, so reporting never depends on the caller.
        """
        height, width = frame_shape
        boxes = (
            np.asarray(face_boxes, dtype=np.float32)
            if face_boxes is not None and len(face_boxes) > 0
            else None
        )
        used: set = set()
        out: List[Tuple[int, int, int, int]] = []
        for face_id, lm, _metrics in faces:
            cx, cy = self._histories[face_id].center
            best: Optional[Tuple[float, int, np.ndarray]] = None
            if boxes is not None:
                for j, box in enumerate(boxes):
                    if j in used:
                        continue
                    bcx = 0.5 * (float(box[0]) + float(box[2])) / max(1, width)
                    bcy = 0.5 * (float(box[1]) + float(box[3])) / max(1, height)
                    dist = math.hypot(cx - bcx, cy - bcy)
                    if dist < self.match_radius and (best is None or dist < best[0]):
                        best = (dist, j, box)
            if best is not None:
                used.add(best[1])
                out.append(tuple(int(round(float(v))) for v in best[2]))
            else:
                out.append(self._landmark_box(lm, frame_shape))
        return out

    def _update_calibration(self, metrics: np.ndarray) -> None:
        if self._calib_target is None:
            return
        self._calib_samples.append(metrics)
        if len(self._calib_samples) < self._calib_target:
            return
        data = np.asarray(self._calib_samples, dtype=np.float32)
        self._neutral = {
            "ear": float(np.mean(data[:, 0])),
            "width": float(np.mean(data[:, 1])),
            "open": float(np.mean(data[:, 2])),
            "lift": float(np.mean(data[:, 3])),
        }
        self._calib_target = None

    def _smile_score(self, width: float, lip_gap: float, lift: float) -> float:
        """Map mouth geometry to a bounded ``[0, 1]`` smile score.

        A smile is a *wider* mouth whose *corners are raised*, with the lips
        still closed. Each feature becomes a bounded gain relative to the
        person's neutral baseline, and an open mouth gates the result down, so
        talking or yawning is not mistaken for a smile.
        """
        n = self._neutral or FALLBACK_NEUTRAL
        width_gain = _clamp01((width - n["width"]) / max(1e-4, n["width"] * WIDTH_GAIN_SPAN))
        lift_gain = _clamp01((lift - n["lift"]) / LIFT_GAIN_SPAN)
        open_ratio = lip_gap / max(1e-4, n["open"])
        close_gate = _clamp01(1.0 - (open_ratio - 1.0) / OPEN_GATE_SPAN)
        return float((WIDTH_WEIGHT * width_gain + LIFT_WEIGHT * lift_gain) * close_gate)

    def _blink_threshold(self, neutral_ear: float) -> float:
        if neutral_ear <= 1e-6:
            return self.blink_threshold
        adaptive = neutral_ear * BLINK_THRESHOLD_RATIO
        return float(max(0.10, min(adaptive, neutral_ear * 0.85)))

    # ------------------------------------------------------------------- API
    def update(
        self,
        frame_bgr: np.ndarray,
        face_boxes: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[FaceExpression]:
        """Evaluate every face in ``frame_bgr``.

        Args:
            frame_bgr: BGR frame.
            face_boxes: Optional ``(N, 4)`` ``(x1, y1, x2, y2)`` boxes from the
                detector; used to keep per-face blink counters stable.
            timestamp: Optional seconds value (defaults to ``time.time()``).
        """
        now = time.time() if timestamp is None else float(timestamp)
        result = self._get_mesh().process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return []
        landmark_sets = [face_list.landmark for face_list in result.multi_face_landmarks]
        return self.update_from_landmarks(
            landmark_sets,
            frame_shape=frame_bgr.shape[:2],
            face_boxes=face_boxes,
            timestamp=now,
        )

    def update_from_landmarks(
        self,
        landmark_sets: Sequence[Sequence[Any]],
        frame_shape: Tuple[int, int],
        face_boxes: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[FaceExpression]:
        """Score already-computed FaceMesh landmarks.

        Split out from :meth:`update` so the expression logic can be exercised
        with synthetic landmarks (tests) or with landmarks produced by another
        detector, without running MediaPipe again.

        Args:
            landmark_sets: One sequence of 468 landmarks per face, as returned
                by ``FaceMesh.process`` (points expose ``.x`` / ``.y`` in
                normalized frame coordinates).
            frame_shape: ``(height, width)`` in pixels, used when no detection
                boxes are supplied.
            face_boxes: Optional ``(N, 4)`` ``(x1, y1, x2, y2)`` boxes; keeps
                per-face counters attached to the same person between frames.
            timestamp: Optional seconds value (defaults to ``time.time()``).
        """
        now = time.time() if timestamp is None else float(timestamp)
        height, width = frame_shape
        if not landmark_sets:
            return []

        # Pass 1: per-face geometry plus the identity match.
        centers = [self._landmark_center(lm) for lm in landmark_sets]
        ids = self._match_faces(centers)
        faces: List[Tuple[int, Any, Tuple[float, float, float, float]]] = []
        for i, lm in enumerate(landmark_sets):
            metrics = self._face_metrics(lm)
            if metrics is None:
                continue
            face_id = ids[i]
            hist = self._histories[face_id]
            hist.center = centers[i]
            hist.last_seen = now
            faces.append((face_id, lm, metrics))

        if not faces:
            return []

        # Calibration learns one neutral face, from the largest one in view.
        boxes = self._boxes_for(faces, frame_shape, face_boxes)
        largest = max(
            range(len(faces)),
            key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        )
        ear, m_width, m_open, m_lift, _iod = faces[largest][2]
        self._update_calibration(
            np.asarray([ear, m_width, m_open, m_lift], dtype=np.float32)
        )

        out: List[FaceExpression] = []
        for i, (face_id, lm, metrics) in enumerate(faces):
            ear, m_width, m_open, m_lift, _iod = metrics
            hist = self._histories[face_id]
            face_box = boxes[i]

            # ---- blink -------------------------------------------------
            neutral_ear = self._neutral["ear"] if self._neutral else 0.0
            threshold = self._blink_threshold(neutral_ear) if self._neutral else self.blink_threshold
            hist.ear_smooth = self._ema(hist.ear_smooth, ear)
            ear_e = hist.ear_smooth
            closed = ear_e < threshold
            if closed:
                hist.blink_frames += 1
            else:
                hist.blink_frames = 0
            blink_event = (
                hist.blink_frames == self.blink_consecutive
                and (now - hist.last_blink_at) >= self.blink_refractory_s
            )
            if blink_event:
                hist.blink_frames = 0
                hist.last_blink_at = now
                hist.blink_count += 1
                self.total_blinks += 1
                self.last_blink_at = now

            # ---- smile -------------------------------------------------
            score = self._smile_score(m_width, m_open, m_lift)
            hist.smile_smooth = self._ema(hist.smile_smooth, score)
            smiling = hist.smile_smooth >= self.smile_threshold
            if smiling and not hist.smiling:
                hist.smile_count += 1
                self.total_smiles += 1
                self.last_smile_at = now
            hist.smiling = smiling

            out.append(
                FaceExpression(
                    face_index=face_id,
                    ear=float(ear_e),
                    neutral_ear=float(neutral_ear),
                    smile_score=float(hist.smile_smooth),
                    box=face_box,
                    smiling=bool(smiling),
                    blinking=bool(closed),
                    blink_event=bool(blink_event),
                    blink_count=hist.blink_count,
                    smile_count=hist.smile_count,
                    calibrated=self.calibrated,
                )
            )
        return out


# --------------------------------------------------------------------- demo
def main() -> None:
    from . import ui
    from .camera import Camera

    # The expression pass already returns a box per face, so this demo needs no
    # second detector: one FaceMesh run per frame.
    expressions = ExpressionDetector(max_num_faces=3)

    with Camera() as camera, expressions:
        print(camera.describe())
        print("Smile/blink detector. c=calibrate, r=reset counts, f=flip, q=quit")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            vis = frame.copy()
            ui.vignette(vis)

            states = expressions.update(frame)

            for state in states:
                x1, y1, x2, y2 = state.box
                color = ui.ORANGE if state.blinking else (ui.GREEN if state.smiling else ui.WHITE)
                ui.draw_corner_box(vis, (x1, y1, x2 - x1, y2 - y1), color=color)
                ear_ratio = state.ear / state.neutral_ear if state.neutral_ear > 0 else 1.0
                ui.draw_meter(vis, min(1.0, ear_ratio / 1.4), (x1, y2 + 30), width=150,
                              color=ui.CYAN, label=f"EAR {state.ear:.2f}")
                ui.draw_meter(vis, state.smile_score, (x1, y2 + 66), width=150,
                              color=ui.GREEN if state.smiling else ui.GRAY,
                              label=f"smile {state.smile_score:.2f}")
                if state.blinking:
                    ui.draw_badge(vis, "BLINK", (x1, y1 - 8), color=ui.ORANGE, anchor="bottom-left")
                elif state.smiling:
                    ui.draw_badge(vis, "SMILING", (x1, y1 - 8), color=ui.GREEN, anchor="bottom-left")
                if state.blink_count:
                    ui.draw_badge(vis, f"blinks {state.blink_count}", (x2, y1 - 8),
                                  color=ui.GRAY, anchor="bottom-right")

            lines = [
                ("BLINK / SMILE DETECTOR", ui.WHITE),
                (f"faces={len(states)}   blinks={expressions.total_blinks}   smiles={expressions.total_smiles}", ui.CYAN),
                ("calibrated" if expressions.calibrated else "calibrating neutral face...",
                 ui.GREEN if expressions.calibrated else ui.YELLOW),
            ]
            if not expressions.calibrated:
                have, need_total = expressions.calibration_progress
                lines.append((f"{max(0, need_total - have)} frames left - stay relaxed, look at the camera", ui.GRAY))
            elif any(s.blink_event for s in states):
                lines.append(("blink detected!", ui.ORANGE))
            elif any(s.smiling for s in states):
                lines.append(("smiling", ui.GREEN))
            else:
                lines.append(("neutral", ui.GRAY))
            lines.append(("c=calibrate  r=reset counts  f=flip  q=quit", ui.GRAY))
            ui.draw_hud(vis, lines)
            cv2.imshow("expressions", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                expressions.reset()
                print("[expressions] recalibrating neutral face...")
            elif key == ord("r"):
                expressions.total_blinks = 0
                expressions.total_smiles = 0
            elif key == ord("f"):
                camera.toggle_flip()


if __name__ == "__main__":
    main()
