"""Core face pipeline: Haar detection, 5-point landmarks, and ArcFace alignment.

This module provides the shared building blocks used by every stage of the
project:

* :class:`Haar5ptDetector` locates faces with an OpenCV Haar cascade and
  extracts five facial landmarks (left eye, right eye, nose tip, left mouth
  corner, right mouth corner) with MediaPipe FaceMesh. Face order is strict
  and stable: ``[left_eye, right_eye, nose, left_mouth, right_mouth]``.
* :func:`estimate_norm_5pt` builds a similarity transform that maps those five
  points onto the standard ArcFace 112x112 template.
* :func:`align_face_5pt` warps a face crop into a canonical, upright 112x112
  image suitable for embedding extraction.

Modules are kept explicit and replaceable; nothing here performs enrollment or
recognition on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - depends on environment
    mp = None
    MP_IMPORT_ERROR = exc

# FaceMesh landmark indices used for the 5-point contract.
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

# Canonical ArcFace template for a 112x112 aligned crop.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)

LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")


@dataclass
class FaceKpsBox:
    """A detected face: bounding box (in image coordinates) plus 5 landmarks.

    Attributes:
        x1, y1, x2, y2: Axis-aligned bounding box corners.
        score: Detector confidence (1.0 placeholder for Haar).
        kps: ``(5, 2)`` float32 array of landmarks in ``[x, y]`` order.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray


def estimate_norm_5pt(
    kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """Return the 2x3 similarity transform mapping landmarks to the ArcFace template.

    The transform accounts for rotation, scale, and translation only; it never
    models perspective distortion.

    Args:
        kps_5x2: ``(5, 2)`` landmarks in ``[left_eye, right_eye, nose,
            left_mouth, right_mouth]`` order.
        out_size: Expected ``(width, height)`` of the aligned crop.

    Returns:
        A ``2 x 3`` affine matrix (``dtype=float32``).
    """
    kps = np.asarray(kps_5x2, dtype=np.float32).reshape(5, 2)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    template = ARCFACE_TEMPLATE_112.copy()
    if (out_w, out_h) != (112, 112):
        scale = np.array([out_w / 112.0, out_h / 112.0], dtype=np.float32)
        template = template * scale
    matrix, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(
            np.array([kps[0], kps[1], kps[2]], dtype=np.float32),
            np.array([template[0], template[1], template[2]], dtype=np.float32),
        )
    return matrix.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a face into a canonical upright crop.

    Args:
        frame_bgr: Input BGR image containing the face.
        kps_5x2: ``(5, 2)`` landmarks in full-frame coordinates.
        out_size: ``(width, height)`` of the output crop.

    Returns:
        Tuple of ``(aligned_bgr, matrix)``.
    """
    matrix = estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, matrix


def _clip_box_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    box = box.astype(np.float32).copy()
    box[0] = float(np.clip(box[0], 0, width - 1))
    box[1] = float(np.clip(box[1], 0, height - 1))
    box[2] = float(np.clip(box[2], 0, width - 1))
    box[3] = float(np.clip(box[3], 0, height - 1))
    return box


def bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """Build a face bounding box from the five landmarks.

    Asymmetric padding adds extra head-room at the top (forehead) and chin
    room at the bottom so the drawn box looks face-like and centered.

    Returns:
        ``(x1, y1, x2, y2)`` as a float32 array.
    """
    kps = np.asarray(kps, dtype=np.float32)
    x_min = float(np.min(kps[:, 0]))
    x_max = float(np.max(kps[:, 0]))
    y_min = float(np.min(kps[:, 1]))
    y_max = float(np.max(kps[:, 1]))
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    return np.array(
        [
            x_min - pad_x * width,
            y_min - pad_y_top * height,
            x_max + pad_x * width,
            y_max + pad_y_bot * height,
        ],
        dtype=np.float32,
    )


def _ema(prev: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return current.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * current).astype(np.float32)


def kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """Reject implausible landmark geometries.

    Checks the inter-eye distance is not collapsed and the mouth sits below
    the nose, which filters out many false positives.
    """
    kps = np.asarray(kps, dtype=np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < min_eye_dist:
        return False
    if not (left_mouth[1] > nose[1] and right_mouth[1] > nose[1]):
        return False
    return True


class Haar5ptDetector:
    """Face detection plus stable 5-point landmarks.

    Haar cascades are fast on CPU; MediaPipe FaceMesh confirms a real face and
    provides stable landmarks. Detections whose landmarks are inconsistent with
    the Haar box are rejected.

    Args:
        haar_xml: Path to a Haar cascade; defaults to the OpenCV frontal face.
        min_size: Minimum face size in pixels.
        min_neighbors: Haar ``detectMultiScale`` candidate threshold (lower
            finds smaller / further faces, at the cost of more candidates that
            the landmark consistency checks then filter).
        scale_factor: Haar down-scaling step (lower is more thorough).
        smooth_alpha: Exponential smoothing factor for box/landmark jitter.
        debug: Print rejection reasons.
    """

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (48, 48),
        min_neighbors: int = 2,
        scale_factor: float = 1.05,
        smooth_alpha: float = 0.80,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.min_neighbors = int(min_neighbors)
        self.scale_factor = float(scale_factor)
        self.smooth_alpha = float(smooth_alpha)

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                "mediapipe is required for 5-point landmarks.\n"
                f"Import failed: {MP_IMPORT_ERROR}\n"
                "Install with: pip install mediapipe==0.10.21"
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def haar_faces(self, gray: np.ndarray) -> np.ndarray:
        """Return Haar boxes as a ``(N, 4)`` int32 array of ``(x, y, w, h)``."""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        height, width = roi_bgr.shape[:2]
        if height < 20 or width < 20:
            return None
        roi, scale = roi_bgr, 1.0
        if max(height, width) < 120:
            scale = max(2.0, 120.0 / max(height, width))
            scale_roi = min(scale, 4.0)
            roi = cv2.resize(
                roi_bgr,
                None,
                fx=scale_roi,
                fy=scale_roi,
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        indices = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
        kps = np.array(points, dtype=np.float32)
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 0]]
        return kps

    def _points_inside(self, kps: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = box
        margin = 0.35
        x1 = x - margin * w
        y1 = y - margin * h
        x2 = x + (1.0 + margin) * w
        y2 = y + (1.0 + margin) * h
        inside = (
            (kps[:, 0] >= x1)
            & (kps[:, 0] <= x2)
            & (kps[:, 1] >= y1)
            & (kps[:, 1] <= y2)
        )
        return float(inside.mean()) >= 0.60

    def _facemesh_global_5pt(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Full-frame FaceMesh run returning cores of ``(5, 2)`` landmark lists.

        Used as the primary detection path: MediaPipe FaceMesh is far more
        robust to pose and lighting than a Haar cascade, and it is the source
        of the 5-point contract anyway.
        """
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        indices = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
        cores = []
        for landmarks_list in result.multi_face_landmarks:
            landmarks = landmarks_list.landmark
            kps = np.array(
                [[landmarks[i].x * width, landmarks[i].y * height] for i in indices],
                dtype=np.float32,
            )
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 0]]
            cores.append(kps)
        return cores

    def _haar_path_kps(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Fallback: Haar cascade + FaceMesh on each ROI."""
        height, width = frame_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        faces = self.haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        cores: List[np.ndarray] = []
        for (x, y, w, h) in faces:
            expand_x, expand_y = 0.25 * w, 0.35 * h
            rx1 = int(max(0, min(width - 1, round(x - expand_x))))
            ry1 = int(max(0, min(height - 1, round(y - expand_y))))
            rx2 = int(max(0, min(width - 1, round(x + w + expand_x))))
            ry2 = int(max(0, min(height - 1, round(y + h + expand_y))))
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                continue
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            if not self._points_inside(kps, (int(x), int(y), int(w), int(h))):
                if self.debug:
                    print("[haar_5pt] landmarks not consistent with Haar box -> reject")
                continue
            cores.append(kps)
        return cores

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        """Detect faces and landmarks.

        Prefers a full-frame MediaPipe FaceMesh pass (robust to pose/lighting).
        When it finds nothing and Haar candidates exist, falls back to the
        Haar + per-ROI FaceMesh path. Landmarks are sanity checked and
        temporally smoothed.

        Args:
            frame_bgr: Input BGR image.
            max_faces: Maximum number of faces to return (largest first).

        Returns:
            A list of :class:`FaceKpsBox`, ordered by area (largest first).
        """
        height, width = frame_bgr.shape[:2]

        cores = self._facemesh_global_5pt(frame_bgr)
        if not cores:
            cores = self._haar_path_kps(frame_bgr)

        if not cores:
            return []

        results: List[FaceKpsBox] = []
        for kps in cores:
            if not kps_span_ok(kps, min_eye_dist=max(8.0, 0.12 * width)):
                if self.debug:
                    print("[haar_5pt] landmark geometry sanity failed -> reject")
                continue

            box = _clip_box_xyxy(bbox_from_5pt(kps), width, height)

            box_smooth = _ema(self._prev_box, box, self.smooth_alpha)
            kps_smooth = _ema(self._prev_kps, kps, self.smooth_alpha)
            self._prev_box = box_smooth.copy()
            self._prev_kps = kps_smooth.copy()

            x1, y1, x2, y2 = box_smooth.tolist()
            results.append(
                FaceKpsBox(
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    score=1.0,
                    kps=kps_smooth.astype(np.float32),
                )
            )

        results.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        return results[:max_faces]


def draw_face_overlay(frame: np.ndarray, face: FaceKpsBox, color=(0, 255, 0)) -> None:
    """Draw a bounding box and the five landmark points in place."""
    cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
    for (px, py) in face.kps.astype(int):
        cv2.circle(frame, (int(px), int(py)), 3, color, -1)


def main() -> None:
    """Standalone demo: live Haar + 5-point landmark preview."""
    from .camera import Camera

    detector = Haar5ptDetector(debug=True)
    with Camera() as camera:
        print("Haar + 5pt (FaceMesh) test. Press 'q' to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            faces = detector.detect(frame, max_faces=1)
            vis = frame.copy()
            if faces:
                draw_face_overlay(vis, faces[0])
                cv2.putText(vis, "OK", (faces[0].x1, max(0, faces[0].y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
            cv2.imshow("haar_5pt", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break


if __name__ == "__main__":
    main()