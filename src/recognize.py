"""Live multi-face recognition.

Brings every stage together: detect -> 5-point landmarks -> align (112x112) ->
ArcFace embedding -> cosine distance against the enrolled database -> label.
Smile and blink detection run alongside, so one window shows who is in front
of the camera and what they are doing.

Controls:
    q  quit
    r  reload the database from disk
    +  loosen the distance threshold (accept more)
    -  tighten the distance threshold (accept fewer)
    e  toggle the expression overlay (smile / blink)
    c  recalibrate the neutral face used by the expression detector
    d  toggle the debug overlay
    f  toggle the 180 degree image rotation

The overlay states the current scenario explicitly: ``NO FACE IN FRAME`` when
nobody is visible, a green name for an enrolled identity, and ``Unknown`` for
a face that does not match the database.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, DEFAULT_MODEL_PATH
from .expressions import ExpressionDetector, FaceExpression
from .haar_5pt import Haar5ptDetector, align_face_5pt

DEFAULT_DB_PATH = Path("data/db/face_db.npz")
# Calibrated from live measurements: enrolled identities match at a cosine
# distance of ~0.05, while a mismatched face lands around 0.4 and above, so
# 0.20 separates the two with a wide margin. Tune live with +/- or with
# `python -m src.evaluate` once two or more identities are enrolled.
DEFAULT_DIST_THRESHOLD = 0.20


@dataclass
class MatchResult:
    """Outcome of comparing a query embedding against the database."""

    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {key: np.asarray(data[key], dtype=np.float32).reshape(-1) for key in data.files}


class FaceDBMatcher:
    """Fast 1-to-N cosine matching against a set of enrolled embeddings."""

    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = DEFAULT_DIST_THRESHOLD) -> None:
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self.names: List[str] = []
        self.matrix: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self) -> None:
        self.names = sorted(self.db.keys())
        if self.names:
            self.matrix = np.stack(
                [self.db[name].reshape(-1).astype(np.float32) for name in self.names], axis=0
            )
        else:
            self.matrix = None

    def reload_from(self, path: Path) -> None:
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, embedding: np.ndarray) -> MatchResult:
        if self.matrix is None or not self.names:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        query = embedding.reshape(1, -1).astype(np.float32)
        similarities = (self.matrix @ query.T).reshape(-1)
        best_index = int(np.argmax(similarities))
        best_sim = float(similarities[best_index])
        best_dist = 1.0 - best_sim
        accepted = best_dist <= self.dist_thresh
        return MatchResult(
            name=self.names[best_index] if accepted else None,
            distance=best_dist,
            similarity=best_sim,
            accepted=accepted,
        )


@dataclass
class FaceRecognition:
    """A cached identity decision for one tracked face."""

    result: MatchResult
    box: Tuple[int, int, int, int]
    at: float
    reused: bool = False

    @property
    def label(self) -> str:
        return self.result.name if self.result.name is not None else "Unknown"


class RecognitionCache:
    """Re-embed a tracked face only when its identity could have changed.

    Embedding a face is by far the most expensive step (a ResNet-50 pass), while
    a face that barely moved between frames produces essentially the same
    vector. The cache therefore refreshes a decision when the face has moved
    more than ``move_px``, when the entry is older than ``refresh_s``, or when
    it has never been matched. Between refreshes the previous label is reused,
    which keeps the overlay smooth at the frame rate of the detector.
    """

    def __init__(
        self,
        embedder: ArcFaceEmbedderONNX,
        matcher: "FaceDBMatcher",
        refresh_s: float = 0.40,
        move_px: float = 18.0,
    ) -> None:
        self.embedder = embedder
        self.matcher = matcher
        self.refresh_s = float(refresh_s)
        self.move_px = float(move_px)
        self._entries: List[FaceRecognition] = []
        self.embed_calls = 0

    def reset(self) -> None:
        self._entries.clear()

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
        area_b = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _shift_px(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax = 0.5 * (a[0] + a[2]) - 0.5 * (b[0] + b[2])
        ay = 0.5 * (a[1] + a[3]) - 0.5 * (b[1] + b[3])
        return float(np.hypot(ax, ay))

    def identify(
        self, frame: np.ndarray, faces: List, now: Optional[float] = None
    ) -> List[FaceRecognition]:
        """Return one :class:`FaceRecognition` per face, in the input order."""
        now = time.time() if now is None else now
        used: set = set()
        out: List[FaceRecognition] = []
        for face in faces:
            box = (face.x1, face.y1, face.x2, face.y2)
            best_index, best_iou = None, 0.2
            for i, entry in enumerate(self._entries):
                if i in used:
                    continue
                iou = self._iou(box, entry.box)
                if iou > best_iou:
                    best_index, best_iou = i, iou

            if best_index is not None:
                entry = self._entries[best_index]
                used.add(best_index)
                fresh_enough = (now - entry.at) < self.refresh_s
                still_enough = self._shift_px(box, entry.box) <= self.move_px
                if fresh_enough and still_enough:
                    out.append(
                        FaceRecognition(result=entry.result, box=box, at=entry.at, reused=True)
                    )
                    continue
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                self.embed_calls += 1
                result = self.matcher.match(self.embedder.embed(aligned).embedding)
                entry.result, entry.box, entry.at, entry.reused = result, box, now, False
                out.append(entry)
                continue

            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            self.embed_calls += 1
            result = self.matcher.match(self.embedder.embed(aligned).embedding)
            out.append(FaceRecognition(result=result, box=box, at=now))

        # The cache holds exactly the faces currently visible, so a face that
        # left the frame is forgotten and a returning face is embedded again.
        self._entries = list(out)
        return out


def match_expression(
    states: List[FaceExpression], box: Tuple[int, int, int, int]
) -> Optional[FaceExpression]:
    """Find the expression state belonging to a detection box.

    The expression tracker runs its own landmark pass, so states are matched
    back to detections by the overlap of their boxes rather than by index.
    """
    best: Optional[FaceExpression] = None
    best_iou = 0.05
    bx1, by1, bx2, by2 = box
    for state in states:
        sx1, sy1, sx2, sy2 = state.box
        ix1, iy1 = max(bx1, sx1), max(by1, sy1)
        ix2, iy2 = min(bx2, sx2), min(by2, sy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            continue
        area_box = max(1, bx2 - bx1) * max(1, by2 - by1)
        area_state = max(1, sx2 - sx1) * max(1, sy2 - sy1)
        iou = inter / (area_box + area_state - inter)
        if iou > best_iou:
            best, best_iou = state, iou
    return best


def main() -> None:
    from . import ui
    from .camera import Camera

    db_path = DEFAULT_DB_PATH
    detector = Haar5ptDetector(min_size=(48, 48))
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL_PATH, debug=True)
    matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=DEFAULT_DIST_THRESHOLD)
    cache = RecognitionCache(embedder, matcher)
    expressions = ExpressionDetector(max_num_faces=5)

    with Camera() as camera, expressions:
        print("Recognize (multi-face). q=quit, r=reload DB, +/- threshold, "
              "e=expressions, c=recalibrate, d=debug, f=flip")
        print(camera.describe())
        show_debug = False
        show_expressions = True
        t0, frames, fps = time.time(), 0, None
        blink_flash_until = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            faces = detector.detect(frame, max_faces=5)
            vis = frame.copy()
            ui.vignette(vis)

            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frames / elapsed
                frames, t0 = 0, time.time()

            h, w = vis.shape[:2]
            boxes = (
                np.array([[f.x1, f.y1, f.x2, f.y2] for f in faces], dtype=np.float32)
                if faces
                else None
            )
            states = expressions.update(frame, boxes) if (show_expressions and faces) else []
            if any(s.blink_event for s in states):
                blink_flash_until = time.time() + 0.6

            thumb, pad = 112, 8
            x0 = w - thumb - pad
            y0 = 92
            thumbnails_shown = 0
            any_smiling = False

            recognitions = cache.identify(frame, faces)
            for face, recognition in zip(faces, recognitions):
                result_box = (face.x1, face.y1, face.x2, face.y2)
                expression = match_expression(states, result_box) if states else None
                if expression is not None and expression.smiling:
                    any_smiling = True

                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                result = recognition.result
                label = recognition.label
                color = ui.GREEN if result.accepted else ui.RED
                if expression is not None and expression.blinking:
                    color = ui.ORANGE

                ui.draw_corner_box(vis, (face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1),
                                   color=color)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 2, color, -1)

                ui.draw_badge(vis, label, (face.x1, face.y1 - 6), color=color,
                              scale=0.56, anchor="bottom-left")
                ui.draw_text(vis, f"dist {result.distance:.3f}", (face.x1, max(14, face.y1 - 30)),
                             scale=0.48, color=ui.GRAY)
                if expression is not None:
                    tags = []
                    if expression.smiling:
                        tags.append(f"SMILE {expression.smile_score:.2f}")
                    if expression.blinking:
                        tags.append("BLINK")
                    if tags:
                        ui.draw_badge(vis, "  ".join(tags), (face.x2, face.y1 - 6),
                                      color=ui.GREEN if expression.smiling else ui.ORANGE,
                                      scale=0.45, anchor="bottom-right")
                    ui.draw_meter(vis, expression.smile_score, (face.x1, face.y2 + 16),
                                  width=110, height=6,
                                  color=ui.GREEN if expression.smiling else ui.CYAN,
                                  label=f"smile {expression.smile_score:.2f}")

                if y0 + thumb <= h and thumbnails_shown < 4:
                    vis[y0 : y0 + thumb, x0 : x0 + thumb] = aligned
                    caption = label if not recognition.reused else f"{label} (cached)"
                    ui.draw_badge(vis, f"{thumbnails_shown + 1}:{caption}", (x0, y0 - 6),
                                  color=color, scale=0.45, anchor="bottom-left")
                    y0 += thumb + pad
                    thumbnails_shown += 1

            if show_debug and faces:
                ui.draw_text(vis, f"kpsLeye=({faces[0].kps[0, 0]:.0f},{faces[0].kps[0, 1]:.0f})",
                             (16, h - 18), scale=0.5, color=ui.GRAY)
                ui.draw_text(vis, f"embeddings this run: {cache.embed_calls}", (16, h - 40),
                             scale=0.5, color=ui.GRAY)

            if not matcher.names:
                state, state_color = "EMPTY DATABASE - enroll someone first", ui.RED
            elif not faces:
                state, state_color = "NO FACE IN FRAME", ui.YELLOW
            else:
                state, state_color = f"{len(faces)} FACE(S) DETECTED", ui.GREEN
            header = f"IDs={len(matcher.names)}  thr(dist)={matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f"  fps={fps:.1f}"
            lines = [("FACELOCKING", ui.WHITE), (header, ui.CYAN), (state, state_color)]
            if show_expressions:
                if not expressions.calibrated:
                    have, need_total = expressions.calibration_progress
                    lines.append((f"calibrating neutral face ({need_total - have} frames)", ui.YELLOW))
                elif time.time() < blink_flash_until:
                    lines.append((f"blink detected  total={expressions.total_blinks}", ui.ORANGE))
                elif any_smiling:
                    lines.append((f"smiling  total={expressions.total_smiles}", ui.GREEN))
                else:
                    lines.append((f"expressions on  blinks={expressions.total_blinks}  "
                                  f"smiles={expressions.total_smiles}", ui.GRAY))
            else:
                lines.append(("expressions off (press e)", ui.GRAY))
            lines.append(("q quit  r reload  +/- threshold  e expressions  "
                          "c calibrate  d debug  f flip", ui.GRAY))
            ui.draw_hud(vis, lines)
            cv2.imshow("recognize", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload_from(db_path)
                cache.reset()
                print(f"[recognize] reloaded DB: {len(matcher.names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                cache.reset()  # cached decisions used the old threshold
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} "
                      f"(sim~{1.0 - matcher.dist_thresh:.2f})")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                cache.reset()  # cached decisions used the old threshold
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} "
                      f"(sim~{1.0 - matcher.dist_thresh:.2f})")
            elif key == ord("e"):
                show_expressions = not show_expressions
                print(f"[recognize] expressions={'ON' if show_expressions else 'OFF'}")
            elif key == ord("c"):
                expressions.reset()
                print("[recognize] recalibrating neutral face...")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
            elif key == ord("f"):
                state = camera.toggle_flip()
                print(f"[recognize] flip={'ON (rotated 180)' if state else 'OFF'}")


if __name__ == "__main__":
    main()