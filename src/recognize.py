"""Live multi-face recognition.

Brings every stage together: detect -> 5-point landmarks -> align (112x112) ->
ArcFace embedding -> cosine distance against the enrolled database -> label.

Controls:
    q  quit
    r  reload the database from disk
    +  loosen the distance threshold (accept more)
    -  tighten the distance threshold (accept fewer)
    d  toggle the debug overlay
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, DEFAULT_MODEL_PATH, cosine_similarity
from .haar_5pt import Haar5ptDetector, align_face_5pt

DEFAULT_DB_PATH = Path("data/db/face_db.npz")
DEFAULT_DIST_THRESHOLD = 0.34


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


def main() -> None:
    from .camera import Camera

    db_path = DEFAULT_DB_PATH
    detector = Haar5ptDetector(min_size=(48, 48))
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL_PATH)
    matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=DEFAULT_DIST_THRESHOLD)

    with Camera() as camera:
        print("Recognize (multi-face). q=quit, r=reload DB, +/- threshold, d=debug overlay")
        show_debug = False
        t0, frames, fps = time.time(), 0, None
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            faces = detector.detect(frame, max_faces=5)
            vis = frame.copy()

            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frames / elapsed
                frames, t0 = 0, time.time()

            h, w = vis.shape[:2]
            thumb, pad = 112, 8
            x0 = w - thumb - pad
            y0 = 80
            thumbnails_shown = 0

            for face in faces:
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 2, (0, 255, 0), -1)

                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                embedding = embedder.embed(aligned).embedding
                result = matcher.match(embedding)

                label = result.name if result.name is not None else "Unknown"
                color = (0, 255, 0) if result.accepted else (0, 0, 255)
                cv2.putText(vis, label, (face.x1, max(0, face.y1 - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis, f"dist={result.distance:.3f} sim={result.similarity:.3f}",
                            (face.x1, max(0, face.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if y0 + thumb <= h and thumbnails_shown < 4:
                    vis[y0 : y0 + thumb, x0 : x0 + thumb] = aligned
                    cv2.putText(vis, f"{thumbnails_shown + 1}:{label}", (x0, y0 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    y0 += thumb + pad
                    thumbnails_shown += 1

            if show_debug and faces:
                cv2.putText(vis, f"kpsLeye=({faces[0].kps[0, 0]:.0f},{faces[0].kps[0, 1]:.0f})",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            header = f"IDs={len(matcher.names)} thr(dist)={matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" fps={fps:.1f}"
            cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("recognize", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload_from(db_path)
                print(f"[recognize] reloaded DB: {len(matcher.names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} "
                      f"(sim~{1.0 - matcher.dist_thresh:.2f})")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} "
                      f"(sim~{1.0 - matcher.dist_thresh:.2f})")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")


if __name__ == "__main__":
    main()