"""Alignment validation: warp the detected face into a canonical 112x112 crop.

A similarity transform (rotation, scale, translation only) maps the five
landmarks onto the ArcFace template. The aligned crop is what every downstream
stage consumes.

Run::

    python -m src.align

Keys:
    q  quit
    s  save the current aligned crop to ``data/debug_aligned/``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt


def main(out_size: Tuple[int, int] = (112, 112)) -> None:
    from .camera import Camera

    out_w, out_h = int(out_size[0]), int(out_size[1])
    detector = Haar5ptDetector(min_size=(48, 48), smooth_alpha=0.80)
    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)
    last_aligned = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    with Camera() as camera:
        print("align running. q=quit, s=save aligned face.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            vis = frame.copy()
            aligned = None
            faces = detector.detect(frame, max_faces=1)
            if faces:
                face = faces[0]
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
                aligned, _ = align_face_5pt(frame, face.kps, out_size=out_size)
                if aligned.size:
                    last_aligned = aligned
                cv2.putText(vis, "OK (Haar + FaceMesh 5pt)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
            cv2.imshow("align - camera", vis)
            cv2.imshow("align - aligned", last_aligned)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_path = save_dir / f"{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(out_path), last_aligned)
                print(f"[align] saved: {out_path}")


if __name__ == "__main__":
    main()