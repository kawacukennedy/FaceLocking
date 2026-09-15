#!/usr/bin/env python3
"""Capture tests/data/sample_face.jpg for the pipeline test suite.

Usage:
    python scripts/capture_test_face.py

Saves a single frame from the default USB camera to
``tests/data/sample_face.jpg``. Tests under ``tests/test_pipeline.py`` use
this image and skip cleanly while it is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import Camera  # noqa: E402


def main() -> None:
    out_dir = ROOT / "tests" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_face.jpg"

    with Camera() as camera:
        print("Keep your face in view; capturing in 2 seconds... Press 'q' to abort.")
        cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
        final_frame = None
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("camera read failed")
            cv2.putText(frame, "capture in 2s (q=quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            final_frame = frame
        cv2.destroyAllWindows()

    if final_frame is None:
        raise SystemExit("no frame captured")
    cv2.imwrite(str(out_path), final_frame)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()