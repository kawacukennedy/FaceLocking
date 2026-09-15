"""Face detection validation: draw a Haar bounding box around the largest face.

Run::

    python -m src.detect
"""

from __future__ import annotations

import cv2


def build_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    return cascade


def main() -> None:
    from .camera import Camera

    cascade = build_cascade()
    with Camera() as camera:
        print("Haar face detect (minimal). Press 'q' to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Detection", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break


if __name__ == "__main__":
    main()