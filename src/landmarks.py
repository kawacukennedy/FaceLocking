"""5-point landmark validation: bounding box plus five labeled points.

Landmark order is a strict contract for the whole project:
``[left_eye, right_eye, nose, left_mouth, right_mouth]``.

Run::

    python -m src.landmarks
"""

from __future__ import annotations

import cv2

from .haar_5pt import Haar5ptDetector, LANDMARK_NAMES


def main() -> None:
    from .camera import Camera

    detector = Haar5ptDetector(debug=False)
    with Camera() as camera:
        print("5pt landmarks. Press 'q' to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            faces = detector.detect(frame, max_faces=1)
            vis = frame.copy()
            if faces:
                face = faces[0]
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.putText(vis, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("5pt Landmarks", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break


if __name__ == "__main__":
    main()