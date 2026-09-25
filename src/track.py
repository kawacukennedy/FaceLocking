"""Pan/Tilt face tracking: the rig rotates to find and follow a face.

Spins the motorized mount (``src.tracker``) so the largest face stays centered
in the frame. When no face is visible the camera sweeps in a search pattern
until one reappears.

Controls:
    q  quit
    s  toggle search mode explicitly
    c  re-center the mount
    f  toggle the 180 degree image rotation
"""

from __future__ import annotations

import time

import cv2

from .camera import Camera
from .haar_5pt import Haar5ptDetector
from .tracker import PanTiltController, FaceTracker, SERVO_CENTER


def main() -> None:
    detector = Haar5ptDetector(min_size=(48, 48), smooth_alpha=0.70)

    try:
        motor = PanTiltController(center_pan=1550, center_tilt=1420)
        status = f"Motor: {'OK' if motor.ok else 'UNAVAILABLE'}"
        if not motor.ok and motor.last_error:
            status += f" ({motor.last_error})"
        print(status)
    except RuntimeError as exc:
        motor = None
        status = f"Motor unavailable: {exc}"

    tracker = FaceTracker(motor=motor)
    if motor and not motor.ok:
        tracker.motor = None

    with Camera() as camera:
        print(camera.describe())
        print("Tracking started. q=quit, s=search, c=center, f=flip.")
        t0 = time.time()
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            now = time.time()
            dt = max(1e-3, now - t0)
            t0 = now
            vis = frame.copy()
            h, w = vis.shape[:2]

            faces = detector.detect(frame, max_faces=1)
            face_box = None
            if faces:
                face = faces[0]
                face_box = (face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1)
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)

            searching, msg = tracker.update(face_box, w, h, dt)
            color = (0, 255, 255) if searching else (0, 255, 0)
            if faces:
                cv2.circle(vis, (w // 2, h // 2), 6, (255, 0, 0), 1)
            state = "SEARCH" if searching else "TRACK"
            top = [
                f"{state} | {msg}",
                f"pan={tracker.motor.pan if tracker.motor else 1500} "
                f"tilt={tracker.motor.tilt if tracker.motor else 1500}",
            ]
            y = 30
            for line in top:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 26
            cv2.imshow("face track", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                tracker.searching = not tracker.searching
                if not tracker.searching:
                    tracker.last_seen = time.time()
            elif key == ord("c"):
                if tracker.motor is not None:
                    tracker.motor.center()
            elif key == ord("f"):
                camera.toggle_flip()

    if tracker.motor is not None:
        tracker.motor.close()


if __name__ == "__main__":
    main()