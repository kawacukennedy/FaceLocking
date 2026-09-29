"""Pan/Tilt face tracking: the rig rotates to find and follow a face.

Spins the motorized mount (``src.tracker``) so the largest face stays centered
in the frame. When no face is visible the camera sweeps in a search pattern
until one reappears.

Tracking is tuned for responsiveness rather than stillness: the detector runs
single-face with a low detection threshold and a re-detection pass whenever the
face is lost, while the control loop smooths the face centre with a One Euro
filter so the mount neither buzzes nor lags.

Controls:
    q  quit
    s  toggle search mode explicitly
    c  re-center the mount
    e  toggle the expression overlay (smile / blink)
    f  toggle the 180 degree image rotation
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import ui
from .camera import Camera
from .expressions import ExpressionDetector
from .haar_5pt import Haar5ptDetector
from .recognize import match_expression
from .tracker import FaceTracker, PanTiltController


def main() -> None:
    # Single-face video mode with a low threshold finds a face sooner; box
    # smoothing is left to the One Euro filter inside FaceTracker, which does
    # not add lag the way an exponential average of boxes would.
    detector = Haar5ptDetector(
        min_size=(48, 48),
        max_num_faces=1,
        min_detection_confidence=0.30,
        reacquire_after_frames=2,
        smooth_alpha=0.0,
    )
    expressions = ExpressionDetector(max_num_faces=1)

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

    with Camera() as camera, expressions:
        print(camera.describe())
        print("Tracking started. q=quit, s=search, c=center, e=expressions, f=flip.")
        t0 = time.time()
        show_expressions = True
        blink_flash_until = 0.0
        frames, fps_t0, fps = 0, time.time(), None
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            now = time.time()
            dt = max(1e-3, now - t0)
            t0 = now
            vis = frame.copy()
            ui.vignette(vis)
            h, w = vis.shape[:2]

            frames += 1
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                frames, fps_t0 = 0, now

            faces = detector.detect(frame, max_faces=1)
            face_box = None
            expression = None
            if faces:
                face = faces[0]
                face_box = (face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1)
                if show_expressions:
                    boxes = np.array([[face.x1, face.y1, face.x2, face.y2]], dtype=np.float32)
                    states = expressions.update(frame, boxes)
                    expression = match_expression(states, (face.x1, face.y1, face.x2, face.y2))
                    if expression is not None and expression.blink_event:
                        blink_flash_until = now + 0.6

            searching, msg = tracker.update(face_box, w, h, dt)
            color = ui.YELLOW if searching else ui.GREEN
            if expression is not None and expression.blinking:
                color = ui.ORANGE
            if faces:
                ui.draw_corner_box(vis, (faces[0].x1, faces[0].y1,
                                         faces[0].x2 - faces[0].x1,
                                         faces[0].y2 - faces[0].y1), color=color)
                for (px, py) in faces[0].kps.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 3, color, -1)
                cx, cy = tracker.face_center
                if abs(tracker.error[0]) > 2 or abs(tracker.error[1]) > 2:
                    cv2.arrowedLine(vis, (int(w / 2), int(h / 2)),
                                    (int(cx), int(cy)), ui.CYAN, 2, tipLength=0.25)
            ui.draw_crosshair(vis, w // 2, h // 2, color=ui.CYAN)

            state = "SEARCHING" if searching else "TRACKING"
            pan = tracker.motor.pan if tracker.motor else tracker.center_pan
            tilt = tracker.motor.tilt if tracker.motor else tracker.center_tilt
            header = f"faces={len(faces)}"
            if fps is not None:
                header += f"  fps={fps:.1f}"
            lines = [
                ("FACE TRACKING", ui.WHITE),
                (f"{state} | {msg}", color),
                (f"pan={pan}  tilt={tilt}  err=({tracker.error[0]:+.0f},{tracker.error[1]:+.0f})", ui.CYAN),
            ]
            if searching:
                lines.append(("sweeping for a face...", ui.YELLOW))
            if show_expressions and expression is not None:
                if now < blink_flash_until:
                    lines.append((f"blink  total={expressions.total_blinks}", ui.ORANGE))
                elif expression.smiling:
                    lines.append((f"smiling {expression.smile_score:.2f}", ui.GREEN))
                else:
                    lines.append((f"neutral (smile {expression.smile_score:.2f})", ui.GRAY))
                if not expressions.calibrated:
                    have, need_total = expressions.calibration_progress
                    lines.append((f"calibrating neutral face ({need_total - have} frames)", ui.YELLOW))
            elif show_expressions:
                lines.append(("expressions on", ui.GRAY))
            else:
                lines.append(("expressions off (press e)", ui.GRAY))
            ui.draw_hud(vis, lines)
            ui.draw_text(vis, "q quit   s search   c center   e expressions   f flip",
                         (16, h - 16), scale=0.45, color=ui.GRAY)
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
            elif key == ord("e"):
                show_expressions = not show_expressions
            elif key == ord("f"):
                camera.toggle_flip()

    if tracker.motor is not None:
        tracker.motor.close()


if __name__ == "__main__":
    main()
