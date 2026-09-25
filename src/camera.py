"""Camera access with automatic pixel-quality fallback.

Some UVC cameras deliver all-black frames through the OpenCV AVFoundation
backend even though the hardware is healthy. To keep the pipeline reliable on
such cameras, :class:`Camera` first tries OpenCV and verifies the frames are
not uniformly black; if they are, it transparently falls back to an ffmpeg raw
frame stream.

Run the standalone validation with::

    python -m src.camera
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

BLACK_FRAME_MEAN = 3.0
WARMUP_FRAMES = 8
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FRAMERATE = 30

# Environment overrides (handy when hardware changes):
#   PYCAMERA_DEVICE  pick the avfoundation device whose name contains this text
#   PYCAMERA_FLIP    "1"/"true" rotates frames by 180 degrees, "0" keeps them
#                    as captured (default)
DEVICE_ENV = "PYCAMERA_DEVICE"
FLIP_ENV = "PYCAMERA_FLIP"

# Names that mean "not a USB camera": the laptop's own camera, screen/screen
# recording sources, and software (virtual) cameras. Any device whose name does
# not match one of these markers is treated as an external USB camera.
BUILTIN_DEVICE_MARKERS = (
    "facetime",
    "built-in",
    "built in",
    "capture screen",
    "screen capture",
    "display",
    "virtual",
    "continuity",
    "iphone",
    "ipad",
    "droidcam",
    "epoccam",
    "manycam",
    "snap camera",
    "obs",
)


def is_builtin_camera_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in BUILTIN_DEVICE_MARKERS)


def ffmpeg_device_names() -> List[Tuple[int, str]]:
    """Return ``(index, name)`` pairs for avfoundation video devices.

    Parses ``ffmpeg -list_devices`` output, skipping the audio section.
    Returns ``[]`` when ffmpeg is unavailable.
    """
    if not ffmpeg_available():
        return []
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stderr.splitlines():
        lowered = line.lower()
        if "audio devices" in lowered:
            break
        match = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def external_device_names() -> List[Tuple[int, str]]:
    """Return ``(index, name)`` pairs for external (USB) video devices."""
    return [(i, n) for i, n in ffmpeg_device_names() if not is_builtin_camera_name(n)]


def auto_camera_index() -> int:
    """Pick the external (USB) camera to use, never the built-in one.

    Preference order:
    1. The device whose name contains ``PYCAMERA_DEVICE`` when that variable is
       set, so any USB camera can be pinned by name.
    2. The first avfoundation video device that is not a built-in, screen or
       virtual camera (USB webcams show up as such, e.g. ``Wed Camera``).

    Raises:
        RuntimeError: if ``PYCAMERA_DEVICE`` does not match any device, or if
            only built-in/screen devices are connected. The built-in camera is
            never selected silently; connect a USB camera or set
            ``PYCAMERA_DEVICE`` explicitly.
    """
    devices = ffmpeg_device_names()
    wanted = os.environ.get(DEVICE_ENV, "").strip().lower()
    if wanted:
        for index, name in devices:
            if wanted in name.lower():
                return index
        available = [f"{i}:{n}" for i, n in devices] or ["<none detected>"]
        raise RuntimeError(
            f"{DEVICE_ENV}={wanted!r} matched no camera. Available: {available}"
        )

    externals = external_device_names()
    if externals:
        return externals[0][0]

    if devices:
        # ffmpeg can enumerate, so we know what is there and none of it is USB.
        raise RuntimeError(
            "No external camera connected; only built-in/screen devices found: "
            + ", ".join(f"{i}:{n}" for i, n in devices)
            + ". Plug in a USB camera (or set PYCAMERA_DEVICE=<name>)."
        )

    # Device names unavailable (ffmpeg missing): fall back to probing indices.
    for index in range(4):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        means = []
        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None:
                means.append(float(np.mean(frame)))
            time.sleep(0.05)
        cap.release()
        if means and max(means) >= BLACK_FRAME_MEAN:
            return index
    return 1


def auto_flip_enabled(index: int) -> bool:
    """Whether frames from ``index`` should be rotated 180 degrees.

    Orientation follows the physical mount, which can change at any time, so
    the default is *no rotation*. Force it with ``PYCAMERA_FLIP=1`` (disable
    again with ``PYCAMERA_FLIP=0``); every demo also accepts the ``f`` key to
    toggle the rotation live.
    """
    override = os.environ.get(FLIP_ENV, "").strip().lower()
    return override in {"1", "true", "yes", "on"}


def list_camera_indices(max_devices: int = 6) -> list:
    """Return the indices of cameras that OpenCV can open.

    Each entry is a ``(index, backend, width, height)`` tuple.
    """
    available = []
    with contextlib.redirect_stderr(io.StringIO()):
        for index in range(max_devices):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available.append(
                    (
                        index,
                        cap.getBackendName(),
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    )
                )
                cap.release()
    return available


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FFmpegFrameReader:
    """Reads BGR frames from an ffmpeg rawvideo pipe.

    This sidesteps broken OpenCV backends. The ffmpeg process writes raw
    ``width * height * 3`` byte BGR frames to stdout; each call to :meth:`read`
    blocks until one complete frame is available.
    """

    def __init__(self, index: int, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg not found on PATH; cannot start raw frame reader")
        self.width = width
        self.height = height
        self.index = index
        self.frame_bytes = width * height * 3
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "avfoundation",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(FRAMERATE),
                "-i",
                str(index),
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._stream = self._proc.stdout

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        data = self._stream.read(self.frame_bytes)
        if data is None or len(data) != self.frame_bytes:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._stream = None


class Camera:
    """Unified camera reader that never returns systematically black frames.

    The external USB camera is preferred over the laptop's built-in one, and
    its orientation can be corrected with a 180 degree rotation when the mount
    is installed upside down. Rotation is off by default because it follows the
    physical mount, not the device name; use ``PYCAMERA_FLIP=1``, the ``flip``
    argument, or the ``f`` key in the demos to enable it.

    Attributes:
        backend: ``"opencv"`` or ``"ffmpeg"``, whichever is actually in use.
        device_name: avfoundation name of the active device.
        value_flip: whether frames are currently rotated 180 degrees.
        fps: Measured frame rate of the active reader (float, or None).
    """

    def __init__(
        self,
        index: Optional[int] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        prefer_ffmpeg: bool = False,
        black_mean: float = BLACK_FRAME_MEAN,
        flip: Optional[bool] = None,
    ) -> None:
        if index is None:
            index = auto_camera_index()
        self.index = index
        self.width = width
        self.height = height
        self.black_mean = black_mean
        self.backend: Optional[str] = None
        self.fps: Optional[float] = None
        self._cv: Optional[cv2.VideoCapture] = None
        self._ff: Optional[FFmpegFrameReader] = None
        self._t0 = time.time()
        self._n = 0
        self.device_name = dict(ffmpeg_device_names()).get(self.index, f"index {self.index}")

        if flip is None:
            self.value_flip = auto_flip_enabled(self.index)
        else:
            self.value_flip = bool(flip)

        if not prefer_ffmpeg:
            try:
                self._open_opencv()
            except RuntimeError:
                self._open_ffmpeg()
        else:
            self._open_ffmpeg()

    def _open_opencv(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        means = []
        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None:
                means.append(float(np.mean(frame)))
            time.sleep(0.05)
        if not means or max(means) < self.black_mean:
            cap.release()
            raise RuntimeError("OpenCV backend returned black frames; falling back to ffmpeg")
        self._cv = cap
        self.backend = "opencv"

    def _open_ffmpeg(self) -> None:
        self._ff = FFmpegFrameReader(self.index, self.width, self.height)
        self.backend = "ffmpeg"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cv is not None:
            ok, frame = self._cv.read()
        elif self._ff is not None:
            ok, frame = self._ff.read()
        else:  # pragma: no cover
            return False, None
        self._n += 1
        if ok and frame is not None and self.value_flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return ok, frame

    def report_fps(self) -> float:
        """Return frames per second measured over the reader's lifetime."""
        elapsed = time.time() - self._t0
        return self._n / elapsed if elapsed > 0 else 0.0

    def toggle_flip(self) -> bool:
        """Flip the 180 degree rotation on/off and return the new state."""
        self.value_flip = not self.value_flip
        return self.value_flip

    def describe(self) -> str:
        """One-line summary of the active device, for demo startup banners."""
        kind = "built-in" if is_builtin_camera_name(self.device_name) else "external"
        flip = "ON (rotated 180)" if self.value_flip else "OFF"
        line = (
            f"Camera: '{self.device_name}' [index {self.index}, {kind}] "
            f"backend={self.backend} flip={flip}"
        )
        others = [n for i, n in external_device_names() if i != self.index]
        if others:
            line += f" | other USB cameras: {', '.join(others)} (set {DEVICE_ENV}=<name>)"
        return line

    def release(self) -> None:
        if self._cv is not None:
            self._cv.release()
            self._cv = None
        if self._ff is not None:
            self._ff.release()
            self._ff = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def main() -> None:
    """Live camera validation. Press 'q' to exit."""
    names = ffmpeg_device_names()
    print("Cameras found (USB cameras are preferred over the built-in one):")
    if names:
        for index, name in names:
            kind = "built-in/screen" if is_builtin_camera_name(name) else "external USB"
            print(f"  index {index}: {name} [{kind}]")
    else:
        print("  <ffmpeg unavailable, device names unknown>")
    for index, backend, w, h in list_camera_indices():
        name = dict(names).get(index, "<unnamed>")
        print(f"  opencv check: index {index} ({name}) {backend} {w}x{h}")

    try:
        index = auto_camera_index()
    except RuntimeError as exc:
        print(f"Cannot select a camera: {exc}")
        return
    with Camera(index=index) as camera:
        print(camera.describe())
        print("Press 'f' to toggle the 180 degree rotation, 'q' to quit.")
        t0, n = time.time(), 0
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read frame.")
                break
            n += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                print(f"FPS: {n / elapsed:.1f}")
                n, t0 = 0, time.time()
            cv2.imshow("Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                print(f"flip={'ON (rotated 180)' if camera.toggle_flip() else 'OFF'}")


if __name__ == "__main__":
    main()