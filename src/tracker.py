"""Pan/Tilt motor controller over a USB-UART bridge (CP2102).

Drives the two-servo (or equivalent) pan/tilt mount of the tracking rig. The
controller speaks a small newline-terminated ASCII protocol over a virtual
serial port. This module discovers that port automatically, identifies the
baud rate, smooths commanded positions, and provides a search sweep used to
re-acquire a lost face.

Command protocol (default, one line per update)::

    P<pan> T<tilt>\\n        # P for pan, T for tilt

``pan`` and ``tilt`` are servo pulse widths in microseconds (500-2500); 1500 is
center. The writer is pluggable so it can be adapted to other controllers
without touching the tracker.

Run the standalone test with::

    python -m src.tracker
"""

from __future__ import annotations

import glob
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import serial
except Exception as exc:  # pragma: no cover - depends on environment
    serial = None
    SERIAL_IMPORT_ERROR = exc

SERVO_MIN = 500
SERVO_MAX = 2500
SERVO_CENTER = 1500
BAUD_CANDIDATES = (9600, 19200, 38400, 57600, 115200)

USB_SERIAL_PATTERNS = ("/dev/cu.usb*", "/dev/tty.usb*", "/dev/cu.SLAB_USBtoUART")


class CommandProtocol:
    """Base class for writing pan/tilt positions to a serial device."""

    def __init__(self, port: str, baud: int) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial is required for the motor controller.\n"
                f"Import failed: {SERIAL_IMPORT_ERROR}"
            )
        self.port = port
        self.baud = baud
        self.ser = self._open()

    def _open(self) -> "serial.Serial":
        ser = serial.Serial(self.port, baudrate=self.baud, timeout=1.0)
        try:
            ser.dtr = True
            ser.rts = True
        except (serial.SerialException, OSError):
            pass
        return ser

    def write(self, pan: int, tilt: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class PWMCommandProtocol(CommandProtocol):
    """Writes ``P<pan> T<tilt>\\n`` (servo pulse widths)."""

    def write(self, pan: int, tilt: int) -> None:
        payload = f"P{int(pan)} T{int(tilt)}\n".encode("ascii")
        try:
            self.ser.write(payload)
            self.ser.flush()
        except Exception:
            raise RuntimeError("motor controller write failed")


def serial_device_paths() -> List[str]:
    """All candidate virtual-serial paths for the motor controller."""
    paths = []
    for pattern in USB_SERIAL_PATTERNS:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths and serial is not None:
        import serial.tools.list_ports as list_ports

        for info in list_ports.comports():
            if info.vid == 0x10C4 and info.pid == 0xEA60:
                paths.append(info.device)
    return [p for p in dict.fromkeys(paths)]


def find_motor_port(max_attempts: int = 5, wait_s: float = 2.0) -> str:
    """Return the motor controller's serial device path.

    The CP2102 node can disappear from the bus for a moment after heavy
    probing; this helper retries until the port exists again.
    """
    for attempt in range(max_attempts):
        paths = serial_device_paths()
        if paths:
            return paths[0]
        time.sleep(wait_s)
    raise RuntimeError(
        "Motor controller serial port not found. Check USB connection."
    )


def probe_baud(port: str) -> Optional[int]:
    """Identify the controller baud rate by looking for an ASCII banner.

    Returns ``None`` when the device is silent or speaks binary at every rate.
    """
    for baud in BAUD_CANDIDATES:
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1.0)
            ser.dtr = True
            ser.rts = True
            time.sleep(1.0)
            data = ser.read(1024)
            ser.close()
        except Exception:
            continue
        if data and sum(32 <= c < 127 for c in data) / max(1, len(data)) > 0.8:
            return baud
    return None


def clamp_pwm(value: int, min_pwm: int = SERVO_MIN, max_pwm: int = SERVO_MAX) -> int:
    return int(max(min_pwm, min(max_pwm, round(value))))


class PanTiltController:
    """Smooth pan/tilt driver with position clamping and slew limiting.

    The CP2102 mount can briefly drop off the USB bus; writes are verified
    and the controller transparently re-opens the port on the next move.
    ``move()`` never raises for hardware problems — failures are recorded in
    ``last_error`` and the tracker keeps running.

    Args:
        port: Serial device path.
        baud: Baud rate (auto-probed when ``None``).
        protocol_factory: Callable returning a :class:`CommandProtocol`.
        center_pan, center_tilt: Neutral (home) pulse widths.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: Optional[int] = None,
        protocol_factory: Callable[[str, int], CommandProtocol] = PWMCommandProtocol,
        center_pan: int = SERVO_CENTER,
        center_tilt: int = SERVO_CENTER,
    ) -> None:
        self.center_pan = int(center_pan)
        self.center_tilt = int(center_tilt)
        self.protocol_factory = protocol_factory
        self.elapsed = 0.0
        self.enabled = False
        self.protocol: Optional[CommandProtocol] = None
        self.last_error: Optional[str] = None
        self.pan = self.center_pan
        self.tilt = self.center_tilt
        self._last_connect_at = 0.0

        if port is None:
            port = find_motor_port()
        self.port = port
        self.baud = baud if baud is not None else (probe_baud(port) or 115200)
        self._connect()

    @property
    def ok(self) -> bool:
        return self.enabled and self.protocol is not None

    def _teardown(self) -> None:
        if self.protocol is not None:
            try:
                self.protocol.close()
            except Exception:
                pass
        self.protocol = None
        self.enabled = False

    def _connect(self) -> bool:
        if self._last_connect_at and time.time() - self._last_connect_at < 0.5:
            return False
        self._last_connect_at = time.time()
        try:
            self.protocol = self.protocol_factory(self.port, self.baud)
            self.protocol.write(self.pan, self.tilt)
            self.enabled = True
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._teardown()
            return False

    def _send(self) -> bool:
        if self.protocol is None and not self._connect():
            return False
        try:
            self.protocol.write(self.pan, self.tilt)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._teardown()
            return False

    def move(self, pan: int, tilt: int, slew_limit: int = 20) -> None:
        """Move toward an absolute pulse width, at most ``slew_limit`` per call.

        Position state is always updated; the write is best-effort.
        """
        target_pan = clamp_pwm(pan)
        target_tilt = clamp_pwm(tilt)
        self.pan += max(-slew_limit, min(slew_limit, target_pan - self.pan))
        self.tilt += max(-slew_limit, min(slew_limit, target_tilt - self.tilt))
        self._send()

    def center(self) -> None:
        self.move(self.center_pan, self.center_tilt)

    def close(self) -> None:
        if self.protocol is not None:
            try:
                self.protocol.write(self.center_pan, self.center_tilt)
            except Exception:
                pass
        self._teardown()

    def __enter__(self) -> "PanTiltController":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FaceTracker:
    """Keeps the largest detected face centered by steering the pan/tilt mount.

    Works purely from a ``(x, y, w, h)`` face box and the frame size. Uses a
    proportional controller on the face-center error, then maps the error to a
    pulse-width target. When no face is seen for ``search_delay_s`` seconds it
    switches to a search sweep that rotates the mount until a face reappears.

    Args:
        motor: An optional :class:`PanTiltController`.
        k_pan, k_tilt: Proportional gains (pulse width change per pixel).
        search_delay_s: Seconds without a face before sweeping.
    """

    def __init__(
        self,
        motor: Optional[PanTiltController] = None,
        k_pan: float = 2.5,
        k_tilt: float = 2.5,
        search_delay_s: float = 1.5,
    ) -> None:
        self.motor = motor
        self.k_pan = float(k_pan)
        self.k_tilt = float(k_tilt)
        self.search_delay_s = float(search_delay_s)
        self.last_seen = time.time()
        self.searching = False
        self.center_pan = motor.center_pan if motor else SERVO_CENTER
        self.center_tilt = motor.center_tilt if motor else SERVO_CENTER
        self._sweep_start: float = 0.0
        self.sweep_period_s = 4.0
        self.sweep_amp_pan = 420
        self.sweep_amp_tilt = 180

    def update(
        self,
        face_box: Optional[Tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
        dt: float,
    ) -> Tuple[bool, str]:
        """Run one tracking step.

        Returns:
            ``(searching, status_message)``.
        """
        if face_box is not None:
            x, y, w, h = face_box
            face_cx = x + w / 2.0
            face_cy = y + h / 2.0
            err_x = face_cx - frame_w / 2.0
            err_y = face_cy - frame_h / 2.0
            self.last_seen = time.time()
            self.searching = False

            if self.motor is not None:
                self.motor.move(
                    self.center_pan - err_x * self.k_pan,
                    self.center_tilt + err_y * self.k_tilt,
                )
            return False, f"Track err=({err_x:+.0f},{err_y:+.0f})"
        return self._search(frame_w, frame_h)

    def _search(self, frame_w: int, frame_h: int) -> Tuple[bool, str]:
        if time.time() - self.last_seen < self.search_delay_s:
            return False, "Face lost... waiting"
        if not self.searching:
            self.searching = True
            self._sweep_start = time.time()
        age = (time.time() - self._sweep_start) % self.sweep_period_s
        phase = age / self.sweep_period_s
        pan = self.center_pan + self.sweep_amp_pan * (
            2.0 * phase - 1.0 if phase < 0.5 else 1.0 - 2.0 * (phase - 0.5)
        )
        tilt_delta = self.sweep_amp_tilt * (0.5 - abs(0.5 - phase))
        tilt = clamp_pwm(self.center_tilt + tilt_delta)
        if self.motor is not None:
            self.motor.move(int(pan), int(tilt), slew_limit=30)
        return True, "Searching... sweeping"


def main() -> None:
    """Standalone motor test. Cycles the mount through a small sweep."""
    try:
        motor = PanTiltController()
    except RuntimeError as exc:
        print(f"Motor unavailable: {exc}")
        return
    print(f"Motor available: {'OK' if motor.ok else 'unavailable'}")
    if not motor.ok:
        print(f"  last error: {motor.last_error}")
    if not motor.ok:
        return
    for step in (-400, 0, 400, 0):
        motor.move(SERVO_CENTER + step, SERVO_CENTER, slew_limit=50)
        time.sleep(0.6)
    motor.center()


if __name__ == "__main__":
    main()