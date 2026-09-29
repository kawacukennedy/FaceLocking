"""Tracking tests: One Euro smoothing, deadband, command throttling, search.

No hardware is needed — the motor is a recording fake and time is injected by
patching ``src.tracker.time.time``.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

import src.tracker as tracker
from src.tracker import (
    SERVO_CENTER,
    FaceTracker,
    OneEuroFilter,
    PanTiltController,
    clamp_pwm,
)


class FakeClock:
    """Monotonic clock stub so timing behaviour is deterministic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, dt: float) -> float:
        self.now += dt
        return self.now


class FakeMotor:
    """Records the pulse-width targets requested by the tracker."""

    def __init__(self) -> None:
        self.calls: List[Tuple[int, int]] = []
        self.center_pan = SERVO_CENTER
        self.center_tilt = SERVO_CENTER

    def move(self, pan: int, tilt: int, slew_limit: int = 20) -> None:
        self.calls.append((int(pan), int(tilt)))

    def center(self) -> None:
        self.move(self.center_pan, self.center_tilt)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(tracker.time, "time", fake)
    return fake


# ------------------------------------------------------------- One Euro
def test_one_euro_first_sample_passes_through():
    f = OneEuroFilter()
    assert f.filter(12.5, timestamp=0.0) == 12.5


def test_one_euro_smooths_a_stationary_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    f.filter(100.0, timestamp=t)
    outputs = []
    for i in range(20):  # +/- 1 px detector jitter around 100
        t += 1 / 30.0
        outputs.append(f.filter(100.0 + (1 if i % 2 else -1), timestamp=t))
    # A still face should produce a much smaller wobble than the raw signal.
    assert max(outputs) - min(outputs) < 0.3
    assert abs(outputs[-1] - 100.0) < 0.3


def test_one_euro_follows_a_fast_move_with_less_lag_than_ema():
    """Fast motion must not be smoothed away like a plain EMA would."""
    dt = 1 / 30.0
    target = [float(i * 30) for i in range(40)]  # ~900 px/s sweep
    euro = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    ema = 0.3  # weight on the new sample, i.e. lag-heavy
    prev = None
    euro_lag, ema_out = 0.0, 0.0
    for i, value in enumerate(target):
        t = i * dt
        out = euro.filter(value, timestamp=t)
        euro_lag = max(euro_lag, abs(target[i] - out))
        prev = value if prev is None else ema * value + (1 - ema) * prev
        ema_out = abs(target[i] - prev)
    assert euro_lag < ema_out


def test_one_euro_reset_restarts_from_the_new_value():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f.filter(10.0, timestamp=0.0)
    f.filter(10.0, timestamp=0.1)
    f.reset()
    assert f.filter(55.0, timestamp=0.2) == 55.0


def test_one_euro_handles_duplicate_timestamps():
    f = OneEuroFilter()
    f.filter(1.0, timestamp=0.0)
    out = f.filter(2.0, timestamp=0.0)
    assert 1.0 <= out <= 2.0


# --------------------------------------------------------------- helpers
def test_clamp_pwm_keeps_servo_pulse_widths_in_range():
    assert clamp_pwm(0) == tracker.SERVO_MIN
    assert clamp_pwm(99999) == tracker.SERVO_MAX
    assert clamp_pwm(1500) == 1500
    assert clamp_pwm(1500.6) == 1501


# -------------------------------------------------------------- tracker
def test_face_is_reported_locked_when_centred(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, deadband_px=12.0)
    searching, status = trk.update((290, 210, 60, 60), 640, 480, 1 / 30)
    assert searching is False
    assert "locked" in status
    assert motor.calls == []  # no correction needed


def test_off_centre_face_drives_the_motor_the_right_way(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, deadband_px=10.0)
    # Face centre at x=120 (left of 320): the mount must pan towards it.
    trk.update((80, 200, 80, 80), 640, 480, 1 / 30)
    assert motor.calls, "expected a pan correction"
    pan, tilt = motor.calls[-1]
    assert pan > SERVO_CENTER  # face is left of centre -> pan left
    assert abs(tilt - SERVO_CENTER) < 1  # vertical error is zero

    motor.calls.clear()
    clock.tick(0.2)  # past the command throttle window
    trk.update((540, 200, 80, 80), 640, 480, 1 / 30)  # face now right of centre
    pan, _ = motor.calls[-1]
    assert pan < SERVO_CENTER


def test_tilt_error_drives_tilt_axis(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, deadband_px=10.0)
    trk.update((290, 40, 60, 60), 640, 480, 1 / 30)  # face high in frame
    _, tilt = motor.calls[-1]
    assert tilt != SERVO_CENTER


def test_small_errors_are_ignored_by_the_deadband(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, deadband_px=20.0)
    for _ in range(5):
        clock.tick(1 / 30)
        trk.update((309, 229, 60, 60), 640, 480, 1 / 30)  # ~11 px off centre
    assert motor.calls == []


def test_commands_are_throttled_to_avoid_flooding_the_serial_port(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, command_interval_s=0.1, deadband_px=0.0)
    for _ in range(30):  # 30 frames at 60 Hz = 0.5 s of input
        clock.tick(1 / 60)
        trk.update((20, 20, 80, 80), 640, 480, 1 / 60)
    # 0.5 s at ~10 commands/s is about five writes, not thirty.
    assert len(motor.calls) <= 7


def test_repeated_identical_targets_are_not_resent(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, command_deadband_us=6, deadband_px=0.0)
    trk.update((100, 100, 80, 80), 640, 480, 1 / 30)
    first = len(motor.calls)
    for _ in range(10):
        clock.tick(0.2)  # plenty of time has passed
        trk.update((100, 100, 80, 80), 640, 480, 1 / 30)
    assert len(motor.calls) == first  # target unchanged -> no writes


def test_lost_face_waits_then_sweeps(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, search_delay_s=0.5)
    trk.update((290, 210, 60, 60), 640, 480, 1 / 30)

    searching, status = trk.update(None, 640, 480, 1 / 30)
    assert searching is False and "waiting" in status

    clock.tick(0.6)
    searching, status = trk.update(None, 640, 480, 1 / 30)
    assert searching is True and "sweep" in status.lower()
    assert motor.calls, "search should move the mount"


def test_sweep_stays_within_servo_limits(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, search_delay_s=0.0)
    clock.tick(1.0)
    for i in range(200):
        clock.tick(0.05)
        trk.update(None, 640, 480, 0.05)
    for pan, tilt in motor.calls:
        assert tracker.SERVO_MIN <= pan <= tracker.SERVO_MAX
        assert tracker.SERVO_MIN <= tilt <= tracker.SERVO_MAX


def test_reacquiring_a_face_resumes_tracking(clock):
    motor = FakeMotor()
    trk = FaceTracker(motor=motor, search_delay_s=0.2)
    trk.update((290, 210, 60, 60), 640, 480, 1 / 30)
    clock.tick(0.5)
    trk.update(None, 640, 480, 1 / 30)
    assert trk.searching is True

    clock.tick(0.1)
    searching, _ = trk.update((291, 211, 60, 60), 640, 480, 1 / 30)
    assert searching is False
    assert trk.searching is False
    # The filter restarts from the re-acquired position instead of easing in
    # from the stale sweep-time value.
    assert trk.face_center[0] == pytest.approx(321.0, abs=1.0)


def test_tracker_without_motor_still_reports_status(clock):
    trk = FaceTracker(motor=None)
    searching, status = trk.update((100, 100, 60, 60), 640, 480, 1 / 30)
    assert searching is False and status
    clock.tick(2.0)
    searching, status = trk.update(None, 640, 480, 1 / 30)
    assert searching is True  # sweeps logically even with no hardware


# ----------------------------------------------------- motor controller
def test_controller_clamps_and_slews(fake_protocol_factory=None):
    writes: List[Tuple[int, int]] = []

    class FakeProtocol:
        def __init__(self, port, baud):
            self.port, self.baud = port, baud

        def write(self, pan, tilt):
            writes.append((pan, tilt))

        def close(self):
            pass

    motor = PanTiltController(
        port="/dev/null", baud=115200, protocol_factory=FakeProtocol
    )
    motor.move(2400, 1500, slew_limit=20)
    # A 900 us jump only moves the limited distance in one call.
    assert motor.pan == SERVO_CENTER + 20
    for _ in range(100):
        motor.move(2400, 1500, slew_limit=20)
    assert motor.pan == 2400
    assert writes, "expected protocol writes"


def test_controller_survives_write_failures():
    class BrokenProtocol:
        def __init__(self, port, baud):
            pass

        def write(self, pan, tilt):
            raise OSError("Device not configured")

        def close(self):
            pass

    motor = PanTiltController(port="/dev/null", baud=115200, protocol_factory=BrokenProtocol)
    assert motor.ok is False
    assert motor.last_error
    motor.move(1600, 1400)  # must not raise
