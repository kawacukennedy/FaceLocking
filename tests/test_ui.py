"""Overlay drawing smoke tests.

The camera apps draw onto NumPy frames, so every helper is exercised here on a
synthetic frame. These catch clipping mistakes, bad anchors and helpers that
mutate or crash on a normal video-sized array.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import ui


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((480, 640, 3), 40, dtype=np.uint8)


def test_draw_hud_does_not_crash_and_keeps_pixels(frame):
    before = frame.copy()
    bottom = ui.draw_hud(frame, [("FACELOCKING", ui.WHITE), ("1 FACE(S) DETECTED", ui.GREEN)])
    assert bottom > 16
    assert not np.array_equal(frame, before)


def test_draw_hud_handles_no_lines(frame):
    assert ui.draw_hud(frame, []) == 16


def test_badge_anchors_stay_in_frame(frame):
    h, w = frame.shape[:2]
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    for x, y in corners:
        for anchor in ("top-left", "top-right", "bottom-left", "bottom-right"):
            x1, y1, x2, y2 = ui.draw_badge(frame, "Kennedy", (x, y), anchor=anchor)
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h


def test_panel_is_clipped_to_the_frame(frame):
    # Deliberately oversized: must not raise or write out of bounds.
    ui.draw_panel(frame, -50, -50, 10_000, 10_000)
    ui.draw_panel(frame, 700, 500, 100, 100)
    assert frame.shape == (480, 640, 3)


def test_meter_clamps_out_of_range_values(frame):
    ui.draw_meter(frame, -3.0, (20, 100))
    ui.draw_meter(frame, 12.0, (20, 140), label="smile")
    assert frame.any()


def test_corner_box_accepts_float_and_int_input(frame):
    ui.draw_corner_box(frame, (10, 10, 100.4, 120.6))
    ui.draw_corner_box(frame, (10, 10, 5, 5), length=50)  # length > box size


def test_crosshair_and_legend(frame):
    ui.draw_crosshair(frame, 320, 240, radius=12)
    ui.draw_legend(frame, [("smile", ui.GREEN), ("blink", ui.ORANGE)], (450, 400))


def test_vignette_darkens_edges_but_keeps_shape(frame):
    flat = np.full((120, 160, 3), 200, dtype=np.uint8)
    ui.vignette(flat, strength=0.5)
    assert flat[0, 0, 0] < flat[60, 80, 0]
    assert flat.shape == (120, 160, 3)


def test_vignette_ignores_tiny_frames():
    tiny = np.full((4, 4, 3), 100, dtype=np.uint8)
    ui.vignette(tiny)
    assert tiny.max() <= 100


def test_text_size_matches_open_cv(frame):
    text = "dist 0.042"
    (w, h), _ = cv2.getTextSize(text, ui.FONT, 0.5, 1)
    assert ui.text_size(text, 0.5, 1) == (w, h)
