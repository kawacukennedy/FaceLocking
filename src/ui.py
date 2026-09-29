"""Shared OpenCV overlay drawing helpers.

One place for the look and feel of every demo window: translucent HUD panels,
text with an outline so it stays readable over any background, rounded badges
for labels, progress meters, corner-bracket face boxes and a target
crosshair. Keeping this in a single module means the recognize, tracking and
enrollment windows look like one application.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (245, 245, 245)
GREEN = (96, 224, 132)
RED = (96, 96, 248)
YELLOW = (72, 212, 245)
CYAN = (236, 224, 96)
ORANGE = (72, 156, 245)
GRAY = (186, 186, 186)
DARK = (24, 24, 24)


def text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    """Return ``(width, height)`` of ``text`` for the given font settings."""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def draw_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.58,
    color: Tuple[int, int, int] = WHITE,
    thickness: int = 1,
    outline: bool = True,
) -> int:
    """Draw outlined text; returns the vertical advance to the next line."""
    x, y = org
    if outline:
        cv2.putText(frame, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    _, h = text_size(text, scale, thickness)
    return h + 10


def draw_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int] = DARK,
    alpha: float = 0.55,
    radius: int = 10,
) -> None:
    """Draw a translucent rounded panel (clipped to the frame)."""
    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    layer = frame.copy()
    cv2.rectangle(layer, (x, y), (x + w, y + h), color, -1, cv2.LINE_AA)
    if radius > 0:
        cv2.rectangle(layer, (x + radius, y), (x + w - radius, y + h), color, -1, cv2.LINE_AA)
    roi = frame[y : y + h, x : x + w]
    cv2.addWeighted(layer[y : y + h, x : x + w], alpha, roi, 1.0 - alpha, 0, roi)


def draw_badge(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = GREEN,
    scale: float = 0.52,
    pad: int = 7,
    anchor: str = "top-left",
) -> Tuple[int, int, int, int]:
    """Draw a filled label. ``anchor`` picks the reference point of ``org``."""
    tw, th = text_size(text, scale, 1)
    bw, bh = tw + 2 * pad, th + 2 * pad
    x, y = org
    if anchor == "top-left":
        x1, y1 = x, y
    elif anchor == "top-right":
        x1, y1 = x - bw, y
    elif anchor == "bottom-left":
        x1, y1 = x, y - bh
    else:
        x1, y1 = x - bw, y - bh
    fh, fw = frame.shape[:2]
    x1 = max(0, min(x1, fw - bw))
    y1 = max(0, min(y1, fh - bh))
    sub = frame[y1 : y1 + bh, x1 : x1 + bw]
    layer = np.empty_like(sub)
    layer[:] = color
    cv2.addWeighted(layer, 0.80, sub, 0.20, 0, sub)
    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad, y1 + pad + th - 2), FONT, scale, WHITE, 1, cv2.LINE_AA)
    return x1, y1, x1 + bw, y1 + bh


def draw_meter(
    frame: np.ndarray,
    value: float,
    org: Tuple[int, int],
    width: int = 150,
    height: int = 10,
    color: Tuple[int, int, int] = CYAN,
    track: Tuple[int, int, int] = (70, 70, 70),
    label: Optional[str] = None,
) -> None:
    """Horizontal progress bar in ``[0, 1]`` with an optional label above it."""
    x, y = org
    value = float(min(1.0, max(0.0, value)))
    if label:
        draw_text(frame, label, (x, y - 6), scale=0.45, color=GRAY, thickness=1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), track, -1, cv2.LINE_AA)
    if value > 0:
        cv2.rectangle(frame, (x, y), (x + int(width * value), y + height), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + height), GRAY, 1, cv2.LINE_AA)


def draw_corner_box(
    frame: np.ndarray,
    box: Sequence[float],
    color: Tuple[int, int, int] = GREEN,
    thickness: int = 2,
    length: int = 22,
) -> None:
    """Draw a face box as four corner brackets (less clutter than a rectangle)."""
    x, y, w, h = box
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    x2, y2 = x + w, y + h
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * min(length, w), cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * min(length, h)), color, thickness, cv2.LINE_AA)


def draw_crosshair(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int = 10,
    color: Tuple[int, int, int] = CYAN,
    thickness: int = 1,
) -> None:
    """Draw the frame center used as the tracking target."""
    cv2.circle(frame, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx - radius - 8, cy), (cx - radius + 2, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx + radius - 2, cy), (cx + radius + 8, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - radius - 8), (cx, cy - radius + 2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + radius - 2), (cx, cy + radius + 8), color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    lines: Iterable[Tuple[str, Tuple[int, int, int]]],
    x: int = 14,
    y: int = 16,
    scale: float = 0.56,
    pad: int = 12,
    panel: bool = True,
) -> int:
    """Draw a top-left status stack; returns the y after the last line."""
    rendered: list[Tuple[str, Tuple[int, int, int]]] = []
    for text, color in lines:
        rendered.append((str(text), color))
    if not rendered:
        return y
    heights = [text_size(t, scale, 1)[1] for t, _ in rendered]
    total_h = sum(h + 8 for h in heights) + 2 * pad
    width = max(text_size(t, scale, 1)[0] for t, _ in rendered) + 2 * pad
    if panel:
        draw_panel(frame, x - pad, y - pad, width, total_h)
    cursor = y
    for (text, color), h in zip(rendered, heights):
        draw_text(frame, text, (x, cursor + h), scale=scale, color=color, thickness=1)
        cursor += h + 8
    return cursor


def draw_legend(
    frame: np.ndarray,
    entries: Sequence[Tuple[str, Tuple[int, int, int]]],
    org: Tuple[int, int],
    scale: float = 0.5,
    swatch: int = 16,
    gap: int = 22,
) -> None:
    """Small color legend, e.g. to explain badge colors."""
    x, y = org
    for text, color in entries:
        cv2.rectangle(frame, (x, y - swatch + 3), (x + swatch - 4, y), color, -1, cv2.LINE_AA)
        draw_text(frame, text, (x + swatch + 4, y), scale=scale, color=color, thickness=1)
        y += gap


def vignette(frame: np.ndarray, strength: float = 0.18) -> None:
    """Subtle darkening towards the frame edges to make overlays pop."""
    h, w = frame.shape[:2]
    if h < 8 or w < 8:
        return
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    d = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.707
    mask = np.clip(1.0 - strength * d, 0.0, 1.0)
    frame[:] = (frame.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
