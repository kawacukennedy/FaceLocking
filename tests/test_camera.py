"""Live camera tests: enumerate devices and assert the camera delivers usable frames.

The USB camera check exercises the pixel-quality fallback: on host machines
where the OpenCV backend returns black frames, :class:`Camera` silently switches
to the ffmpeg raw frame stream.
"""

import pytest

from src.camera import (
    Camera,
    auto_camera_index,
    ffmpeg_device_names,
    is_builtin_camera_name,
    list_camera_indices,
)


@pytest.mark.skipif(not list_camera_indices(), reason="no camera available")
def test_camera_listing():
    cameras = list_camera_indices()
    assert cameras
    index, backend, w, h = cameras[0]
    assert index >= 0
    assert w > 0 and h > 0
    assert backend


@pytest.mark.skipif(not list_camera_indices(), reason="no camera available")
def test_auto_index_prefers_usb_camera():
    names = ffmpeg_device_names()
    if not names:
        pytest.skip("ffmpeg device list unavailable")
    assert names
    builtins = [i for i, n in names if is_builtin_camera_name(n)]
    externals = [i for i, n in names if not is_builtin_camera_name(n)]
    if externals:
        assert auto_camera_index() in externals


@pytest.mark.skipif(not list_camera_indices(), reason="no camera available")
def test_usb_camera_captures_non_black_frames():
    means = []
    with Camera() as camera:
        assert camera.backend is not None
        for _ in range(15):
            ok, frame = camera.read()
            assert ok
            assert frame is not None
            means.append(float(frame.mean()))
    assert max(means) >= 3.0, "camera delivered only black frames"


@pytest.mark.skipif(not list_camera_indices(), reason="no camera available")
def test_ffmpeg_reader_directly():
    from src.camera import FFmpegFrameReader, ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg not on PATH")
    reader = FFmpegFrameReader(index=auto_camera_index())
    try:
        means = []
        for _ in range(10):
            ok, frame = reader.read()
            assert ok
            means.append(float(frame.mean()))
        assert max(means) >= 3.0
    finally:
        reader.release()