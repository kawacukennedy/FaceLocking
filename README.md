# FaceLocking

Face recognition with ArcFace (ONNX) and 5-point alignment.

A self-contained, CPU-friendly face recognition system: MediaPipe FaceMesh
detection and 5-point landmarks, similarity-transform alignment onto the
standard ArcFace template, and ResNet-50 (w600k) embedding extraction through
ONNX Runtime. Multi-identity enrollment, live recognition, and pan/tilt face
tracking are provided as small, composable command-line applications.

This is the Week-01 practical submission for the cloud / face-recognition
course by Gabriel Baziramwabo (Benax Technologies).

## Pipeline

```
camera frame (BGR)
  -> MediaPipe FaceMesh (full frame): face + 5 landmarks
       [left_eye(33), right_eye(263), nose(1), left_mouth(61), right_mouth(291)]
  -> Haar cascade used as a fast fallback when FaceMesh finds nothing
  -> similarity transform (estimateAffinePartial2D) -> 112x112 canonical crop
  -> ArcFace ResNet-50 w600k ONNX -> 512-D embedding (L2-normalized)
  -> cosine distance against enrolled templates -> identity
```

FaceMesh is used as the primary detector because it is far more robust to pose
and lighting than a Haar cascade; the Haar cascade remains as an additional
recall path (and as the only detector in the minimal `src/detect.py` demo).

Landmark order is a strict contract across the whole project; see
`src/haar_5pt.py` (`ARCFACE_TEMPLATE_112`, `estimate_norm_5pt`,
`align_face_5pt`).

Embeddings are matched with cosine distance `1 - cos(a, b)` (the dot product
for unit vectors). Database defaults:

| item | path |
| --- | --- |
| identity templates | `data/db/face_db.npz` |
| database metadata | `data/db/face_db.json` |
| aligned enrollment crops | `data/enroll/<name>/*.jpg` |
| recognition threshold (dist) | `0.20` (tunable live with `+` / `-`) |

## Components

| module | purpose |
| --- | --- |
| `src/camera.py` | frame reader: auto-detects a USB vs built-in camera and transparently flips upside-down USB cameras |
| `src/detect.py` | minimal Haar face detection demo |
| `src/landmarks.py` | 5-point landmark overlay demo |
| `src/align.py` | 112x112 canonical crop demo (save with `s`) |
| `src/embed.py` | ArcFace ONNX embedding demo + embedder classes |
| `src/enroll.py` | multi-sample enrollment of a person |
| `src/recognize.py` | live multi-face recognition |
| `src/evaluate.py` | threshold sweep to calibrate FAR / FRR |
| `src/haar_5pt.py` | shared core: FaceMesh/Haar detection, landmark order, alignment math |
| `src/tracker.py` | pan/tilt motor controller driver + face-tracking PID on a serial mount |
| `src/track.py` | live demo: spins the mount to find a face, then keeps it centered |

## Requirements

- Python 3.11+ (tested on macOS with Python 3.11)
- A webcam (a USB camera and the built-in camera are detected automatically)
- `ffmpeg` on `PATH` (used only when OpenCV returns black frames on some
  USB cameras; detection is automatic, see `src/camera.py`)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
```

`download_models.sh` downloads the InsightFace `buffalo_l` bundle and extracts
`w600k_r50.onnx` to `models/embedder_arcface.onnx` (~171 MB). The ONNX model is
git-ignored; it is re-downloadable so the repository stays self-contained.

## Usage

All stages run as single-folder modules **from the repository root**:

```bash
cd FaceLocking
source .venv/bin/activate
```

Or, from *any* directory, use the included launcher (resolves the repo root
and the project virtualenv for you):

```bash
bash FaceLocking/scripts/run.sh track    # -> python -m src.track
```

Detect and inspect each stage:

```bash
python -m src.detect       # Haar bounding box
python -m src.landmarks    # 5-point landmarks
python -m src.align        # canonical 112x112 crop
python -m src.embed        # live 512-D embedding + heatmap
python -m src.haar_5pt     # detector + landmarks overlay
```

### Enroll a person

```bash
python -m src.enroll
```

Enter a name, then collect samples:

| key | action |
| --- | --- |
| `Space` | capture one sample (requires a detected face) |
| `a` | toggle auto-capture (every 0.25 s) |
| `s` | save the identity template to the database |
| `r` | reset NEW samples (existing ones are kept) |
| `q` | quit |

Enrollment crops are stored under `data/enroll/<name>/` so templates can be
recomputed offline (`evaluate.py` reuses them). Saving continues an existing
template automatically.

### Recognize

```bash
python -m src.recognize
```

> Note: if you get `ModuleNotFoundError: No module named 'src'`, you are running
> from the wrong folder — switch into the repository root first, or use
> `bash scripts/run.sh recognize`.

| key | action |
| --- | --- |
| `q` | quit |
| `r` | reload the database from disk |
| `+` / `-` | loosen / tighten the distance threshold |
| `d` | toggle the debug overlay |

### Calibrate the threshold

```bash
python -m src.evaluate
```

Requires aligned crops for at least two people under `data/enroll/`. Computes
genuine and impostor cosine-distance distributions and suggests a distance
threshold for a target 1% false-acceptance rate.

## Tests

```bash
pytest -v
```

The suite covers the alignment math (synthetic landmark warps), the ONNX
embedder (512-D, unit norm, stability), an end-to-end pipeline using a sample
face image, and live camera behavior (a non-black-frame check that exercises
the ffmpeg fallback). Tests that need the model or a camera skip cleanly when
they are unavailable.

## Camera notes

`src/camera.py` enumerates the avfoundation devices with `ffmpeg` and prefers
the first **external** camera over the laptop's built-in one, so the external
HD camera is used regardless of the index macOS assigns. Set
`PYCAMERA_DEVICE=Wed` to pin a specific device by name. Every demo prints the
device it opened on startup:

```
Camera: 'Wed Camera' [index 1, external] backend=opencv flip=OFF
```

The camera can be mounted upside down, so frames can optionally be rotated 180
degrees. Rotation is **off by default** (it follows the physical mount, not the
device name) and can be changed in three ways:

- press `f` in any demo to toggle it live,
- `PYCAMERA_FLIP=1 python -m src.enroll` to force it on,
- `Camera(flip=True)` in code.

Some USB cameras return all-black frames from the OpenCV backend; in that case
`Camera` warms up, detects the black frames (`mean brightness < 3.0`), and
transparently switches to an `ffmpeg` raw `bgr24` pipe, which yields healthy
frames. The built-in FaceTime camera works with either backend.

## Pan / tilt tracking (`src/track.py`)

A parallel port (a CP2102 USB-UART bridge) drives the servo pan/tilt mount.
`src/tracker.py` implements a robust driver:

- `P<pan> T<tilt>\n` protocol with servo pulse widths 500-2500 us (center 1500)
- slew limiting so the mount moves smoothly instead of instantly
- a P-controller that maps the face-center offset to pulse-width targets
- a search sweep that rotates the mount to find the face when it is lost
- a self-healing serial layer: if the device briefly drops off the USB bus,
  the controller re-opens the port and resyncs on the next move

Run it with:

```bash
python -m src.tracker   # cycle the mount to verify the controller is connected
python -m src.track     # live spin-to-find + face tracking
```

The tracker degrades gracefully when the mount is unplugged: tracking uses the
smoothed face box either way.

## Repository layout

```
FaceLocking/
  init_project.py          project scaffold generator
  src/                     pipeline modules (run with python -m src.<module>)
  scripts/download_models.sh
  tests/                   pytest suite
  data/                    generated at runtime (git-ignored)
  models/embedder_arcface.onnx   downloaded model (git-ignored)
  book/                    course reference material (git-ignored)
```

## License

MIT. See `LICENSE`.