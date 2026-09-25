"""Enrollment: build a reference embedding template for a person.

Live pipeline: camera -> detect -> 5-point landmarks -> align (112x112) ->
ArcFace embedding -> mean template (L2-normalized).

Identities are stored in ``data/db/face_db.npz`` (name -> embedding vector)
with metadata in ``data/db/face_db.json``. Aligned crops are optionally saved
to ``data/enroll/<name>/`` as debugging artifacts; they are not part of the
recognition database.

Controls:
    SPACE  capture one sample (requires a detected face)
    a      toggle auto-capture
    s      save the enrollment template
    r      reset NEW samples (existing crops on disk are kept)
    f      toggle the 180 degree image rotation
    q      quit
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, DEFAULT_MODEL_PATH
from .haar_5pt import Haar5ptDetector, align_face_5pt


@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")
    save_crops: bool = True
    crops_dir: Path = Path("data/enroll")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.25
    max_existing_crops: int = 300
    window_main: str = "enroll"
    window_aligned: str = "aligned 112"


# --------------------------------------------------------------------------- DB
def ensure_dirs(cfg: EnrollConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if not cfg.out_db_npz.exists():
        return {}
    data = np.load(cfg.out_db_npz, allow_pickle=True)
    return {key: data[key].astype(np.float32) for key in data.files}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean of embeddings, then L2-normalized."""
    stack = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    mean = stack.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    return mean.astype(np.float32)


# ---------------------------------------------------------------- existing crops
def list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    if not person_dir.exists():
        return []
    files = sorted(p for p in person_dir.glob("*.jpg") if p.is_file())
    if len(files) > max_count:
        files = files[-max_count:]
    return files


def load_existing_samples_from_crops(
    cfg: EnrollConfig,
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
) -> List[np.ndarray]:
    """Re-embed aligned crops saved on disk during a previous enrollment."""
    if not cfg.save_crops:
        return []
    samples: List[np.ndarray] = []
    for path in list_existing_crops(person_dir, cfg.max_existing_crops):
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            samples.append(embedder.embed(img).embedding)
        except Exception:
            continue
    return samples


# ---------------------------------------------------------------------------- UI
def draw_status(
    frame: np.ndarray,
    name: str,
    base_count: int,
    new_count: int,
    needed: int,
    auto: bool,
    msg: str = "",
) -> None:
    total = base_count + new_count
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} / {needed}",
        f"Auto: {'ON' if auto else 'OFF'} (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]
    if msg:
        lines.insert(0, msg)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


def main() -> None:
    from .camera import Camera

    cfg = EnrollConfig()
    ensure_dirs(cfg)
    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return

    detector = Haar5ptDetector(min_size=(48, 48), smooth_alpha=0.80)
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL_PATH)
    db = load_db(cfg)
    person_dir = cfg.crops_dir / name
    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)

    base_samples = load_existing_samples_from_crops(cfg, embedder, person_dir)
    new_samples: List[np.ndarray] = []
    status_msg = ""
    if base_samples:
        status_msg = f"Loaded {len(base_samples)} existing samples from disk."

    auto = False
    last_auto = 0.0

    with Camera() as camera:
        cv2.namedWindow(cfg.window_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(cfg.window_aligned, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_aligned, 240, 240)

        if base_samples:
            print(f"Re-enroll mode: found {len(base_samples)} existing samples in {person_dir}/")
        print("Tip: keep lighting stable, move slightly left/right, use different expressions.")
        print("Controls: SPACE=capture, a=auto, s=save, r=reset NEW, f=flip, q=quit\n")

        try:
            print(camera.describe())
            print("\nEnrollment started.")
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                vis = frame.copy()
                aligned = None
                faces = detector.detect(frame, max_faces=1)
                if faces:
                    face = faces[0]
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                    for (px, py) in face.kps.astype(int):
                        cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
                    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                    cv2.imshow(cfg.window_aligned, aligned)
                else:
                    cv2.imshow(cfg.window_aligned, np.zeros((112, 112, 3), dtype=np.uint8))

                now = time.time()
                if auto and aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                    new_samples.append(embedder.embed(aligned).embedding)
                    last_auto = now
                    status_msg = f"Auto captured NEW ({len(new_samples)})"
                    if cfg.save_crops:
                        cv2.imwrite(str(person_dir / f"{int(now * 1000)}.jpg"), aligned)

                draw_status(vis, name, len(base_samples), len(new_samples),
                            cfg.samples_needed, auto, status_msg)
                cv2.imshow(cfg.window_main, vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    auto = not auto
                    status_msg = f"Auto mode {'ON' if auto else 'OFF'}"
                elif key == ord("r"):
                    new_samples.clear()
                    status_msg = "NEW samples reset (existing kept)."
                elif key == ord("f"):
                    status_msg = (
                        "Flip ON (rotated 180)." if camera.toggle_flip() else "Flip OFF."
                    )
                elif key == ord(" "):
                    if aligned is None:
                        status_msg = "No face detected. Not captured."
                    else:
                        new_samples.append(embedder.embed(aligned).embedding)
                        status_msg = f"Captured NEW ({len(new_samples)})"
                        if cfg.save_crops:
                            cv2.imwrite(str(person_dir / f"{int(time.time() * 1000)}.jpg"), aligned)
                elif key == ord("s"):
                    total = len(base_samples) + len(new_samples)
                    if total < max(3, cfg.samples_needed // 2):
                        status_msg = f"Not enough total samples to save (have {total})."
                        continue
                    template = mean_embedding(base_samples + new_samples)
                    db[name] = template
                    meta = {
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "embedding_dim": int(template.size),
                        "names": sorted(db.keys()),
                        "samples_existing_used": len(base_samples),
                        "samples_new_used": len(new_samples),
                        "samples_total_used": len(base_samples) + len(new_samples),
                        "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
                    }
                    save_db(cfg, db, meta)
                    status_msg = f"Saved '{name}' to DB. Total identities: {len(db)}"
                    print(status_msg)
                    base_samples = load_existing_samples_from_crops(cfg, embedder, person_dir)
                    new_samples.clear()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()