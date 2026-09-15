"""Threshold evaluation using enrolled aligned crops.

Computes cosine-distance distributions for genuine pairs (same person) and
impostor pairs (different people), then sweeps candidate thresholds and
suggests one for a target false-accept rate (FAR, default 1%).

Run::

    python -m src.evaluate

Requires aligned crops saved during enrollment under ``data/enroll/<name>/``.
Enroll at least two people for a meaningful impostor distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, DEFAULT_MODEL_PATH


@dataclass
class EvalConfig:
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80
    target_far: float = 0.01
    thresholds: Tuple[float, float, float] = (0.10, 1.20, 0.01)
    require_size: Tuple[int, int] = (112, 112)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1).astype(np.float32), b.reshape(-1).astype(np.float32)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def list_people(cfg: EvalConfig) -> List[Path]:
    if not cfg.enroll_dir.exists():
        raise FileNotFoundError(
            f"Enroll dir not found: {cfg.enroll_dir}. Run enrollment first."
        )
    return sorted(p for p in cfg.enroll_dir.iterdir() if p.is_dir())


def load_embeddings_for_person(
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
    cfg: EvalConfig,
) -> List[np.ndarray]:
    images = sorted(p for p in person_dir.glob("*.jpg") if p.is_file())[: cfg.max_imgs_per_person]
    embeddings: List[np.ndarray] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if cfg.require_size is not None and img.shape[:2] != (cfg.require_size[1], cfg.require_size[0]):
            continue
        embeddings.append(embedder.embed(img).embedding)
    return embeddings


def pairwise_distances(
    embeddings_a: List[np.ndarray],
    embeddings_b: List[np.ndarray],
    same: bool,
) -> List[float]:
    distances: List[float] = []
    if same:
        for i in range(len(embeddings_a)):
            for j in range(i + 1, len(embeddings_a)):
                distances.append(cosine_distance(embeddings_a[i], embeddings_a[j]))
    else:
        for ea in embeddings_a:
            for eb in embeddings_b:
                distances.append(cosine_distance(ea, eb))
    return distances


def sweep_thresholds(
    genuine: np.ndarray,
    impostor: np.ndarray,
    cfg: EvalConfig,
):
    start, stop, step = cfg.thresholds
    rows = []
    for thr in np.arange(start, stop + 1e-9, step, dtype=np.float32):
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        rows.append((float(thr), far, frr))
    return rows


def describe(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
        f"p05={np.percentile(arr, 5):.3f} p50={np.percentile(arr, 50):.3f} "
        f"p95={np.percentile(arr, 95):.3f}"
    )


def main() -> None:
    cfg = EvalConfig()
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL_PATH)

    people_dirs = list_people(cfg)
    if not people_dirs:
        print("No enrolled people found.")
        return

    per_person: Dict[str, List[np.ndarray]] = {}
    for person_dir in people_dirs:
        name = person_dir.name
        embeddings = load_embeddings_for_person(embedder, person_dir, cfg)
        if len(embeddings) >= cfg.min_imgs_per_person:
            per_person[name] = embeddings
        else:
            print(f"Skipping {name}: only {len(embeddings)} valid aligned crops "
                  f"(need >= {cfg.min_imgs_per_person}).")

    names = sorted(per_person.keys())
    if not names:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    genuine_all: List[float] = []
    for name in names:
        genuine_all.extend(pairwise_distances(per_person[name], per_person[name], same=True))

    impostor_all: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_all.extend(
                pairwise_distances(per_person[names[i]], per_person[names[j]], same=False)
            )

    genuine = np.array(genuine_all, dtype=np.float32)
    impostor = np.array(impostor_all, dtype=np.float32)

    print("\n=== Distance Distributions (cosine distance = 1 - cosine similarity) ===")
    print(f"Genuine (same person): {describe(genuine)}")
    print(f"Impostor (diff persons): {describe(impostor)}")

    results = sweep_thresholds(genuine, impostor, cfg)
    best = None
    for thr, far, frr in results:
        if far <= cfg.target_far:
            if best is None or frr < best[2]:
                best = (thr, far, frr)

    print("\n=== Threshold Sweep ===")
    stride = max(1, len(results) // 10)
    for thr, far, frr in results[::stride]:
        print(f"thr={thr:.2f} FAR={far * 100:5.2f}% FRR={frr * 100:5.2f}%")

    if best is not None:
        thr, far, frr = best
        print(
            f"\nSuggested threshold (target FAR {cfg.target_far * 100:.1f}%): "
            f"thr={thr:.2f} FAR={far * 100:.2f}% FRR={frr * 100:.2f}%"
        )
        print(f"\n(Equivalent cosine similarity threshold ~ {1.0 - thr:.3f}, since sim = 1 - dist)")
    else:
        print(
            f"\nNo threshold in range met FAR <= {cfg.target_far * 100:.1f}%. "
            "Try widening the sweep range or collecting more varied samples."
        )
    print()


if __name__ == "__main__":
    main()