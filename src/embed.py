"""Embedding extraction with ArcFace (ONNX) on CPU.

The embedder consumes an aligned 112x112 BGR crop and returns an L2-normalized
feature vector (512-D for the ResNet-50 w600k model). Preprocessing follows the
ArcFace convention: BGR -> RGB, ``(x - 127.5) / 128.0``, NCHW float32.

Run the live validation with::

    python -m src.embed

Keys:
    q  quit
    p  print embedding statistics to the terminal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import align_face_5pt, Haar5ptDetector

DEFAULT_MODEL_PATH = "models/embedder_arcface.onnx"


@dataclass
class EmbeddingResult:
    """An L2-normalized embedding plus normalization diagnostics.

    Attributes:
        embedding: ``(D,)`` float32 unit vector.
        norm_before: L2 norm of the raw network output prior to normalization.
        dim: Embedding dimensionality ``D``.
    """

    embedding: np.ndarray
    norm_before: float
    dim: int


class ArcFaceEmbedderONNX:
    """ArcFace / InsightFace-style ONNX embedder.

    Args:
        model_path: Path to the ONNX model (defaults to the bundled path).
        input_size: ``(width, height)`` of the expected aligned input.
        debug: Print model metadata on construction.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ) -> None:
        self.input_width, self.input_height = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        if debug:
            print("[embed] model loaded")
            print("[embed] input:", self.session.get_inputs()[0].shape)
            print("[embed] output:", self.session.get_outputs()[0].shape)

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.input_height, self.input_width):
            aligned_bgr = cv2.resize(
                aligned_bgr, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
            )
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32)

    @staticmethod
    def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> tuple:
        vector = vector.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector) + eps)
        return (vector / norm).astype(np.float32), norm

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        tensor = self._preprocess(aligned_bgr)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        raw = np.asarray(output, dtype=np.float32).reshape(-1)
        embedding, norm_before = self.l2_normalize(raw)
        return EmbeddingResult(embedding=embedding, norm_before=norm_before, dim=embedding.size)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings (i.e. dot product)."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def draw_text_block(
    frame: np.ndarray,
    lines,
    origin=(10, 30),
    scale: float = 0.7,
    color=(0, 255, 0),
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(28 * scale)


def draw_embedding_matrix(
    frame: np.ndarray,
    embedding: np.ndarray,
    top_left=(10, 220),
    cell_scale: int = 6,
    title: str = "embedding",
):
    """Visualize an embedding as a grayscale heatmap (educational only)."""
    dim = embedding.size
    cols = int(np.ceil(np.sqrt(dim)))
    rows = int(np.ceil(dim / cols))
    mat = np.zeros((rows, cols), dtype=np.float32)
    mat.flat[:dim] = embedding
    normed = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    gray = (normed * 255).astype(np.uint8)
    heat = cv2.resize(
        cv2.applyColorMap(gray, cv2.COLORMAP_JET),
        (cols * cell_scale, rows * cell_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    x, y = top_left
    h, w = heat.shape[:2]
    img_h, img_w = frame.shape[:2]
    if x + w > img_w or y + h > img_h:
        return 0, 0
    frame[y : y + h, x : x + w] = heat
    cv2.putText(frame, title, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2)
    return w, h


def main() -> None:
    from .camera import Camera

    detector = Haar5ptDetector(min_size=(48, 48), smooth_alpha=0.80)
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL_PATH, debug=True)
    prev_emb: Optional[np.ndarray] = None

    with Camera() as camera:
        print("Embedding demo running. q=quit, p=print embedding.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            vis = frame.copy()
            info = []
            faces = detector.detect(frame, max_faces=1)
            if faces:
                face = faces[0]
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                result = embedder.embed(aligned)
                info.append(f"embedding dim: {result.dim}")
                info.append(f"norm(before L2): {result.norm_before:.2f}")
                if prev_emb is not None:
                    info.append(f"cos(prev,this): {cosine_similarity(prev_emb, result.embedding):.3f}")
                prev_emb = result.embedding

                thumb = cv2.resize(aligned, (160, 160))
                h, w = vis.shape[:2]
                vis[10:170, w - 170 : w - 10] = thumb
                draw_text_block(vis, info, origin=(10, 30))
                draw_embedding_matrix(vis, result.embedding)
            else:
                draw_text_block(vis, ["no face"], origin=(10, 30), color=(0, 0, 255))

            cv2.imshow("Face Embedding", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p") and prev_emb is not None:
                print("[embedding]")
                print(" dim:", prev_emb.size)
                print(" min/max:", prev_emb.min(), prev_emb.max())
                print(" first10:", prev_emb[:10])


if __name__ == "__main__":
    main()