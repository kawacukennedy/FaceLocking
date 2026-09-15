#!/usr/bin/env bash
# Download the ArcFace (ResNet-50, w600k) ONNX embedding model.
#
# Usage:
#   bash scripts/download_models.sh
#
# The script downloads insightface's buffalo_l bundle, keeps only the ArcFace
# ONNX model, and places it at models/embedder_arcface.onnx.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p models
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading buffalo_l (ArcFace model bundle)..."
curl -L --fail --retry 3 -o "$TMP/buffalo_l.zip" \
  "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"

echo "Extracting w600k_r50.onnx..."
unzip -o "$TMP/buffalo_l.zip" w600k_r50.onnx -d "$TMP"

mv "$TMP/w600k_r50.onnx" models/embedder_arcface.onnx
echo "Done: models/embedder_arcface.onnx"