"""Test helpers and pipeline entry point for the test-suite."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_FACE = PROJECT_ROOT / "tests" / "data" / "sample_face.jpg"