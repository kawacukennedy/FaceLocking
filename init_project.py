# Syntax: python init_project.py
# Creates the canonical project structure. Safe to re-run: existing files are kept.

from pathlib import Path

STRUCTURE = {
    "data/enroll": [],
    "data/db": [],
    "models": ["embedder_arcface.onnx"],
    "src": [
        "camera.py",
        "detect.py",
        "landmarks.py",
        "align.py",
        "embed.py",
        "enroll.py",
        "recognize.py",
        "evaluate.py",
        "haar_5pt.py",
        "tracker.py",
        "track.py",
    ],
    "tests": [],
    "scripts": [],
    "book": [],
}

PROJECT_NAME = "FaceLocking"


def main() -> None:
    for folder, files in STRUCTURE.items():
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            file_path = folder_path / file
            if not file_path.exists():
                file_path.touch()
    for marker in (Path("src") / "__init__.py", Path("tests") / "__init__.py"):
        if not marker.exists():
            marker.touch()
    print(f"{PROJECT_NAME} project structure created successfully.")


if __name__ == "__main__":
    main()