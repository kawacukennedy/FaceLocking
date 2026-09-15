# Contributing

Thanks for taking the time to contribute. The project is intentionally small;
keep it that way.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
```

Run the test suite before any pull request:

```bash
pytest -v
```

Tests that depend on the downloaded model or a live camera skip themselves
when those resources are absent, so a green suite does not require hardware.

## Conventions

- Python 3.11+, type hints on public signatures, no comments that restate the
  code.
- Keep every module runnable as `python -m src.<name>` with its own small demo.
- Preserve the 5-point landmark contract and the ArcFace template exactly; it
  is what makes all stages interchangeable.
- Match with cosine distance (`1 - dot`); keep embeddings L2-normalized.
- Models and generated data stay git-ignored; provide download scripts rather
  than committing binaries.
- No emojis, no boilerplate, no generated doc noise.

## Pull requests

- One logical change per PR.
- Add or update a test that covers the change.
- Confirm `pytest -v` passes before requesting review.