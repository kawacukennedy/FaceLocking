#!/usr/bin/env bash
# Run an `src` module from any directory. Resolves the repository root from
# this script's location, so you never need to remember the working directory
# or the virtualenv path.
#
# Usage:
#   bash scripts/run.sh <module> [args...]
#   bash scripts/run.sh track          # -> python -m src.track
#   bash scripts/run.sh enroll         # -> python -m src.enroll
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -x "$repo_root/.venv/bin/python" ]]; then
    pybin="$repo_root/.venv/bin/python"
else
    pybin="python3"
fi

module="${1:?Usage: scripts/run.sh <module> [args...]}"
shift || true

exec "$pybin" -m "${module#src.}" "$@"