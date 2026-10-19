#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Environment missing. Run: uv sync --python 3.12 --extra validation'
  exit 1
fi
exec .venv/bin/python -m ceap serve "$@"
