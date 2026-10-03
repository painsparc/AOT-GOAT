#!/usr/bin/env bash
# One-step setup + test + run.  Usage: ./run.sh   (add "test" to only run tests, "eval" to only run the evaluation)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
case "${1:-run}" in
  test) python test_pipeline.py ;;
  eval) python evaluate.py ;;
  *)    python test_pipeline.py 2>&1 | tail -4; echo "Open http://127.0.0.1:${PORT:-8000}"; python app.py ;;
esac
