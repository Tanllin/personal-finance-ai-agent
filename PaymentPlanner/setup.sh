#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
else
    echo "Payment Planner requires Python 3.10 or newer."
    exit 1
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Add OPENAI_API_KEY or start Ollama."
fi

echo "Payment Planner environment is ready."
echo "Activate it with: source .venv/bin/activate"

