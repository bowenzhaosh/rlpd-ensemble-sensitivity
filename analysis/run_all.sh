#!/bin/bash
# THE button. When the fleet finishes (~2026-06-15):
#   bash analysis/run_all.sh          # sync + tidy + figures + tables + compile paper
#   bash analysis/run_all.sh --local  # skip the washu sync (offline rebuild)
set -euo pipefail
cd "$(dirname "$0")/.."
# pyenv 3.11.7 carries pandas/scipy/matplotlib on this Mac (06-2026);
# fall back to whatever python3 has the stack.
PY="${RLPD_PY:-$HOME/.pyenv/versions/3.11.7/bin/python3}"
command -v "$PY" >/dev/null || PY=python3

if [[ "${1:-}" != "--local" ]]; then
  bash analysis/collect_results.sh
fi
$PY analysis/build_tidy.py
$PY analysis/make_figures.py
$PY analysis/make_tables.py

if command -v latexmk >/dev/null 2>&1 && [ -f paper/main.tex ]; then
  mkdir -p paper/build
  (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error \
     -outdir=build main.tex >/dev/null 2>&1 \
     && echo "paper/build/main.pdf compiled" \
     || echo "LaTeX FAILED -- run: cd paper && latexmk -pdf -outdir=build main.tex")
fi

echo
echo "Done. Check analysis/out/tidy/progress.json for fleet completeness."
