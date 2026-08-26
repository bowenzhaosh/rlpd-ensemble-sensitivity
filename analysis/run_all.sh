#!/bin/bash
# Rebuild everything from the run logs:
#   bash analysis/run_all.sh --local  # offline: shipped logs -> tidy CSVs -> figures -> tables -> PDF
#   bash analysis/run_all.sh          # same, after rsyncing fresh logs from the cluster (collect_results.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
# RLPD_PY selects the interpreter (needs pandas/scipy/matplotlib, see analysis/requirements.txt);
# defaults to the authors' pyenv, falls back to python3.
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
     || { echo "LaTeX FAILED -- see paper/build/main.log"; exit 1; })
fi

echo
echo "Done. Check analysis/out/tidy/progress.json for fleet completeness."
