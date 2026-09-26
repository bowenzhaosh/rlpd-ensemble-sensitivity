#!/bin/bash
# Rebuild the released paper from the shipped logs. No network access by default.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'HELP'
Usage: bash analysis/run_all.sh [--local] [--no-pdf]

  --local   Use the shipped logs (default; no SSH or network access).
  --no-pdf  Generate validated tables and figures without requiring LaTeX.
  --help    Show this message.

Activate the analysis environment first, or set RLPD_PY to its Python executable.
Install dependencies with: python -m pip install -r requirements/analysis.txt
HELP
}

PDF=true
for arg in "$@"; do
  case "$arg" in
    --local) ;;
    --no-pdf) PDF=false ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${RLPD_PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
  echo "ERROR: Python executable not found: $PY" >&2; exit 1;
}
"$PY" -c 'import numpy, pandas, scipy, matplotlib' || {
  echo "ERROR: install requirements/analysis.txt into $PY's environment" >&2; exit 1;
}
if $PDF && ! command -v latexmk >/dev/null 2>&1; then
  echo "ERROR: latexmk is required for the PDF; install LaTeX or pass --no-pdf" >&2
  exit 1
fi

"$PY" analysis/validate_data.py
"$PY" analysis/build_tidy.py
"$PY" analysis/make_figures.py
"$PY" analysis/make_tables.py

if $PDF; then
  mkdir -p paper/build
  if ! (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=build main.tex >build/latexmk.log 2>&1); then
    tail -40 paper/build/latexmk.log >&2
    echo "ERROR: LaTeX failed; see paper/build/latexmk.log and main.log" >&2
    exit 1
  fi
  echo "Built paper/build/main.pdf"
else
  echo "Built tables and figures; PDF compilation skipped (--no-pdf)."
fi
