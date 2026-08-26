#!/bin/bash
# Idempotent sync of the June-2026 WashU fleet results into data/washu-202606/.
# Assumes an ssh alias `washu` for the cluster and the repo checked out at ~/rlpd_experiments there.
# Pulls only the small artifacts (online_log.csv, summary.json, diagnostic.csv).
# Safe to run while the fleet is live; partial runs are handled downstream.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/washu-202606/results analysis/out

rsync -az --prune-empty-dirs \
  --include='*/' \
  --include='summary.json' --include='online_log.csv' --include='diagnostic.csv' \
  --exclude='*' \
  washu:rlpd_experiments/results/ data/washu-202606/results/

# Best-effort queue snapshot for the provisional banner / progress report.
ssh washu 'squeue -u "$USER" -o "%.10i %.20j %.8T %.10M"' \
  > analysis/out/fleet_queue.txt 2>/dev/null || true

echo "run dirs:   $(ls data/washu-202606/results | wc -l | tr -d ' ')"
echo "summaries:  $(find data/washu-202606/results -name summary.json | wc -l | tr -d ' ')"
