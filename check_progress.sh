#!/bin/bash
# ================================================================
# check_progress.sh — Monitor experiment status and results.
#
# Usage:
#   bash check_progress.sh          # full summary
#   bash check_progress.sh --short  # just counts
# ================================================================

SHORT=false
[ "${1:-}" = "--short" ] && SHORT=true
RESULTS_DIR="results"

# --- Queue ---
echo "=========================================="
echo "CLUSTER QUEUE"
echo "=========================================="
if command -v squeue &>/dev/null; then
  RUNNING=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
  PENDING=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l | tr -d ' ')
  echo "  Running: $RUNNING | Pending: $PENDING"
  if [ "$SHORT" = false ] && [ "$((RUNNING + PENDING))" -gt 0 ]; then
    echo ""
    squeue -u "$USER" -o "  %10i %10j %8T %10M %R" 2>/dev/null
  fi
else
  echo "  (not on cluster)"
fi

# --- Results ---
echo ""
echo "=========================================="
echo "COMPLETED RESULTS"
echo "=========================================="
COMPLETED=0
if [ -d "$RESULTS_DIR" ]; then
  for d in "$RESULTS_DIR"/*/; do
    [ -d "$d" ] || continue
    summary="$d/summary.json"
    if [ -f "$summary" ]; then
      COMPLETED=$((COMPLETED + 1))
      if [ "$SHORT" = false ]; then
        name=$(basename "$d")
        score=$(grep '"final_score"' "$summary" | grep -o '[-0-9.]*' | head -1)
        peak=$(grep '"peak_score"' "$summary" | grep -o '[-0-9.]*' | head -1)
        hours=$(grep '"wall_hours"' "$summary" | grep -o '[0-9.]*' | head -1)
        printf "  %-50s score=%-8s peak=%-8s %sh\n" "$name" "$score" "$peak" "$hours"
      fi
    elif [ -f "$d/online_log.csv" ] && [ "$SHORT" = false ]; then
      lines=$(wc -l < "$d/online_log.csv" | tr -d ' ')
      printf "  %-50s IN PROGRESS (%s evals)\n" "$(basename "$d")" "$lines"
    fi
  done
fi
echo "  Total completed: $COMPLETED"

# --- Errors ---
echo ""
echo "=========================================="
echo "JOB LOGS"
echo "=========================================="
TOTAL_LOGS=$(ls rlpd_*.txt 2>/dev/null | wc -l | tr -d ' ')
DONE_LOGS=$(grep -l "ALL DONE" rlpd_*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "  Output files: $TOTAL_LOGS | Finished: $DONE_LOGS"

NONEMPTY_ERR=0
for f in rlpd_*_err.txt 2>/dev/null; do
  [ -s "$f" ] && NONEMPTY_ERR=$((NONEMPTY_ERR + 1))
done
if [ "$NONEMPTY_ERR" -gt 0 ]; then
  echo "  Error files with content: $NONEMPTY_ERR"
  if [ "$SHORT" = false ]; then
    for f in rlpd_*_err.txt; do
      [ -s "$f" ] && echo "    $f"
    done
  fi
fi

if [ "$DONE_LOGS" -gt 0 ] && [ "$SHORT" = false ]; then
  echo ""
  echo "=========================================="
  echo "FINAL SCORES"
  echo "=========================================="
  grep "FINAL" rlpd_*.txt 2>/dev/null | sed 's/^/  /'
fi
echo ""
