#!/bin/bash
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-12
#SBATCH --job-name=rlpd
#SBATCH --output=rlpd_%j.txt
#SBATCH --error=rlpd_%j_err.txt

# ================================================================
# SLURM job script for RLPD experiments.
# Called by submit_all.sh — you don't run this directly.
#
# EXPERIMENTS format: env,seed,nqs,min_qs,dropout,max_steps
#   Multiple sequential runs per job: separate with |
#   Dropout: 0 means off
#
# Optional: DIAG=1 for diagnostic Q-value logging
# ================================================================

# --- Environment ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlpd

REPO="$HOME/rlpd_experiments"
cd "$REPO"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

TRAIN_SCRIPT="train_abc.py"
if [ "${DIAG:-0}" = "1" ]; then
  TRAIN_SCRIPT="train_diagnostic.py"
fi

IFS='|' read -ra RUNS <<< "$EXPERIMENTS"
for run in "${RUNS[@]}"; do
  IFS=',' read -r env seed nqs minqs dropout maxsteps <<< "$run"

  DROP_FLAG=""
  if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
    DROP_FLAG="--config.critic_dropout_rate=$dropout"
  fi

  echo ""
  echo "=========================================="
  echo "RUN: env=$env seed=$seed nq=$nqs mq=$minqs drop=$dropout steps=$maxsteps diag=${DIAG:-0} — $(date)"
  echo "=========================================="

  python "$TRAIN_SCRIPT" \
    --env_name="$env" \
    --max_steps="$maxsteps" \
    --config=configs/rlpd_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs="$minqs" \
    --config.num_qs="$nqs" \
    --config.critic_layer_norm=True \
    $DROP_FLAG \
    --bootstrap_mask=False \
    --independent_targets=False \
    --critic_reset_step=0 \
    --seed="$seed" \
    --results_dir=results

  echo "FINISHED — $(date)"
done
echo "=== ALL DONE $(date) ==="
