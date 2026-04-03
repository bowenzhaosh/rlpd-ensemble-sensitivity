#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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
#
# Cluster-specific SLURM flags:
#   If your cluster uses a different GPU request syntax (e.g. -G 1
#   instead of --gres=gpu:1), edit line 2. If you need --partition
#   or --account, add them to the SBATCH header.
# ================================================================

# --- Activate conda env ---
# Find conda regardless of install location (miniconda3, anaconda3, miniforge3, etc.)
if [ -n "$CONDA_EXE" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/anaconda3"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniforge3"
else
  echo "ERROR: Cannot find conda installation" >&2; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rlpd

# --- Paths and env vars ---
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
[ -d /usr/lib/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
[ -d /usr/lib64/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/nvidia"

# --- Run ---
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
