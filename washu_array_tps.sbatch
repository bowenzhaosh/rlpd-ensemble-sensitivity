#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --job-name=rlpd-tps
#SBATCH --output=logs/tps_%A_%a.txt
#SBATCH --array=1-24
# ================================================================
# TPS (TD3 target-policy smoothing) arm: one run per task from
# washu_runs_tps.txt (7 fields: env,seed,nqs,minqs,dropout,tps,maxsteps).
# Submitted with a dependency so it starts only after the main seed
# fleet + its own smoke gate. Idempotent (skip if summary.json exists).
# ================================================================
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

LINE=$(grep -vE '^\s*(#|$)' washu_runs_tps.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
  echo "No run defined for array task $SLURM_ARRAY_TASK_ID"; exit 0
fi
IFS=',' read -r env seed nqs minqs dropout tps maxsteps <<< "$LINE"
[ -n "$maxsteps" ] || { echo "ERROR: maxsteps empty for task $SLURM_ARRAY_TASK_ID: $LINE"; exit 1; }
[ -n "$tps" ] || { echo "ERROR: tps empty for task $SLURM_ARRAY_TASK_ID: $LINE"; exit 1; }
if [ "$dropout" = "0.0" ]; then
  echo "ERROR: use '0' not '0.0' for dropout"; exit 1
fi

drop_tag="nodrop"
DROP_FLAG=""
if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
  drop_tag="drop${dropout}"
  DROP_FLAG="--config.critic_dropout_rate=$dropout"
fi
run_name="${env}_nq${nqs}_mq${minqs}_${drop_tag}_tps${tps}_s${seed}"

if [ -f "results/${run_name}/summary.json" ]; then
  echo "SKIP: results/${run_name}/summary.json already exists"; exit 0
fi

echo "=========================================="
echo "RUN: $run_name  steps=$maxsteps  node=$(hostname)  $(date)"
echo "=========================================="

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rlpd
[ "$CONDA_DEFAULT_ENV" = "rlpd" ] || { echo "ERROR: conda activate rlpd failed"; exit 1; }

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib64/nvidia"

python train_abc_tps.py \
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
  --target_smoothing_sigma="$tps" \
  --seed="$seed" \
  --tqdm=False \
  --results_dir=results

echo "FINISHED $run_name — $(date)"
