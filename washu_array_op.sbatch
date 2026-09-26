#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --job-name=rlpd-op
#SBATCH --output=logs/rlpd_op_%A_%a.txt
#SBATCH --array=1-9
# ================================================================
# On-policy actor-side sharpness probe fleet (washu_runs_op.txt).
# Same harness as washu_array.sbatch + --onpolicy_probe=True; writes
# to results_op/ so it never collides with the main results/ fleet.
# Idempotent: a task whose results_op/<run>/summary.json exists exits 0.
# ================================================================
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

LINE=$(grep -vE '^\s*(#|$)' washu_runs_op.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
  echo "No run defined for array task $SLURM_ARRAY_TASK_ID"; exit 0
fi
IFS=',' read -r env seed nqs minqs dropout maxsteps <<< "$LINE"
[ -n "$maxsteps" ] || { echo "ERROR: maxsteps empty for task $SLURM_ARRAY_TASK_ID: $LINE"; exit 1; }
if [ "$dropout" = "0.0" ]; then
  echo "ERROR: use '0' not '0.0' for dropout"; exit 1
fi

drop_tag="nodrop"
DROP_FLAG=""
if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
  drop_tag="drop${dropout}"
  DROP_FLAG="--config.critic_dropout_rate=$dropout"
fi
run_name="${env}_nq${nqs}_mq${minqs}_${drop_tag}_s${seed}"

if [ -f "results_op/${run_name}/summary.json" ]; then
  echo "SKIP: results_op/${run_name}/summary.json already exists"; exit 0
fi

echo "=========================================="
echo "OP-RUN: $run_name  steps=$maxsteps  node=$(hostname)  $(date)"
echo "=========================================="

source "${RLPD_CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate rlpd
[ "$CONDA_DEFAULT_ENV" = "rlpd" ] || { echo "ERROR: conda activate rlpd failed"; exit 1; }

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib64/nvidia"

python train_abc.py \
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
  --tqdm=False \
  --onpolicy_probe=True \
  --results_dir=results_op

echo "FINISHED $run_name — $(date)"
