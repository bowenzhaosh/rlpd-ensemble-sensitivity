#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --job-name=rlpd
#SBATCH --output=logs/rlpd_%A_%a.txt
#SBATCH --array=1-62
# ================================================================
# WashU RTX-4000 fleet: one run per array task, read from
# washu_runs.txt (comments stripped). QoS academic-job caps 4
# running jobs/user; the rest queue automatically.
# Idempotent: a task whose summary.json already exists exits 0,
# so the whole array can be resubmitted to retry failures.
# ================================================================
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

LINE=$(grep -vE '^\s*(#|$)' washu_runs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
  echo "No run defined for array task $SLURM_ARRAY_TASK_ID"; exit 0
fi
IFS=',' read -r env seed nqs minqs dropout maxsteps <<< "$LINE"
[ -n "$maxsteps" ] || { echo "ERROR: maxsteps empty for task $SLURM_ARRAY_TASK_ID: $LINE"; exit 1; }
if [ "$dropout" = "0.0" ]; then
  echo "ERROR: use '0' not '0.0' for dropout (bash/python drop_tag would diverge)"; exit 1
fi

drop_tag="nodrop"
DROP_FLAG=""
if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
  drop_tag="drop${dropout}"
  DROP_FLAG="--config.critic_dropout_rate=$dropout"
fi
run_name="${env}_nq${nqs}_mq${minqs}_${drop_tag}_s${seed}"

if [ -f "results/${run_name}/summary.json" ]; then
  echo "SKIP: results/${run_name}/summary.json already exists"; exit 0
fi

echo "=========================================="
echo "RUN: $run_name  steps=$maxsteps  node=$(hostname)  $(date)"
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
  --results_dir=results

echo "FINISHED $run_name — $(date)"
