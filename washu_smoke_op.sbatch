#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=rlpd-op-smoke
#SBATCH --output=logs/rlpd_op_smoke_%j.txt
# Smoke: one 55k-step run (hits the step-50k probe) verifying the on-policy
# columns populate and the run does not crash. Writes to results_op_smoke/.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate rlpd
[ "$CONDA_DEFAULT_ENV" = "rlpd" ] || { echo "ERROR: conda activate rlpd failed"; exit 1; }
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib64/nvidia"
rm -rf results_op_smoke
python train_abc.py --env_name=pen-binary-v0 --max_steps=55000 \
  --config=configs/rlpd_config.py --config.backup_entropy=False \
  --config.hidden_dims="(256, 256, 256)" --config.num_min_qs=2 --config.num_qs=2 \
  --config.critic_layer_norm=True --bootstrap_mask=False --independent_targets=False \
  --critic_reset_step=0 --seed=0 --tqdm=False --onpolicy_probe=True --results_dir=results_op_smoke
LOG=results_op_smoke/pen-binary-v0_nq2_mq2_nodrop_s0/online_log.csv
echo "=== SMOKE HEADER ==="; head -1 "$LOG"
echo "=== SMOKE rows (50k probe should have roughness_onpolicy) ==="; tail -3 "$LOG"
echo "=== SMOKE SUMMARY ==="; cat results_op_smoke/pen-binary-v0_nq2_mq2_nodrop_s0/summary.json
echo "SMOKE_DONE"
