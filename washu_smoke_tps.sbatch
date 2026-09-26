#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --job-name=tps-smoke
#SBATCH --output=smoke_tps_%j.txt
# 20k-step smoke of the TPS code path (nq=10, sigma=0.2) — gates the TPS
# array. Verifies the subclass trains, the tps run-name tag, probe
# columns, and that timing still fits the 8h cap.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source "${RLPD_CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate rlpd
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib64/nvidia"

rm -rf results_smoke_tps
python train_abc_tps.py \
  --env_name=pen-binary-v0 \
  --max_steps=20000 \
  --config=configs/rlpd_config.py \
  --config.backup_entropy=False \
  --config.hidden_dims="(256, 256, 256)" \
  --config.num_min_qs=2 \
  --config.num_qs=10 \
  --config.critic_layer_norm=True \
  --target_smoothing_sigma=0.2 \
  --seed=0 \
  --tqdm=False \
  --results_dir=results_smoke_tps

python - <<'EOF'
import csv, sys, math
path = ("results_smoke_tps/"
        "pen-binary-v0_nq10_mq2_nodrop_tps0.2_s0/online_log.csv")
rows = list(csv.DictReader(open(path)))  # path itself checks the tps tag
for c in ["roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag"]:
    assert c in rows[0], f"missing column {c}"
    assert math.isfinite(float(rows[0][c])), f"nonfinite step-0 probe: {c}"
print("TPS RUN-NAME + PROBE COLUMNS OK")

t = {int(r["step"]): float(r["elapsed_min"]) for r in rows}
rate = t[20000] - t[10000]
proj = t[20000] + 98.0 * rate
print(f"TIMING: 10k->20k = {rate:.2f} min; projected 1M = {proj:.0f} min")
if proj > 430:
    print("GATE: FAIL — projected over 430 min")
    sys.exit(1)
print("GATE: PASS")
EOF
echo "TPS SMOKE OK"
