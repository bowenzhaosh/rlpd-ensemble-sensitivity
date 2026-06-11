#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --job-name=rlpd-smoke
#SBATCH --output=smoke_rlpd_%j.txt
# 20k-step timed smoke on pen-binary: verifies the full training path on
# RTX 4000 (env + dataset + JAX GPU + patched roughness probe) and
# measures throughput to project the 1M-step wall-clock vs the 8h cap.
set -eo pipefail
cd "$HOME/rlpd_experiments"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rlpd
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib64/nvidia"

rm -rf results_smoke
python train_abc.py \
  --env_name=pen-binary-v0 \
  --max_steps=20000 \
  --config=configs/rlpd_config.py \
  --config.backup_entropy=False \
  --config.hidden_dims="(256, 256, 256)" \
  --config.num_min_qs=2 \
  --config.num_qs=10 \
  --config.critic_layer_norm=True \
  --seed=0 \
  --tqdm=False \
  --results_dir=results_smoke

python - <<'EOF'
import csv, sys
path = "results_smoke/pen-binary-v0_nq10_mq2_nodrop_s0/online_log.csv"
rows = list(csv.DictReader(open(path)))
cols = rows[0].keys()
for c in ["roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag"]:
    assert c in cols, f"missing column {c}"
r0 = rows[0]
for c in ["roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag"]:
    assert r0[c] not in ("", None), f"step-0 probe empty for {c}"
    float(r0[c])
print("PROBE COLUMNS OK:", {c: r0[c] for c in
      ["roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag"]})

t = {int(r["step"]): float(r["elapsed_min"]) for r in rows}
rate_min_per_10k = t[20000] - t[10000]  # steady-state (training active >5k)
# Linear projection: elapsed@20k + 98 remaining 10k-blocks * rate.
# April reference, SAME config (nq=10): rate 2.2 min/10k, actual total 228 min
# (projection formula gives 221 on April data, ~3% under — acceptable).
proj_total_min = t[20000] + 98.0 * rate_min_per_10k
print(f"TIMING: 10k->20k = {rate_min_per_10k:.2f} min "
      f"(April A100 ref: 2.2); projected 1M nq=10 run = "
      f"{proj_total_min:.0f} min = {proj_total_min/60:.2f} h")
if proj_total_min > 430:
    print("GATE: FAIL — projected over 430 min; nq=10 runs do NOT safely "
          "fit the 8h walltime cap")
    sys.exit(1)
print("GATE: PASS — fits the 8h walltime cap with margin")
EOF
echo "SMOKE OK"
