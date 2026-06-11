#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=rlpd-setup
#SBATCH --output=setup_rlpd_%j.txt
# One-shot env setup on an RTX-4000 node (needs GPU for the JAX verify,
# internet confirmed working on r40a nodes). Keeps caches off the 30G
# home quota.
set -eo pipefail
export PIP_NO_CACHE_DIR=1
export CONDA_PKGS_DIRS=/tmp/zhao.b-conda-pkgs
mkdir -p "$CONDA_PKGS_DIRS"

cd "$HOME/rlpd_experiments"
bash setup_cluster.sh

# --- Hard verification beyond setup_cluster.sh's soft warnings ---
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rlpd
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"

python - <<'EOF'
import jax
devs = jax.devices()
print("JAX devices:", devs)
assert any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs), \
    "JAX sees no GPU — fleet would silently run on CPU"

import gym, d4rl
from rlpd.wrappers import wrap_gym
from rlpd.data.binary_datasets import BinaryDataset
for name in ["pen-binary-v0", "door-binary-v0"]:
    env = wrap_gym(gym.make(name), rescale_actions=True)
    ds = BinaryDataset(env, include_bc_data=True)
    n = ds.dataset_dict["observations"].shape
    print(name, "dataset OK, obs shape:", n)
EOF
echo "SETUP + HARD VERIFY OK"
