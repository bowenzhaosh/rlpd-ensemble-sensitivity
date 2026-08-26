#!/bin/bash
#SBATCH --partition=gpu-linuxlab
#SBATCH --account=engr-acad-cse5100
#SBATCH --gres=gpu:rtx4000:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=rlpd-fixjax
#SBATCH --output=fixjax_%j.txt
# Repair the JAX stack: flax pulled unpinned orbax-checkpoint 0.11.x,
# which requires jax>=0.6 and upgraded jax/jaxlib to 0.6.2 while the CUDA
# plugin stayed at 0.4.30 -> GPU silently unusable. Downgrade jointly and
# hard-verify GPU compute + envs + datasets.
set -eo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rlpd
export PIP_NO_CACHE_DIR=1

pip install -q "jax[cuda12]==0.4.30" "orbax-checkpoint==0.5.23" "setuptools<81"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|jax-cuda|orbax|chex|numpy|setuptools)"

cd "$SLURM_SUBMIT_DIR"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia:/usr/lib64/nvidia"

python - <<'EOF'
import jax, jax.numpy as jnp
devs = jax.devices()
print("JAX devices:", devs)
assert any(d.platform == "gpu" for d in devs), "JAX sees no GPU"
x = jnp.ones((512, 512))
y = (x @ x).block_until_ready()
print("GPU matmul OK:", float(y[0, 0]))

import gym, d4rl
from rlpd.wrappers import wrap_gym
from rlpd.data.binary_datasets import BinaryDataset
from sac_learner_v2 import SACLearnerV2
print("SACLearnerV2 import OK")
for name in ["pen-binary-v0", "door-binary-v0"]:
    env = wrap_gym(gym.make(name), rescale_actions=True)
    ds = BinaryDataset(env, include_bc_data=True)
    print(name, "dataset OK, obs:", ds.dataset_dict["observations"].shape)
EOF
echo "FIX + HARD VERIFY OK"
