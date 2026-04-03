#!/bin/bash
# ================================================================
# setup_cluster.sh — Deploy RLPD experiments to a SLURM cluster.
#
# What it does:
#   1. Downloads MuJoCo 210 if missing
#   2. Copies the rlpd/ library from upstream if missing
#   3. Creates a conda env with pinned, tested dependencies
#   4. Installs Adroit binary envs (mjrl, mj_envs, datasets)
#   5. Verifies imports and runs a quick smoke test
#
# Usage (on a GPU node, not login node):
#   bash setup_cluster.sh
#
# If you cloned from GitHub, run this from the repo directory.
# If you copied files manually, set REPO_DIR to the target:
#   REPO_DIR=~/my_rlpd bash setup_cluster.sh
# ================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-rlpd}"
MUJOCO_DIR="$HOME/.mujoco"

echo "============================================"
echo "RLPD Cluster Setup"
echo "  Repo:   $REPO_DIR"
echo "  Env:    conda:$CONDA_ENV"
echo "============================================"
echo ""

# --- 1. Prerequisites ---
echo "[1/6] Prerequisites..."

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found." >&2
  echo "  Install miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
echo "  conda OK"

if ! command -v git &>/dev/null; then
  echo "ERROR: git not found" >&2
  exit 1
fi

if ! command -v gcc &>/dev/null; then
  echo "ERROR: gcc not found. mujoco-py needs a C compiler." >&2
  echo "  Try: module load gcc  (or ask your sysadmin)" >&2
  exit 1
fi
echo "  gcc OK"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "  GPU: $GPU_INFO"
else
  echo "  WARNING: No GPU detected. Run this on a GPU node for full verification."
fi

# --- 2. MuJoCo ---
echo ""
echo "[2/6] MuJoCo..."

if [ -d "$MUJOCO_DIR/mujoco210" ]; then
  echo "  mujoco210 found"
else
  echo "  Downloading mujoco210..."
  mkdir -p "$MUJOCO_DIR" && cd "$MUJOCO_DIR"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar xzf mujoco210-linux-x86_64.tar.gz
  rm -f mujoco210-linux-x86_64.tar.gz
  echo "  Installed to $MUJOCO_DIR/mujoco210"
fi

# Set LD_LIBRARY_PATH globally for the rest of setup (mujoco-py build needs this)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_DIR/mujoco210/bin"
# Add nvidia libs if they exist (needed by mujoco-py at import time)
[ -d /usr/lib/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
[ -d /usr/lib64/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/nvidia"

# --- 3. Clone upstream RLPD library ---
echo ""
echo "[3/6] RLPD library..."

if [ -d "$REPO_DIR/rlpd" ]; then
  echo "  rlpd/ already exists"
else
  echo "  Cloning ikostrikov/rlpd (library only)..."
  UPSTREAM=$(mktemp -d)
  git clone -q https://github.com/ikostrikov/rlpd.git "$UPSTREAM"
  mkdir -p "$REPO_DIR"
  cp -r "$UPSTREAM/rlpd" "$REPO_DIR/rlpd"
  rm -rf "$UPSTREAM"
  echo "  Copied rlpd/ to $REPO_DIR"
fi

# --- 4. Conda environment + dependencies ---
echo ""
echo "[4/6] Conda environment..."

if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
  echo "  Env '$CONDA_ENV' exists, activating..."
else
  echo "  Creating conda env (python 3.10)..."
  conda create -n "$CONDA_ENV" python=3.10 -y -q
fi

# Activate (works regardless of conda install location)
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Build tools needed by mujoco-py
echo "  Installing build dependencies (patchelf, glew, mesalib)..."
conda install -y -q patchelf glew mesalib 2>/dev/null || \
  conda install -y -q -c conda-forge patchelf glew mesalib 2>/dev/null || \
  echo "  WARNING: conda install of build deps failed — mujoco-py may not compile"

# Install in dependency order. mujoco-py compiles C extensions at install
# and again at first import, so numpy + Cython must be present first.
echo "  Installing numpy + Cython..."
pip install -q numpy==1.26.4 "Cython<3"

echo "  Installing mujoco-py (compiles C extensions)..."
pip install -q mujoco-py==2.1.2.14

# Trigger mujoco-py compilation now (it compiles on first import).
# Doing it here means errors surface during setup, not during training.
echo "  Compiling mujoco-py extensions..."
python -c "import mujoco_py" 2>/dev/null || {
  echo "  WARNING: mujoco-py first import failed (may compile at runtime instead)"
}

echo "  Installing JAX (CUDA 12)..."
pip install -q "jax[cuda12]==0.4.30" 2>/dev/null || {
  echo "  JAX CUDA 12 failed, trying CUDA 11..."
  pip install -q "jax[cuda11_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 2>/dev/null || {
    echo "  WARNING: GPU JAX install failed. Installing CPU JAX (training will be slow)."
    pip install -q "jax==0.4.30" "jaxlib==0.4.30"
  }
}

echo "  Installing remaining dependencies..."
pip install -q \
  flax==0.8.5 \
  optax==0.2.3 \
  tensorflow-probability==0.23.0 \
  gym==0.23.1 \
  "dm-control==1.0.20" \
  mujoco==2.3.7 \
  ml-collections==0.1.1 \
  absl-py==2.1.0 \
  scipy==1.13.1 \
  tqdm==4.66.4 \
  wandb==0.17.5 \
  imageio==2.34.2 \
  "moviepy==1.0.3" \
  gdown

echo "  Installing d4rl..."
pip install -q "d4rl @ git+https://github.com/Farama-Foundation/d4rl@master" 2>/dev/null || \
  echo "  WARNING: d4rl install had issues"

# --- 5. Adroit binary envs ---
echo ""
echo "[5/6] Adroit binary environments..."

if python -c "import mjrl" 2>/dev/null; then
  echo "  mjrl OK"
else
  echo "  Installing mjrl..."
  git clone -q https://github.com/aravindr93/mjrl "$HOME/mjrl" 2>/dev/null || true
  pip install -q -e "$HOME/mjrl"
fi

if python -c "import mj_envs" 2>/dev/null; then
  echo "  mj_envs OK"
else
  echo "  Installing mj_envs..."
  if [ ! -d "$HOME/mj_envs" ]; then
    git clone -q --recursive https://github.com/philipjball/mj_envs.git "$HOME/mj_envs"
    cd "$HOME/mj_envs" && git submodule update --remote 2>/dev/null || true
  fi
  # --no-deps: mj_envs pins an old mujoco-py that wants mujoco200
  pip install -q -e "$HOME/mj_envs" --no-deps
fi

if [ -d "$HOME/.datasets/awac-data" ] && ls "$HOME/.datasets/awac-data/"*.npy &>/dev/null; then
  echo "  Adroit datasets OK"
else
  echo "  Downloading Adroit datasets..."
  mkdir -p "$HOME/.datasets"
  cd "$HOME/.datasets"
  gdown "https://drive.google.com/uc?id=1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y" \
    -O awac_dext.zip --fuzzy -q 2>/dev/null || {
    echo "  WARNING: gdown failed. Download manually from:"
    echo "    https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y"
    echo "  Unzip into ~/.datasets/awac-data/"
  }
  if [ -f awac_dext.zip ]; then
    unzip -qo awac_dext.zip -d awac-data/
    rm -f awac_dext.zip
    echo "  Datasets installed"
  fi
fi

# --- 6. Verify ---
echo ""
echo "[6/6] Verifying..."

ERRORS=0
cd "$REPO_DIR"

python -c "import jax; print('  JAX', jax.__version__, '| devices:', jax.devices())" || {
  echo "  ERROR: JAX import failed"; ERRORS=$((ERRORS+1)); }

python -c "import mujoco_py; print('  mujoco_py OK')" || {
  echo "  ERROR: mujoco_py failed"; ERRORS=$((ERRORS+1)); }

python -c "
from rlpd.networks import Ensemble, MLP, StateActionValue, subsample_ensemble
from sac_learner_v2 import SACLearnerV2
print('  SACLearnerV2 OK')
" || { echo "  ERROR: SACLearnerV2 import failed"; ERRORS=$((ERRORS+1)); }

python -c "
import gym; import d4rl
from rlpd.data.binary_datasets import BinaryDataset
env = gym.make('pen-binary-v0')
obs = env.reset()
print('  pen-binary-v0 OK (obs shape:', obs.shape, ')')
" || { echo "  WARNING: pen-binary-v0 env test failed"; }

echo ""
echo "============================================"
if [ "$ERRORS" -eq 0 ]; then
  echo "SETUP COMPLETE"
else
  echo "SETUP COMPLETE WITH $ERRORS ERROR(S)"
fi
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  bash submit_all.sh --dry-run"
echo "  bash submit_all.sh"
echo "============================================"
