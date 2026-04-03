#!/bin/bash
# ================================================================
# setup_cluster.sh — Deploy RLPD experiments to a new SLURM cluster.
#
# What it does:
#   1. Clones upstream ikostrikov/rlpd
#   2. Creates a conda env with pinned dependencies
#   3. Installs MuJoCo, d4rl, and Adroit binary datasets
#   4. Overlays experiment files into the repo
#   5. Verifies everything imports correctly
#
# Usage:
#   bash ~/rlpd_experiments/setup_cluster.sh
#
# Run this on a GPU node (not login node) so JAX can verify CUDA.
# ================================================================
set -eo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rlpd_experiments}"
EXPERIMENT_SRC="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="rlpd"
MUJOCO_DIR="$HOME/.mujoco"

echo "============================================"
echo "RLPD Cluster Setup"
echo "  Repo:   $REPO_DIR"
echo "  Env:    conda:$CONDA_ENV"
echo "  Source: $EXPERIMENT_SRC"
echo "============================================"
echo ""

# --- 1. Prerequisites ---
echo "[1/7] Checking prerequisites..."

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install miniconda first." >&2
  exit 1
fi
echo "  conda OK"

if ! command -v git &>/dev/null; then
  echo "ERROR: git not found" >&2
  exit 1
fi

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "  GPU: $GPU_INFO"
else
  echo "  WARNING: nvidia-smi not found (OK on login node, but JAX won't verify CUDA)"
fi

# --- 2. MuJoCo ---
echo ""
echo "[2/7] MuJoCo..."

if [ -d "$MUJOCO_DIR/mujoco210" ]; then
  echo "  mujoco210 found"
else
  echo "  Downloading mujoco210..."
  mkdir -p "$MUJOCO_DIR"
  cd "$MUJOCO_DIR"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar xzf mujoco210-linux-x86_64.tar.gz
  rm mujoco210-linux-x86_64.tar.gz
  echo "  Installed to $MUJOCO_DIR/mujoco210"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_DIR/mujoco210/bin:/usr/lib/nvidia"

# --- 3. Clone upstream RLPD ---
echo ""
echo "[3/7] Cloning RLPD..."

if [ -d "$REPO_DIR/rlpd" ]; then
  echo "  rlpd/ already exists in $REPO_DIR"
else
  UPSTREAM=$(mktemp -d)
  git clone -q https://github.com/ikostrikov/rlpd.git "$UPSTREAM"
  mkdir -p "$REPO_DIR"
  cp -r "$UPSTREAM/rlpd" "$REPO_DIR/rlpd"
  rm -rf "$UPSTREAM"
  echo "  Copied rlpd/ library to $REPO_DIR"
fi

# --- 4. Conda environment ---
echo ""
echo "[4/7] Conda environment..."

if conda env list | grep -q "^${CONDA_ENV} "; then
  echo "  Env '$CONDA_ENV' exists"
else
  echo "  Creating conda env (python 3.10)..."
  conda create -n "$CONDA_ENV" python=3.10 -y -q
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "  Installing patchelf..."
conda install patchelf -y -q 2>/dev/null || true

echo "  Installing numpy + Cython (must precede mujoco-py)..."
pip install -q numpy==1.26.4 "Cython<3"

echo "  Installing mujoco-py..."
pip install -q mujoco-py==2.1.2.14

echo "  Installing JAX (CUDA 12)..."
pip install -q "jax[cuda12]==0.4.30"

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

# --- 5. Adroit binary envs (mjrl + mj_envs + datasets) ---
echo ""
echo "[5/7] Adroit binary environment setup..."

if [ ! -d "$HOME/mjrl" ]; then
  git clone -q https://github.com/aravindr93/mjrl "$HOME/mjrl"
  pip install -q -e "$HOME/mjrl"
  echo "  Installed mjrl"
else
  echo "  mjrl exists"
fi

if [ ! -d "$HOME/mj_envs" ]; then
  git clone -q --recursive https://github.com/philipjball/mj_envs.git "$HOME/mj_envs"
  cd "$HOME/mj_envs" && git submodule update --remote 2>/dev/null
  pip install -q -e . --no-deps
  echo "  Installed mj_envs"
else
  echo "  mj_envs exists"
fi

if [ ! -d "$HOME/.datasets/awac-data" ] || [ -z "$(ls "$HOME/.datasets/awac-data/"*.npy 2>/dev/null)" ]; then
  echo "  Downloading Adroit datasets..."
  mkdir -p "$HOME/.datasets"
  cd "$HOME/.datasets"
  gdown "https://drive.google.com/uc?id=1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y" -O awac_dext.zip --fuzzy -q
  unzip -qo awac_dext.zip -d awac-data/
  rm -f awac_dext.zip
  echo "  Datasets installed to ~/.datasets/awac-data/"
else
  echo "  Adroit datasets exist"
fi

# --- 6. Deploy experiment files ---
echo ""
echo "[6/7] Deploying experiment files..."

OVERLAY_FILES=(
  sac_learner_v2.py
  diagnostic.py
  train_abc.py
  train_diagnostic.py
  run.sh
  submit_all.sh
  check_progress.sh
  experiments.txt
  requirements.txt
)

for f in "${OVERLAY_FILES[@]}"; do
  if [ -f "$EXPERIMENT_SRC/$f" ]; then
    cp "$EXPERIMENT_SRC/$f" "$REPO_DIR/$f"
    echo "  $f"
  fi
done

mkdir -p "$REPO_DIR/configs"
for f in "$EXPERIMENT_SRC/configs/"*.py; do
  [ -f "$f" ] || continue
  cp "$f" "$REPO_DIR/configs/$(basename "$f")"
  echo "  configs/$(basename "$f")"
done

mkdir -p "$REPO_DIR/results"

# --- 7. Verify ---
echo ""
echo "[7/7] Verifying..."

ERRORS=0
cd "$REPO_DIR"

python -c "import jax; print('  JAX', jax.__version__, '| devices:', jax.devices())" 2>/dev/null || {
  echo "  ERROR: JAX import failed"; ERRORS=$((ERRORS+1)); }

python -c "import flax; print('  Flax', flax.__version__)" 2>/dev/null || {
  echo "  ERROR: Flax import failed"; ERRORS=$((ERRORS+1)); }

python -c "import mujoco_py; print('  mujoco_py OK')" 2>/dev/null || {
  echo "  ERROR: mujoco_py import failed"; ERRORS=$((ERRORS+1)); }

python -c "
from rlpd.networks import Ensemble, MLP, StateActionValue, subsample_ensemble
from sac_learner_v2 import SACLearnerV2
print('  SACLearnerV2 OK')
" 2>/dev/null || {
  echo "  ERROR: SACLearnerV2 import failed"; ERRORS=$((ERRORS+1)); }

python -c "
import gym; import d4rl
from rlpd.data.binary_datasets import BinaryDataset
env = gym.make('pen-binary-v0')
print('  pen-binary-v0 OK')
" 2>/dev/null || {
  echo "  WARNING: pen-binary-v0 env failed"; }

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
