#!/bin/bash
# ================================================================
# setup_cluster.sh — Deploy RLPD experiments to a SLURM cluster.
#
# What it does:
#   1. Downloads MuJoCo 210 if missing
#   2. Copies the rlpd/ library from upstream if missing
#   3. Creates a conda env with pinned, tested dependencies
#   4. Installs Adroit binary envs (mjrl, mj_envs, datasets)
#   5. Verifies imports and compilation
#
# Usage (on a GPU node, not login node):
#   bash setup_cluster.sh
#
# If cloned from GitHub, run from the repo directory.
# ================================================================
set -eo pipefail

# The historical environment uses Linux x86_64, CUDA 12, and Python 3.10.
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "ERROR: training setup requires Linux x86_64; use analysis/requirements.txt for a local paper rebuild" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-rlpd}"
# Keep later pip installs from upgrading the historical JAX/NumPy stack.
export PIP_CONSTRAINT="$REPO_DIR/requirements-training-constraints.txt"
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
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
  echo "ERROR: git not found" >&2; exit 1
fi

if ! command -v gcc &>/dev/null; then
  echo "ERROR: gcc not found. mujoco-py needs a C compiler." >&2
  echo "  Try: module load gcc" >&2
  exit 1
fi
echo "  gcc OK"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "  GPU: $GPU_INFO"
else
  echo "  WARNING: No GPU detected. Run on a GPU node for full verification."
fi

# --- 2. MuJoCo ---
echo ""
echo "[2/6] MuJoCo..."

if [ -d "$MUJOCO_DIR/mujoco210" ]; then
  echo "  mujoco210 found"
else
  echo "  Downloading mujoco210..."
  mkdir -p "$MUJOCO_DIR"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    -O "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz" || {
    echo "ERROR: MuJoCo download failed. Check network." >&2; exit 1
  }
  tar xzf "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz" -C "$MUJOCO_DIR"
  rm -f "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz"
  echo "  Installed to $MUJOCO_DIR/mujoco210"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_DIR/mujoco210/bin"
[ -d /usr/lib/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
[ -d /usr/lib64/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/nvidia"

# --- 3. RLPD library ---
echo ""
echo "[3/6] RLPD library..."

if [ -d "$REPO_DIR/rlpd" ]; then
  echo "  rlpd/ already exists"
else
  echo "  Cloning ikostrikov/rlpd (library only)..."
  UPSTREAM=$(mktemp -d)
  git clone -q https://github.com/ikostrikov/rlpd.git "$UPSTREAM" || {
    rm -rf "$UPSTREAM"
    echo "ERROR: git clone failed. Check network." >&2; exit 1
  }
  mkdir -p "$REPO_DIR"
  cp -r "$UPSTREAM/rlpd" "$REPO_DIR/rlpd"
  rm -rf "$UPSTREAM"
  echo "  Copied rlpd/ to $REPO_DIR"
fi

# --- 4. Conda environment + dependencies ---
echo ""
echo "[4/6] Conda environment..."

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list 2>/dev/null | grep -qw "^${CONDA_ENV}"; then
  echo "  Env '$CONDA_ENV' exists, activating..."
  conda activate "$CONDA_ENV"
  # Verify Python version
  PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  case "$PY_VER" in
    3.10) echo "  Python $PY_VER OK" ;;
    *)
      echo "ERROR: Python $PY_VER detected; select a Python 3.10 conda environment with CONDA_ENV." >&2
      exit 1
      ;;
  esac
else
  echo "  Creating conda env (python 3.10)..."
  conda create -n "$CONDA_ENV" python=3.10 -y -q
  conda activate "$CONDA_ENV"
fi

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
  echo "ERROR: conda activate failed" >&2; exit 1
fi

echo "  Installing build dependencies..."
conda install -y -q patchelf glew mesalib 2>/dev/null || \
  conda install -y -q -c conda-forge patchelf glew mesalib 2>/dev/null || \
  echo "  WARNING: conda build deps install failed — mujoco-py may not compile"

echo "  Installing numpy + Cython..."
pip install -q numpy==1.26.4 "Cython<3" || { echo "ERROR: numpy/Cython install failed" >&2; exit 1; }

echo "  Installing mujoco-py..."
pip install -q mujoco-py==2.1.2.14 || { echo "ERROR: mujoco-py install failed" >&2; exit 1; }

echo "  Compiling mujoco-py extensions (first import)..."
python -c "import mujoco_py" || {
  echo "ERROR: mujoco-py compilation failed." >&2
  echo "  Check: LD_LIBRARY_PATH includes mujoco210/bin and nvidia libs" >&2
  echo "  Check: patchelf and glew are installed (conda install patchelf glew mesalib)" >&2
  exit 1
}

echo "  Installing the pinned training requirements (CUDA 12)..."
python -m pip install -r "$REPO_DIR/requirements.txt" || {
  echo "ERROR: training dependencies failed to resolve/install; see the pip output above" >&2
  exit 1
}

echo "  Installing d4rl..."
python -m pip install "d4rl @ git+https://github.com/Farama-Foundation/d4rl@master"

# --- 5. Adroit binary envs ---
echo ""
echo "[5/6] Adroit binary environments..."

if python -c "import mjrl" 2>/dev/null; then
  echo "  mjrl OK"
else
  echo "  Installing mjrl..."
  [ ! -d "$HOME/mjrl" ] && git clone -q https://github.com/aravindr93/mjrl "$HOME/mjrl"
  pip install -q -e "$HOME/mjrl"
fi

if python -c "import mj_envs" 2>/dev/null; then
  echo "  mj_envs OK"
else
  echo "  Installing mj_envs..."
  if [ ! -d "$HOME/mj_envs" ]; then
    git clone -q --recursive https://github.com/philipjball/mj_envs.git "$HOME/mj_envs"
    (cd "$HOME/mj_envs" && git submodule update --init --recursive)
  fi
  pip install -q -e "$HOME/mj_envs" --no-deps
fi

if [ -d "$HOME/.datasets/awac-data" ] && ls "$HOME/.datasets/awac-data/"*.npy &>/dev/null; then
  echo "  Adroit datasets OK"
else
  echo "  Downloading Adroit datasets..."
  mkdir -p "$HOME/.datasets"
  gdown "https://drive.google.com/uc?id=1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y" \
    -O "$HOME/.datasets/awac_dext.zip" --fuzzy -q 2>/dev/null
  if [ -f "$HOME/.datasets/awac_dext.zip" ]; then
    unzip -qo "$HOME/.datasets/awac_dext.zip" -d "$HOME/.datasets/awac-data/"
    rm -f "$HOME/.datasets/awac_dext.zip"
    echo "  Datasets installed"
  else
    echo "  WARNING: Dataset download failed. Download manually from:"
    echo "    https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y"
    echo "  Unzip into ~/.datasets/awac-data/"
  fi
fi

# --- 6. Verify ---
echo ""
echo "[6/6] Verifying..."

ERRORS=0
cd "$REPO_DIR"

python -c "import jax; print('  JAX', jax.__version__, '| devices:', jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'JAX sees no GPU'" || {
  echo "  ERROR: JAX import failed"; ERRORS=$((ERRORS+1)); }

python -c "import mujoco_py; print('  mujoco_py OK')" || {
  echo "  ERROR: mujoco_py failed"; ERRORS=$((ERRORS+1)); }

python -c "
from rlpd.networks import Ensemble, MLP, StateActionValue, subsample_ensemble
from sac_learner_v2 import SACLearnerV2
print('  SACLearnerV2 OK')
" || { echo "  ERROR: SACLearnerV2 import failed"; ERRORS=$((ERRORS+1)); }

python - <<'VERIFY' || { echo "  ERROR: binary environment/dataset check failed"; ERRORS=$((ERRORS+1)); }
import gym
import d4rl
from rlpd.data.binary_datasets import BinaryDataset
from rlpd.wrappers import wrap_gym
for name in ("pen-binary-v0", "door-binary-v0"):
    env = wrap_gym(gym.make(name), rescale_actions=True)
    dataset = BinaryDataset(env, include_bc_data=True)
    assert dataset.dataset_dict["observations"].shape[0] > 0, f"Empty dataset: {name}"
    print(name, "environment and dataset OK")
    env.close()
VERIFY
python -m pip check || { echo "  ERROR: inconsistent Python dependencies"; ERRORS=$((ERRORS+1)); }

echo ""
echo "============================================"
if [ "$ERRORS" -eq 0 ]; then
  echo "SETUP COMPLETE"
else
  echo "SETUP FAILED WITH $ERRORS ERROR(S)" >&2
  exit 1
fi
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  sbatch washu_smoke_rlpd.sbatch     # timing gate"
echo "  sbatch washu_array.sbatch          # 62-run grid"
echo "  sbatch washu_array_tps.sbatch      # 24-run TPS arm"
echo "============================================"
