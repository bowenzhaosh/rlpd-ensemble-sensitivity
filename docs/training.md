# Training

Rebuilding the paper from archived logs only needs the
[analysis environment](../README.md#reproduce-the-paper). Training additionally
requires Linux x86_64, a CUDA 12 GPU, conda with Python 3.10, a C/C++ compiler,
MuJoCo 210, and the external RLPD, D4RL, mjrl, mj_envs, and Adroit binary datasets.

## Install dependencies

From the repository root, on a machine with an NVIDIA GPU:

```bash
bash scripts/setup_training.sh
conda activate rlpd
```

The script creates or activates conda environment `rlpd`, installs the versions
in `requirements/training.txt`, installs this repository as an editable package,
copies the upstream RLPD library into ignored `rlpd/`, and installs the external
environments and datasets. `CONDA_ENV` selects a different
environment name. Subsequent package installs use
`requirements/training-constraints.txt` to constrain the declared package stack.
Setup checks imports, GPU detection, datasets, and dependency consistency.

The installer fetches external repositories from their current branches. A
complete GPU installation has not been revalidated as part of this cleanup.
Save `python -m pip freeze`, upstream commit hashes, driver versions, and the
repository commit with any new runs.

## Run one experiment

Run commands from the repository root so the configs and upstream `rlpd/` library
are available. The three entrypoints are separate Python modules:

| Module | Experiment |
|---|---|
| `ensemble_sensitivity.training.grid` | Ensemble/dropout grid and optional on-policy probes |
| `ensemble_sensitivity.training.tps` | Target-policy smoothing |
| `ensemble_sensitivity.training.diagnostic` | Historical per-head diagnostics |

```bash
export WANDB_MODE=disabled MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
# Add your NVIDIA library directory to LD_LIBRARY_PATH if needed.

python -m ensemble_sensitivity.training.grid \
  --env_name=pen-binary-v0 --seed=0 --max_steps=1000000 \
  --config=configs/rlpd_config.py --config.num_qs=2 --config.num_min_qs=2 \
  --config.critic_dropout_rate=0.01 --config.critic_layer_norm=True \
  --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
  --bootstrap_mask=False --independent_targets=False --critic_reset_step=0 \
  --results_dir=results
```

For target-policy smoothing, use `ensemble_sensitivity.training.tps` and add
`--target_smoothing_sigma=0.2`. For the supplementary probe, use the grid module
with `--onpolicy_probe=True --results_dir=results_op`.

The [experiment manifests](../experiments/README.md) list the full grid, TPS, and
on-policy configurations. Apply each row's values to the command above. The
training entrypoints work directly with Python and have no scheduler dependency.
The diagnostic module provides the historical per-head diagnostics; it is not
required for the main grid or archived-data rebuild.

Run names encode environment, ensemble size, target subset size, dropout, TPS
when applicable, and seed. They do **not** encode every flag. Use a new
`--results_dir` when changing steps, LayerNorm, actor delay, offline ratio, or
other settings. Reusing an output directory can overwrite earlier logs.

## Reproducibility limits

The released archive supports recomputing the reported tables, figures, and
result macros. Exact retraining has additional limitations:

- Original commit hashes for all external training dependencies and a complete
  environment lock were not recorded. The declared package versions do not
  reconstruct all historical dependencies.
- The historical harness does not seed every NumPy/dataset random generator.
  Equal run seeds do not guarantee identical trajectories or bitwise results.
  TPS also consumes additional random keys, so TPS contrasts are unpaired.
- Trained checkpoints and the April raw logs are not shipped. April replication
  uses `data/april/run_tracker.csv` and is corroborative only.

The public cleanup preserves the learner and historical random-number behavior.
Changes to seeding, training, or statistical estimands should be separately
versioned and evaluated using newly identified runs.

## Code organization

`src/ensemble_sensitivity/agents/` contains the ensemble SAC learner and its TPS
variant. `diagnostics.py` implements fixed-buffer Q probes, and `training/`
contains the three experiment entrypoints. These modules were relocated from
the original root scripts; learner computations, flags, and random-key use are
preserved. The package itself declares no training dependencies; install those
through the setup script above. Installing it alone does not provide MuJoCo or
the external datasets.
