# Training and cluster setup

Rebuilding the paper from archived logs only needs the
[analysis environment](../README.md#reproduce-the-analysis). Training additionally
requires Linux x86_64, a CUDA 12 GPU, conda with Python 3.10, a C compiler, MuJoCo
210, and the external RLPD, D4RL, mjrl, mj_envs, and Adroit binary datasets.

## Install on a GPU node

```bash
bash setup_cluster.sh
```

The script creates or activates conda environment `rlpd`, installs the versions
in `requirements.txt`, copies the upstream RLPD library into ignored `rlpd/`, and
installs the external environments and datasets. Subsequent package installs use
`requirements-training-constraints.txt` to prevent the declared stack from
drifting. Setup exits unsuccessfully if imports, GPU detection, dataset loading,
or dependency consistency checks fail. It does not silently fall back to CPU.

The setup still fetches external repositories from their current branches.
These scripts describe the historical environment, and a complete GPU installation
has not been revalidated as part of this repository cleanup. Save `python -m pip
freeze`, upstream commit hashes, driver versions, and the repository commit before
starting a new experiment campaign.

## Run one experiment

After setup, activate the environment and configure the simulator:

```bash
conda activate rlpd
export WANDB_MODE=disabled MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
# Add your site's NVIDIA library directory to LD_LIBRARY_PATH if needed.

python train_abc.py --env_name=pen-binary-v0 --seed=0 --max_steps=1000000 \
  --config=configs/rlpd_config.py --config.num_qs=2 --config.num_min_qs=2 \
  --config.critic_dropout_rate=0.01 --config.critic_layer_norm=True \
  --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
  --bootstrap_mask=False --independent_targets=False --critic_reset_step=0 \
  --results_dir=results
```

For target-policy smoothing, use `train_abc_tps.py` and add
`--target_smoothing_sigma=0.2`. For the supplementary probe, use `train_abc.py`
with `--onpolicy_probe=True --results_dir=results_op`.

Run names encode environment, ensemble size, target subset size, dropout, TPS
when applicable, and seed. They do **not** encode every flag. Use a new
`--results_dir` when changing steps, LayerNorm, actor delay, offline ratio, or
other settings. Reusing an output directory can overwrite earlier logs.

## Submit the archived manifests

The `washu_*.sbatch` scripts are site-specific examples. Set the partition,
account, GPU request, and wall time for your cluster. They default to
`$HOME/miniforge3`; set `RLPD_CONDA_ROOT` if conda is installed elsewhere.
Submit from the repository root, where the scripts and manifests reside.

```bash
mkdir -p logs
sbatch washu_setup_rlpd.sbatch
# Wait for setup to finish successfully before submitting the smoke job.
sbatch washu_smoke_rlpd.sbatch
# Inspect the successful setup and smoke logs before submitting arrays.
sbatch washu_array.sbatch       # 62 grid runs: washu_runs.txt
sbatch washu_array_tps.sbatch   # 24 TPS runs: washu_runs_tps.txt
sbatch washu_array_op.sbatch    # 9 supplementary runs: washu_runs_op.txt
```

Array scripts skip a run if its `summary.json` exists. This is a convenience for
retrying the original fleet, not a checkpoint-resume mechanism or a validation of
that file. Smoke scripts recreate their dedicated smoke-output directories.
The historical manifests and learner update rules are preserved.

To retrieve new logs, explicitly opt into synchronization:

```bash
RLPD_REMOTE=my-cluster RLPD_REMOTE_DIR=path/to/repo/results \
  bash analysis/run_all.sh --sync --no-pdf
```

The remote path is relative to the remote home unless absolute. Defaults are the
authors' `washu` SSH alias and `rlpd_experiments/results`. Sync can replace local
archived logs; use a separate checkout for new runs. Input checksum verification
will then differ from the fixed release archive, as expected.

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

The cleanup preserves the learner and historical random-number behavior. Changes
to seeding, training, or the statistical estimand should be separately versioned
and evaluated using newly identified runs.
