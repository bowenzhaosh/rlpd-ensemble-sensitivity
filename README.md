# Beyond Pessimism and Diversity: What Critic Ensembles Regularize in Offline-to-Online RL

Code, experiment logs, and analysis pipeline for the paper

> **Beyond Pessimism and Diversity: What Critic Ensembles Regularize in Offline-to-Online RL**
> Bowen Zhao, Zhuoyu Peng (Washington University in St. Louis)
> Prepared as a NeurIPS 2026 workshop submission (double-blind; the manuscript does not link this repository until camera-ready).

RLPD ([Ball et al., 2023](https://arxiv.org/abs/2302.02948)) gets strong offline-to-online performance from a large *N*-head critic ensemble with min-over-*M* target subsetting. This paper asks *which property* of that ensemble tracks performance. It proposes that the operative quantity is the action-space smoothness of the **ensemble-mean** critic (the function the actor's gradient climbs), measured as a *Q*-scale-normalized sharpness, and it separates that quantity from the two usual accounts, pessimism (set by *M*) and head diversity. The paper proposes no new algorithm; it is a pre-registered, falsifiable analysis that reports its own inconclusive intervention.

Every number, table, and figure in the paper is generated from the run logs shipped in this repository by one script (`analysis/run_all.sh --local`). Nothing numeric is typed by hand.

## Headline results

Final score = fraction of the evaluation horizon spent in success (median over seeds, min-max range, *n* seeds), sparse-reward Adroit, 1M environment steps, *M* = 2 unless noted.

| env | *N* | no dropout | dropout *p* = 0.01 | Δ (median) |
|---|---|---|---|---|
| pen | 2 | 0.53 (0.50-0.54; n=5) | 0.73 (0.70-0.75; n=5) | +0.20 |
| pen | 4 | 0.68 (0.65-0.69; n=3) | 0.78 (0.75-0.79; n=3) | +0.11 |
| pen | 6 | 0.77 (0.76-0.77; n=3) | 0.77 (0.76-0.78; n=3) | +0.01 |
| pen | 10 | 0.79 (0.76-0.81; n=5) | 0.79 (0.76-0.80; n=5) | -0.01 |
| pen (*M* = 1) | 2 | 0.00 (0.00-0.01; n=3) | 0.60 (0.60-0.62; n=3) | +0.60 |
| door | 2 | 0.56 (0.48-0.79; n=3) | 0.72 (0.71-0.81; n=3) | +0.16 |
| door | 4 | 0.83 (0.30-0.83; n=3) | 0.82 (0.80-0.82; n=3) | -0.01 |
| door | 6 | 0.77 (0.76-0.84; n=3) | 0.77 (0.73-0.79; n=3) | +0.01 |
| door | 10 | 0.82 (0.79-0.83; n=3) | 0.80 (0.75-0.81; n=3) | -0.02 |

What the paper draws from the 86-run fleet (62 grid runs + 24 target-policy-smoothing runs):

1. **Interaction.** Dropout helps at small *N* and is inert at large *N*, decaying monotonically (pen Δ = +0.20, +0.11, +0.01, -0.01 at *N* = 2, 4, 6, 10). Normalized sharpness of the mean critic shows the same saturation. This reconciles DroQ's small-ensemble gains with RLPD's finding that dropout adds nothing at *N* = 10.
2. **Dissociation from pessimism.** Removing the min (*M* = 1) inflates |Q̄| on the probe set by ~1009× and collapses the run, while normalized sharpness does not rise (its *M* = 1 median is 0.1× the healthy median). Pessimism acts on amplitude; *N* and dropout act on geometry.
3. **Intervention, pre-registered and inconclusive.** TD3-style target-policy smoothing at σ ∈ {0.1, 0.2, 0.3} did not reproduce dropout's effect (Mann-Whitney *p* = 1.000 at both *N*), and it reduced probe roughness by only ~5%, so it is reported as inconclusive rather than as a refutation.
4. **Prospective signal, weak.** Sharpness at the 100k-step checkpoint is associated with final score among non-divergent pen configurations (config-level Spearman ρ = -0.74, permutation *p* = 0.045, *n* = 8 configs); pooling all configurations the association is not significant (ρ = -0.32, *p* = 0.37, *n* = 10). It is not presented as a selection rule.

The pre-registered analysis discipline (median/min-max, exact sign tests with one unit per config, no seed-pairing for the TPS arm, *Q*-scale normalization, σ = 0.05 headline with a {0.01, 0.1} robustness sweep) was written down on 2026-06-11 while the fleet was 12/62 complete; the full text is the module docstring of `analysis/rlpd_common.py`.

## Reproduce the paper from the shipped logs (no cluster needed)

Requires Python 3.11 with `pandas`, `scipy`, `matplotlib`, and a LaTeX installation with `latexmk`.

```bash
git clone https://github.com/bowenzhaosh/rlpd-ensemble-sensitivity.git
cd rlpd-ensemble-sensitivity
RLPD_PY=python3 bash analysis/run_all.sh --local
# -> analysis/out/tidy/{runs,timeseries,prospective}.csv
# -> paper/figures/fig_*.pdf (7 figures)
# -> paper/tables/*.tex incl. numbers.tex (every inline number in the paper)
# -> paper/build/main.pdf
```

`--local` skips the cluster rsync and rebuilds from `data/washu-202606/results/` (86 run directories, each with `online_log.csv` and `summary.json`). The generated tables and figures are committed, so `git diff --stat paper/tables paper/figures` after a rebuild should be empty.

## Re-run the experiments

The fleet ran on a Slurm cluster (WashU, one RTX 4000 Ada 20 GB per run, 3.5-5.7 wall-hours per 1M-step run). The harness is JAX on the reference RLPD codebase.

```bash
# on a GPU node: conda env `rlpd`, MuJoCo 210, d4rl, mjrl, mj_envs, Adroit binary datasets
bash setup_cluster.sh                 # or: sbatch washu_setup_rlpd.sbatch (setup + hard GPU/dataset verify)

sbatch washu_smoke_rlpd.sbatch        # 20k-step timing gate
sbatch washu_array.sbatch             # 62-run grid  (manifest: washu_runs.txt)
sbatch washu_array_tps.sbatch         # 24-run target-policy-smoothing arm (washu_runs_tps.txt)
sbatch washu_array_op.sbatch          # optional 9-run on-policy sharpness probe (washu_runs_op.txt)

bash analysis/collect_results.sh      # rsync logs back, then analysis/run_all.sh
```

The array scripts are idempotent (a task whose `summary.json` exists exits 0), so a whole array can be resubmitted to retry failures. Partition, account, and GPU lines at the top of each `.sbatch` are site-specific. Pinned dependency versions are in `requirements.txt`; `setup_cluster.sh` installs them in order (numpy/Cython, mujoco-py, JAX, then the rest) and clones the upstream `rlpd/` library, which is not vendored here.

A single run:

```bash
python train_abc.py --env_name=pen-binary-v0 --seed=0 --max_steps=1000000 \
  --config=configs/rlpd_config.py --config.num_qs=2 --config.num_min_qs=2 \
  --config.critic_dropout_rate=0.01 --config.critic_layer_norm=True \
  --config.backup_entropy=False --config.hidden_dims="(256, 256, 256)" \
  --bootstrap_mask=False --independent_targets=False --critic_reset_step=0 \
  --results_dir=results
```

Each run writes `results/<env>_nq<N>_mq<M>_<nodrop|drop<p>>_s<seed>/{online_log.csv,summary.json}`. `online_log.csv` carries the evaluation score every 5k steps and, every 50k steps, the sharpness probe at σ ∈ {0.01, 0.05, 0.1} plus `q_abs_mean_diag` (the |Q̄| scale used for normalization). `summary.json` holds `final_score` (mean of the last 10 evaluations, fixed before launch) and `peak_score`.

## Repository layout

```
sac_learner_v2.py        SAC agent with an N-head critic, min-over-M targets, per-head dropout masks
sac_learner_v2_tps.py    subclass adding TD3-style target-policy smoothing (the pre-registered intervention)
train_abc.py             training script for the grid (+ --onpolicy_probe for the on-policy sharpness probe)
train_abc_tps.py         training script for the TPS arm
diagnostic.py            sharpness/roughness probe (fixed 1000-pair probe set, sigma sweep, |Q| scale), diversity metrics
train_diagnostic.py      April-era training with per-head diversity diagnostics (head/mask variance, rank, OOD gap)
configs/                 ml_collections configs: rlpd_config.py (N=10, M=2, LayerNorm) on sac_config.py / td_config.py

washu_*.sbatch           Slurm scripts used for the June-2026 fleet: setup, smoke, grid array, TPS array, on-policy array
washu_runs*.txt          run manifests (env,seed,N,M,dropout,steps) for the three arrays
setup_cluster.sh         one-shot environment install (conda env, MuJoCo 210, d4rl, mjrl, mj_envs, datasets)
run.sh, submit_all.sh, experiments.txt, check_progress.sh
                         April-2026 launchers (Delta A100 era); kept for the replication appendix, superseded by washu_*

analysis/
  rlpd_common.py         paths, run-name parser, statistics; docstring = the pre-registered analysis discipline
  build_tidy.py          run dirs -> out/tidy/{runs,timeseries,prospective}.csv + progress.json
  make_figures.py        tidy CSVs -> paper/figures/fig_*.pdf
  make_tables.py         tidy CSVs -> paper/tables/*.tex (incl. numbers.tex inline macros)
  onpolicy_analysis.py   analysis of the on-policy probe runs (data/onpolicy-202606/)
  run_all.sh             the one button: [sync] -> tidy -> figures -> tables -> PDF
  collect_results.sh     rsync of run logs from the cluster
  out/tidy/              generated tidy CSVs (committed)

data/
  washu-202606/results/  the 86 run directories behind every headline number (online_log.csv + summary.json)
  onpolicy-202606/       9-run on-policy sharpness probe (seed 0) + VERDICT.md: confounded, not claimed in the paper
  april/                 April-2026 course-report era: run_tracker.csv is the only attributable source; corroboration only
  README.md              provenance of the two eras

paper/
  main.tex, refs.bib     the manuscript (NeurIPS 2026 style; \author{Anonymous} for review)
  figures/, tables/      generated by analysis/ (committed)
  README.md              build notes and claim discipline
```

## Data provenance

Two eras of runs exist and are never mixed inside a claim (details in `data/README.md`):

- **June 2026, WashU RTX 4000** (`data/washu-202606/`): all headline numbers. 62 grid runs (pen: *N* ∈ {2,4,6,10} × *p* ∈ {0, 0.01} with 5 seeds on the four headline cells and 3 elsewhere, plus *M* = 1 arms at *N* = 2; door: 8 configs × 3 seeds) and 24 TPS runs. One harness with the probe compiled in, so every run carries score and sharpness.
- **April 2026, Delta A100** (`data/april/`): the course-report era, single seed, probe-free harness, and run directories whose names did not encode *M* or dropout. Used only for the cross-era replication check in the appendix. Raw April run dirs and Slurm logs are not tracked; `run_tracker.csv` is the curated source.

The on-policy probe (`data/onpolicy-202606/`) measured sharpness at the actor's own actions after the fleet landed. It correlated with score better than the offline probe but was judged confounded with *Q*-overestimation magnitude and single-seed; `VERDICT.md` records the analysis and why it is not claimed.

## Spectral-normalization branch

The course report also tried spectral normalization of critic layers as a smoothness intervention. That line of code (`critic_spec_norm.py`, gradient-based sharpness diagnostics, a `spec_norm_coef` config knob, by Zhuoyu Peng) lives on the branch [`spectral-norm-probe`](https://github.com/bowenzhaosh/rlpd-ensemble-sensitivity/tree/spectral-norm-probe). It is not merged into `main` because it changes the run-directory naming that the analysis parser expects, and because the paper reports those probes as superseded single-seed work (App. F): they were confounded with LayerNorm, and target-policy smoothing replaced them as the controlled intervention.

## Citation

```bibtex
@misc{zhao2026beyondpessimism,
  title  = {Beyond Pessimism and Diversity: What Critic Ensembles Regularize in Offline-to-Online RL},
  author = {Zhao, Bowen and Peng, Zhuoyu},
  year   = {2026},
  note   = {NeurIPS 2026 workshop submission. Code: https://github.com/bowenzhaosh/rlpd-ensemble-sensitivity}
}
```

## License and acknowledgements

Experiment and analysis code in this repository is released under the MIT License (see `LICENSE`). The agent builds on the reference RLPD implementation, [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd) (MIT), which `setup_cluster.sh` clones into `rlpd/` at install time. Environments and datasets come from D4RL, mjrl, and mj_envs under their own licenses. Compute for the June 2026 fleet was provided by Washington University in St. Louis (Engineering IT GPU pool).
