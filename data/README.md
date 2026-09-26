# data/ — experiment results, two eras

| Era | Dir | Harness | Role in paper |
|-----|-----|---------|---------------|
| April 2026 (Delta A100) | `april/` | pre-probe (`95fac9d`-era), scores only, mostly single seed | **Corroboration only** (cross-era replication check, App. D). Result-dir names collide (no `mq`/`drop` in the name), so `april/run_tracker.csv` is the only attributable source; see `april/README.md`. |
| June 2026 (WashU RTX 4000 + A6000 shards) | `washu-202606/` | fleet harness: sharpness probe at σ ∈ {0.01, 0.05, 0.1} + `q_abs_mean_diag` every 50k steps, in every run | **All headline numbers.** 62-run multi-seed grid (Slurm job 78870 on RTX 4000, tasks 39-62 re-laned as 79201 on A6000 shards) + 24-run TD3 target-policy-smoothing arm (79210 on shards, 79211 on RTX 4000). |
| June 2026 on-policy probe | `onpolicy-202606/` | fleet harness + `--onpolicy_probe` (sharpness at the actor's own actions) | **Not claimed.** 9 pen runs, seed 0 (job 79790); `VERDICT.md` records why the result is confounded with Q-overestimation. |

`washu-202606/results/` holds all 86 run directories (`online_log.csv` + `summary.json` each);
`washu-202606/smoke_rlpd_78869.txt` is the timing-gate smoke log. These logs are the
input to `analysis/run_all.sh --local`, which regenerates every table and figure in the
paper without cluster access. `analysis/collect_results.sh` is the idempotent rsync that
produced them from the cluster.

Raw April run dirs and Slurm logs are not tracked (see `.gitignore`). The dense-reward
halfcheetah runs mentioned in the paper's Limitation (ii) and the spectral-norm probes of
App. F were course-report runs whose logs are not shipped; no number in the paper comes
from them.

Score scale: `final_score` in `summary.json` = mean of the last 10 evaluations of
`normalized_score` = per-episode return × 100, in [−100·H, 0] with H = 100 (pen) and
200 (door). Fraction-of-horizon-in-success, the paper's score, = 1 + score/(100·H).

## Integrity and schema

`checksums.sha256` records the 194 released inputs: 95 log/summary pairs, three
run manifests, and the April tracker. Verify without modifying evidence:

```bash
python analysis/validate_data.py --checksums --onpolicy
```

The June main/TPS archive has 201 evaluation rows per run (0 to 1M in 5k-step
increments). Offline diagnostics appear every 50k steps, including step 0.
On-policy diagnostics begin at 50k. Empty diagnostic cells between probe steps
are expected; a missing scheduled probe is an error.

| Raw field | Meaning |
|---|---|
| `step` | Training loop index used for evaluation/checkpoint labels |
| `success_rate` | Historical field name for the evaluation return; not a probability |
| `normalized_score` | In these binary environments, evaluation return × 100 |
| `roughness`, `roughness_s001`, `roughness_s01` | Offline probe roughness at sigma 0.05, 0.01, and 0.1 |
| `q_abs_mean_diag` | Absolute mean-Q scale for sharpness normalization |
| `final_score` (summary) | Mean of the last ten logged `normalized_score` values |
| `peak_score`, `peak_step` (summary) | Maximum logged score and its first occurrence |

Run-directory names provide ensemble/subset sizes, dropout, TPS dose, and seed.
The historical summaries do not record all flags or dependency revisions; see
[training limits](../docs/training.md#reproducibility-limits). The supplementary
verdict and the smoke log are historical records, not generated result inputs.
