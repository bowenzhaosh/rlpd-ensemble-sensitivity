# data/ — experiment results, two eras

| Era | Dir | Harness | Role in paper |
|-----|-----|---------|---------------|
| April 2026 (Delta A100) | `april/` | pre-roughness-probe (`95fac9d`-era), scores only | **Corroboration only.** Never headline. Single-harness-era runs; result dir names collide (no mq/drop in name) — see `april/README.md`. |
| June 2026 (WashU RTX-4000) | `washu-202606/` | patched harness: roughness σ∈{0.01,0.05,0.1} + `q_abs_mean_diag` every 50k, all runs | **All headline numbers.** 62-run multi-seed fleet (78870) + 24-run TD3-TPS arm (78877). |

Sync the washu era with `analysis/collect_results.sh` (idempotent rsync; safe to re-run
while the fleet is still going — partial data is handled downstream).

Raw run dirs and Slurm logs are gitignored (only curated CSVs + READMEs are tracked).
The single source of truth for April final scores is `april/run_tracker.csv`
(curated from Slurm logs), NOT the April result dirs.
