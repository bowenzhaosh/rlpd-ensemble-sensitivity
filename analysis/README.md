# analysis/ — fleet → paper pipeline

One command after the fleet lands (~06-15): `bash analysis/run_all.sh`
(sync from washu → tidy CSVs → all paper figures → all paper tables → compile
`paper/build/main.pdf`). Idempotent; safe on partial data — missing configs
render as red `[pending]` markers and a provisional banner in the PDF, so the
draft can never silently present incomplete numbers as final.

## Stages
| Script | In | Out |
|---|---|---|
| `collect_results.sh` | washu:`~/rlpd_experiments/results/` | `data/washu-202606/results/` (logs+summaries only) |
| `build_tidy.py` | both data eras | `out/tidy/{runs,timeseries}.csv`, `progress.json` |
| `make_figures.py` | tidy CSVs | `paper/figures/fig_*.pdf` (7 figures) |
| `make_tables.py` | tidy CSVs | `paper/tables/*.tex` incl. `numbers.tex` inline macros |

## Pre-registered analysis discipline
Locked 2026-06-11 at fleet 12/62, TPS 0/24 — i.e. **before** the data existed.
Full text in `rlpd_common.py` docstring. Summary:
1. Median + min–max bands across seeds; never mean±SEM at n≤5.
2. Cross-config consistency via exact binomial sign tests (one unit per config).
3. Same-seed pairing allowed for dropout contrasts; **never** for TPS arms
   (extra RNG split ⇒ unpaired trajectories) — TPS gets distribution stats
   (median/min–max + Mann-Whitney U) only.
4. No single-seed claims; n=1 cells are typographically flagged.
5. Sharpness ≡ roughness / |Q̄|² (Q-scale normalization); probe rows with
   |Q̄|<1 masked (step 0 only in practice).
6. σ=0.05 headline; σ∈{0.01,0.1} robustness sweep must preserve orderings.
7. Prospective test: Spearman ρ(sharp@t, final score), pen primary,
   leave-one-config-out range; with/without divergent M=1 configs.
8. Final score = `summary.json` `final_score` (mean of last 10 evals, fixed in
   the harness pre-launch). Final sharpness = median of probes in last 200k.

April-era data (`data/april/`) is corroboration only — see `data/april/README.md`.
