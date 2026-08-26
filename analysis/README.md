# analysis/ — fleet logs → paper pipeline

One command rebuilds everything from the logs shipped in `data/`:

```bash
RLPD_PY=python3 bash analysis/run_all.sh --local
```

(tidy CSVs → all paper figures → all paper tables incl. inline-number macros →
`paper/build/main.pdf`). Without `--local` it first rsyncs from the cluster via
`collect_results.sh`. The pipeline is idempotent and safe on partial data: missing
configs render as red `[pending]` markers plus a provisional banner in the PDF, so a
draft can never silently present incomplete numbers as final. With the complete fleet
(`out/tidy/progress.json`: 62/62 + 24/24) the banner is off and `\prov{}` is a
pass-through.

## Stages
| Script | In | Out |
|---|---|---|
| `collect_results.sh` | cluster `~/rlpd_experiments/results/` | `data/washu-202606/results/` (logs + summaries only) |
| `build_tidy.py` | both data eras | `out/tidy/{runs,timeseries,prospective}.csv`, `progress.json` |
| `make_figures.py` | tidy CSVs | `paper/figures/fig_*.pdf` (7 figures) |
| `make_tables.py` | tidy CSVs | `paper/tables/*.tex` incl. `numbers.tex` inline macros |
| `onpolicy_analysis.py` | `data/onpolicy-202606/` | the numbers in `data/onpolicy-202606/VERDICT.md` |

Needs Python ≥ 3.10 with pandas, scipy, matplotlib; `RLPD_PY` selects the interpreter
(default: a pyenv path on the authors' machine).

## Pre-registered analysis discipline
Locked 2026-06-11 at fleet 12/62, TPS 0/24, i.e. before the data existed. Full text in
the `rlpd_common.py` docstring. Summary:
1. Median + min-max bands across seeds; never mean ± SEM at n ≤ 5.
2. Cross-config consistency via exact binomial sign tests (one unit per config).
3. Same-seed pairing allowed for dropout contrasts; **never** for TPS arms (extra RNG
   split ⇒ unpaired trajectories): TPS gets distribution stats (median/min-max +
   Mann-Whitney U) only.
4. No single-seed claims; n = 1 cells are typographically flagged.
5. Sharpness ≡ roughness / |Q̄|² (Q-scale normalization); probe rows with |Q̄| < 1 masked
   (step 0 only in practice).
6. σ = 0.05 headline; σ ∈ {0.01, 0.1} robustness sweep must preserve orderings.
7. Prospective test: Spearman ρ(sharpness@t, final score), pen primary,
   leave-one-config-out range; with/without divergent M = 1 configs.
8. Final score = `summary.json` `final_score` (mean of last 10 evals, fixed in the
   harness pre-launch). Final sharpness = median of probes in the last 200k steps.

Added after the fleet landed, in response to a results audit (2026-06-13): the
prospective and sharpness-score correlations are reported at the **config level** with a
permutation null and leave-one-config-out range (`spearman_config_perm`), because the
pre-registered pooled-over-seeds statistic double-counts correlated seeds; the pooled
figure is still printed and labelled anticonservative. The dropout × N interaction
Mann-Whitney test is post-hoc and is labelled as such in the paper.

April-era data (`data/april/`) is corroboration only; see `data/april/README.md`.
