# April 2026 archive (Delta A100 era): corroboration only

Course-report era runs, moved here 2026-06-11 from the local results archive.

## What ships here
- `run_tracker.csv`: **the source of truth** for April runs (env, nq, mq, dropout,
  seed, status, Slurm job id, final/peak score), curated from the job logs at the
  time; 45 done rows (36 standard + 9 diagnostic). Its `maxsteps` column carries the
  planning value 2,000,000; every done run trained for 1,000,000 steps (the Slurm logs
  and `experiments.txt` record `steps=1000000`).
- `run_status.md`: a later human-readable snapshot. It lists five extra runs (pen
  M=1 at N=4/6/10; door (2,2,0) seeds 1-2) and one crash that are not in the tracker
  and are not used by any analysis.

The raw April result directories and Slurm logs are kept in the authors' archive and
are not tracked in this repository (`.gitignore`); nothing in the paper reads them.

## Why this era is corroboration only
1. **Result-dir name collisions.** April run names were
   `<env>_nq<N>_lnTrue_<tag>_s<seed>` with no `mq` or `dropout` in the name, so
   configs differing only in those overwrote each other's dirs (fixed in commit
   `95fac9d`, after these runs). Only `run_tracker.csv` rows (keyed by job id) are
   attributable.
2. **Pre-probe harness.** No roughness/sharpness columns; scores only. The June fleet
   re-runs every claim-bearing config on the patched harness so no claim mixes eras.

## Seeds
Mostly single-seed. Five cells carry 3-5 seeds (pen (2,2,0), (2,2,0.01), (10,2,0) with
5 seeds; door (2,2,0.01) and (10,2,0) with 3). The paper's replication table (App. D)
tabulates April seed 0 against the June min-max band +/-0.08; every additional April
seed also satisfies that criterion (checked 2026-08-25 from `analysis/out/tidy/runs.csv`).

Score scale: `final_score` = mean of last 10 evals of `normalized_score`
= per-episode return x 100. Floors: pen -10000 (H=100), door -20000 (H=200).
Fraction-of-horizon-in-success = 1 + score/(100*H).
