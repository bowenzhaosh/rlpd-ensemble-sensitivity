# April 2026 archive (Delta A100 era) — corroboration only

Moved here 2026-06-11 from `~/Downloads/results_all/`.

## Contents
- `run_tracker.csv` — **the source of truth** for April runs: env, nq, mq, dropout,
  seed, status, Slurm job id, final/peak score. Curated from job logs at the time.
  51 runs done (pen 29 standard + 6 diag; door 13 standard + 3 diag; 1 crash).
- `run_status.md` — human-readable April status snapshot (same content, with notes).
- `results/` — 25 surviving result dirs (`online_log.csv`, `summary.json`, some
  `diagnostic.csv`).
- `slurm_logs/` — 37 raw Slurm stdout logs (`rlpd_<jobid>.txt`).

## ⚠ Two reasons this era is corroboration-only
1. **Result-dir name collisions.** April run names were
   `<env>_nq<N>_lnTrue_<tag>_s<seed>` — no `mq` or `dropout` in the name, so
   configs differing only in those overwrote each other's dirs (fixed in commit
   `95fac9d`, after these runs). A dir here cannot be attributed to a unique
   config; **only `run_tracker.csv` rows (keyed by job id) are attributable.**
   Do not parse `results/` for paper numbers.
2. **Pre-probe harness.** No roughness/sharpness columns; scores only. Mixing
   eras inside one claim invites the mixed-harness reviewer attack — the June
   fleet re-runs every claim-bearing config on the patched harness.

## Allowed uses
- Cross-era replication check (e.g. seed-0 pen scores: June fleet replicated
  April within noise — see memory note / paper appendix).
- Appendix "history" only, clearly labeled single-seed April.

Score scale: `final_score` = mean of last 10 evals of `normalized_score`
= per-episode return × 100. Floors: pen −10000 (H=100), door −20000 (H=200).
Fraction-of-horizon-in-success = 1 + score/(100·H).
