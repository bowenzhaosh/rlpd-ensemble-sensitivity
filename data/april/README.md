# April 2026 replication data

`run_tracker.csv` is the curated source for the April experiments. It contains
45 completed rows: 36 standard and nine diagnostic runs, with configurations,
seeds, scores, and original job identifiers retained for provenance. Raw logs are
not included. Planning rows remain in the tracker and are excluded from the
completed-run analysis.

The tracker's `maxsteps` field records a planning value of 2,000,000; completed
experiments used 1,000,000 steps according to the historical experiment records.
The tracker is preserved as received, including this discrepancy.

These results provide corroboration only. Earlier output names omitted target
subset size and dropout, allowing different configurations to reuse a directory.
The tracker was curated before those ambiguities were resolved. This harness also
predates the roughness probes used by the June study, so the analyses do not pool
the two eras.

Most configurations have one seed. Five configurations have three to five seeds.
The paper's replication table compares April seed 0 with the June seed ranges
using the stated tolerance. Its numerical source is this tracker alone.

`final_score` is the mean of the last ten evaluations of return × 100. Convert to
the paper's fraction-of-horizon score with `1 + final_score / (100 * H)`, where
H = 100 for pen and H = 200 for door.
