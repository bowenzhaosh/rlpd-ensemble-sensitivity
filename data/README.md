# Experiment data

The raw evaluation logs and summaries needed for the paper are included.

| Dataset | Location | Role |
|---|---|---|
| June 2026 main study | `june-2026/results/` | 62 grid runs and 24 target-policy-smoothing runs; headline results |
| Supplementary on-policy probe | `onpolicy/results/` | Nine single-seed pen runs; exploratory analysis only |
| April 2026 replication | `april/run_tracker.csv` | Curated scores from earlier experiments; corroboration only |

The June study used RTX 4000 Ada and A6000 GPUs at Washington University in
St. Louis. The April study used A100 GPUs. Each June run records evaluations and
sharpness probes in the same harness. April runs predate those probes and are
kept separate in every analysis.

The raw April logs, trained checkpoints, dense-reward halfcheetah runs, and
superseded spectral-normalization probes are not shipped. No quantitative result
is reconstructed from those unavailable files. See [April provenance](april/README.md)
and [on-policy limitations](onpolicy/README.md).

## Integrity

`checksums.sha256` covers 194 inputs: 95 log/summary pairs, three experiment
manifests, and the April tracker.

```bash
python analysis/validate_data.py --checksums --onpolicy
```

The public cleanup relocated the archived files and replaced deployment comments
in the manifests. Raw CSV/JSON bytes, the April tracker, and every manifest run
value and ordering were preserved. The checksum inventory reflects the new paths
and manifest comments.

## Schema

There are 201 evaluation rows per June run, from 0 to 1M in 5k-step increments.
Offline probes appear every 50k steps, including step 0. On-policy probes begin
at 50k. Empty diagnostic cells between probe steps are expected.

| Field | Meaning |
|---|---|
| `step` | Training loop index used for evaluation labels |
| `success_rate` | Historical field name for evaluation return; not a probability |
| `normalized_score` | In these binary environments, evaluation return × 100 |
| `roughness`, `roughness_s001`, `roughness_s01` | Offline probe roughness at sigma 0.05, 0.01, and 0.1 |
| `q_abs_mean_diag` | Absolute mean-Q scale for sharpness normalization |
| `final_score` (summary) | Mean of the last ten logged `normalized_score` values |
| `peak_score`, `peak_step` (summary) | Maximum logged score and its first occurrence |

The paper's score is `1 + final_score / (100 * H)`, with horizon H = 100 for pen
and H = 200 for door. This is the fraction of the evaluation horizon spent in
success. Run-directory names encode ensemble/subset sizes, dropout, TPS dose,
and seed. Historical summaries do not record every flag or dependency revision;
see [training limits](../docs/training.md#reproducibility-limits).
