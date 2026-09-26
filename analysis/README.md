# Analysis pipeline

From the repository root, activate the environment described in the
[top-level README](../README.md#reproduce-the-paper), then run:

```bash
bash analysis/run_all.sh --no-pdf  # tables and figures
bash analysis/run_all.sh           # also compile paper/build/main.pdf
```

Both commands use the shipped data by default. `--local` is accepted for backward
compatibility. The pipeline reads local archived inputs only. `RLPD_PY`
selects the Python executable without a silent fallback to another environment.
Missing LaTeX is an error unless `--no-pdf` is explicit.

## Stages

| Script | Input | Output |
|---|---|---|
| `validate_data.py` | Manifest and raw June logs | Fails if membership, steps, probes, metadata, or scores disagree |
| `build_tidy.py` | June logs and April tracker | `out/tidy/{runs,timeseries}.csv`, `progress.json` |
| `make_figures.py` | Tidy CSVs | Seven figure PDFs, README PNG, `out/tidy/prospective.csv` |
| `make_tables.py` | Tidy CSVs | LaTeX tables and `paper/tables/numbers.tex` result macros |
| `onpolicy_analysis.py` | `data/onpolicy/results/` | Supplementary analysis printed to the terminal |

The release build requires the complete manifest and scheduled diagnostics.
Validation happens before generated artifacts are replaced. Individual plotting
and table functions retain some provisional-data handling for development, but
partial-data builds are not a supported publication workflow.

To also verify the fixed input hashes and the supplementary probe archive:

```bash
python analysis/validate_data.py --checksums --onpolicy
python analysis/onpolicy_analysis.py
```

The tested stack is Python 3.11.7 with
[`requirements/analysis.txt`](../requirements/analysis.txt).
Generated numerical artifacts and figure PDFs matched the committed versions in
that environment. Figure timestamps are suppressed, but rendering dependencies
and platforms can still affect PDF bytes.

## Recorded analysis plan and amendments

The plan in the `rlpd_common.py` docstring was recorded on 2026-06-11, when 12/62
grid runs were complete and before TPS runs began. It calls for seed medians and
min–max bands, same-seed dropout contrasts, unpaired TPS comparisons, Q-scale
normalization, a headline probe scale of 0.05 with 0.01/0.1 sensitivity checks,
and prospective sharpness-score associations.

After the fleet completed, the 2026-06-13 analysis added config-level Spearman
correlations with a permutation null and leave-one-config-out ranges. The
pooled-over-seeds correlations remain available and are labeled anticonservative.
A post-hoc dropout × ensemble-size interaction Mann–Whitney test is also computed;
the paper uses the descriptive separation statement for that contrast.

## Definitions and aggregation

- Final score is the last ten evaluations' mean, taken from `summary.json` and
  independently checked against `online_log.csv`.
- Normalized sharpness is roughness divided by `q_abs_mean_diag` squared. Rows
  with Q scale below 1 are masked.
- Final sharpness is each run's median over **four probes**, at 850k, 900k, 950k,
  and 1M steps (`step > 800000`). This preserves the released implementation.
- Headline dropout deltas are differences of arm medians. Sign-table config
  directions use the median of **same-seed paired differences**, which can have
  a different sign. Seed-level p-values and config-direction counts have
  different denominators; the table labels them separately.
- The TPS raw-roughness manipulation percentage pools last-window probe rows
  before taking a median. It is a descriptive manipulation check.
- TPS arms are never seed-paired with baselines, because their random-key
  consumption differs. April results are kept separate as corroboration.

Tests cover corrupted/missing inputs, duplicate/truncated evaluations, missing
probes, summary disagreement, unknown runs, constant-input correlations, and
incomplete dropout pairs. The underlying training reproducibility limits are
listed in [docs/training.md](../docs/training.md#reproducibility-limits).
