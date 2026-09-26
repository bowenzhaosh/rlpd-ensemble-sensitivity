# Results and scope

![Final score by ensemble size and dropout on pen and door](../paper/figures/fig_headline.png)

Final score is the fraction of the evaluation horizon spent in success, averaged
over the last ten evaluations of a 1M-step run. Entries below are seed medians
(min–max; number of seeds); target subset size *M* = 2 unless specified.

| Task | Ensemble *N* | No dropout | Dropout *p* = 0.01 | Difference of medians |
|---|---:|---|---|---:|
| pen | 2 | 0.53 (0.50–0.54; n=5) | 0.73 (0.70–0.75; n=5) | +0.20 |
| pen | 4 | 0.68 (0.65–0.69; n=3) | 0.78 (0.75–0.79; n=3) | +0.11 |
| pen | 6 | 0.77 (0.76–0.77; n=3) | 0.77 (0.76–0.78; n=3) | +0.01 |
| pen | 10 | 0.79 (0.76–0.81; n=5) | 0.79 (0.76–0.80; n=5) | −0.01 |
| pen (*M* = 1) | 2 | 0.00 (0.00–0.01; n=3) | 0.60 (0.60–0.62; n=3) | +0.60 |
| door | 2 | 0.56 (0.48–0.79; n=3) | 0.72 (0.71–0.81; n=3) | +0.16 |
| door | 4 | 0.83 (0.30–0.83; n=3) | 0.82 (0.80–0.82; n=3) | −0.01 |
| door | 6 | 0.77 (0.76–0.84; n=3) | 0.77 (0.73–0.79; n=3) | +0.01 |
| door | 10 | 0.82 (0.79–0.83; n=3) | 0.80 (0.75–0.81; n=3) | −0.02 |

The reported dropout benefit is largest at small ensembles. The target-policy
smoothing intervention is inconclusive, and the sharpness correlations do not
establish a causal mechanism or a validated model-selection rule. The additional
single-seed on-policy probe is excluded from the manuscript's headline claims;
its [analysis record](../data/onpolicy/README.md) explains the confounds.

The analysis plan was recorded when 12 of 62 grid runs were complete and before
TPS runs began. Subsequent changes to the statistical analysis are disclosed in
[analysis/README.md](../analysis/README.md).
