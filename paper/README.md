# paper/ — "Beyond Pessimism and Diversity" (NeurIPS 2026 workshop submission)

Build: `cd paper && latexmk -pdf -outdir=build main.tex`
(or `bash analysis/run_all.sh --local` from the repo root to regenerate figures and
tables first). `neurips_2026.sty` is the official NeurIPS 2026 style; `main.tex` uses
its `dblblindworkshop` option and `\author{Anonymous}` for review. Current build:
11 pages (≈6.5 pages of main text, then references and appendices), 0 errors.

**Nothing numeric is typed by hand.** Every number, figure, and table comes from
`analysis/make_{figures,tables}.py` over the tidy CSVs; `tables/numbers.tex` holds the
58 inline macros. Missing fleet results would render as red `[pending]` plus a
provisional banner, so the draft cannot overstate what the data contains; with the
complete 86-run fleet the banner is off.

## Claim discipline
- Headline claims = June fleet only (multi-seed, probe-instrumented harness).
- April-era numbers appear only in the App. D replication table.
- TPS arms: distribution stats only, never seed-paired (pre-registered).
- The TPS prediction was fixed before the runs (dose-dependent gain at N = 2, none at
  N = 10). The data did not bear it out and the manipulation moved the mediator only
  ~5%, so the paper reports the arm as inconclusive, neither confirmation nor
  refutation.
- Correlations are quoted at the config level with permutation p-values; the pooled
  per-run figures are printed and labelled anticonservative.

## Author list (confirmed 2026-08-24)
Bowen Zhao (corresponding, zhao.b@wustl.edu) and Zhuoyu Peng, Washington University in
St. Louis. Submissions are double-blind, so `main.tex` carries `\author{Anonymous}`; the
real list goes on the submission form and the commented-out camera-ready block is
restored on acceptance. The camera-ready NOTE in App. G is where the public repository
URL and commit hash go.
