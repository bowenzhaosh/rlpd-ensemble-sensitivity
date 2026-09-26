# Manuscript

The source is `main.tex`, with references in `refs.bib` and generated results in
`figures/` and `tables/`.

## Build

From the repository root, after installing the analysis dependencies and LaTeX:

```bash
bash analysis/run_all.sh
# paper/build/main.pdf
```

To compile the already committed result artifacts without running analysis:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Result tables, figures, and designated inline result macros are generated from
archived evidence. Experimental settings and some historical/contextual prose
are written in the manuscript. The main pipeline validates complete evidence
before rebuilding; it does not certify every prose claim or citation.

## Manuscript status and formatting

The source targets the NeurIPS 2026 PTA workshop and retains an explicit
`Anonymous` author block. It currently uses `[dblblindworkshop,final]`, which
suppresses submission line numbering. This is the existing reading-copy format,
not a check that the manuscript meets a venue's submission requirements.
The repository, citation metadata, and license identify the authors, Bowen Zhao
and Zhuoyu Peng, Washington University in St. Louis. Do not treat this public
repository as an anonymous review package.

## Evidence scope

Headline results use the June multi-seed fleet. The April replication table uses
a curated tracker whose raw logs are unavailable. TPS contrasts are unpaired and
the intervention is reported as inconclusive. Config-level permutation
correlations are distinguished from pooled-over-seeds statistics.
The supplementary single-seed on-policy probe is not a headline claim.

See [analysis methods](../analysis/README.md) for exact aggregation definitions
and amendments to the recorded analysis plan, and [data provenance](../data/README.md)
for archive coverage. Check venue formatting and publication status separately
before distributing a submission or camera-ready version.
