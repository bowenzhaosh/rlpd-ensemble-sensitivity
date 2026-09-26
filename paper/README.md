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

## Public manuscript

The source uses the NeurIPS style's `preprint` option and credits Bowen Zhao and
Zhuoyu Peng, Washington University in St. Louis. It links the public code and data
repository. This is a public reading copy; it does not assert conference
acceptance or satisfy a separate anonymous-submission requirement.

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
