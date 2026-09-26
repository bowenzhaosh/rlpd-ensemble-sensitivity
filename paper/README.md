# Manuscript

The manuscript source is [`main.tex`](main.tex), with references in
[`refs.bib`](refs.bib). Tables and figures are generated from the
released data by the [analysis pipeline](../analysis/README.md).

After installing the [analysis dependencies](../requirements/analysis.txt) and
LaTeX with `latexmk`, run from the repository root:

```bash
bash analysis/run_all.sh
```

The compiled manuscript is written to `paper/build/main.pdf`. The build directory
is ignored by Git. To compile the source using the committed tables and figures:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The source uses the NeurIPS `preprint` style with author names. It is a public
manuscript, with no claim of conference acceptance. Rebuilding verifies the
generated results; it does not certify every prose claim or citation.

See [results and scope](../docs/results.md), [analysis methods](../analysis/README.md),
and [data provenance](../data/README.md) for aggregation definitions, analysis
amendments, and archive coverage.
