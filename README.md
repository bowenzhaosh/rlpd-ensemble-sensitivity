# Beyond Pessimism and Diversity
### What Critic Ensembles Regularize in Offline-to-Online RL

**Bowen Zhao · Zhuoyu Peng**<br>
Washington University in St. Louis

[Manuscript source](paper/main.tex) · [Results](docs/results.md) ·
[Reproduce](#reproduce-the-paper) · [Train](docs/training.md) · [Data](data/README.md) ·
[Citation](#citation)

Research code and archived results for a study of critic ensemble size, dropout,
and target pessimism in [RLPD](https://github.com/ikostrikov/rlpd). The release
includes 86 main experiments on sparse Adroit pen and door tasks and nine
supplementary on-policy probe runs.

![Final score by ensemble size and dropout on pen and door](paper/figures/fig_headline.png)

Dropout's observed benefit is largest at small ensembles. Target-policy smoothing
is inconclusive, and the sharpness correlations do not establish a causal
mechanism. See [results and scope](docs/results.md) for seed counts, uncertainty,
and the supplementary probe's limitations.

## Reproduce the paper

The logs are included. Rebuilding the results requires **Python 3.11** and no GPU
or external datasets.

```bash
git clone https://github.com/bowenzhaosh/rlpd-ensemble-sensitivity.git
cd rlpd-ensemble-sensitivity
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/analysis.txt
python analysis/validate_data.py --onpolicy --checksums
bash analysis/run_all.sh --no-pdf
```

This regenerates the tables and figures from the archived inputs. To also build
the manuscript, install LaTeX with `latexmk` and run `bash analysis/run_all.sh`.
The PDF is written to `paper/build/main.pdf`.

[Analysis methods](analysis/README.md) describes the pipeline, recorded plan,
and statistical definitions. [Training](docs/training.md) covers the separate
Linux/CUDA environment, experiment commands, and limits of exact retraining.

## Repository structure

```text
src/ensemble_sensitivity/
  agents/       Ensemble SAC and target-policy smoothing
  training/     Grid, TPS, and per-head diagnostic entrypoints
  diagnostics.py
configs/        Agent configurations
experiments/    Manifests for all 95 archived runs
analysis/       Validation, statistics, tables, and figures
data/           Archived logs, provenance, and checksums
paper/          Manuscript, bibliography, and generated figures/tables
requirements/   Analysis, training, and development dependencies
scripts/        Training environment setup
docs/           Training guide and detailed results
tests/          Regression checks
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks. CI validates the
archive, rebuilds the numerical results, and compiles the manuscript.

## Citation

Software metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{zhao2026beyondpessimism,
  title  = {Beyond Pessimism and Diversity: What Critic Ensembles Regularize in Offline-to-Online RL},
  author = {Zhao, Bowen and Peng, Zhuoyu},
  year   = {2026},
  note   = {Manuscript and code},
  url    = {https://github.com/bowenzhaosh/rlpd-ensemble-sensitivity}
}
```

## License and acknowledgements

Released under the [MIT License](LICENSE). The learner, configurations, and
training code build on [RLPD](https://github.com/ikostrikov/rlpd), with upstream
notices in [LICENSE-rlpd](LICENSE-rlpd). The conference style file, external
environments, and datasets retain their own licenses. June compute was provided
by Washington University in St. Louis's Engineering IT GPU pool.
