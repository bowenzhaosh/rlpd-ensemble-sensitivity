# Contributing

Open an issue for reproduction problems, including the command, repository
commit, Python/platform versions, and relevant error output. Remove credentials
and private paths from logs before posting.

## Local checks

Use Python 3.11 in a virtual environment:

```bash
python -m pip install -r requirements/dev.txt -e .
python -m pytest -q
python -m ruff check .
cffconvert --validate
python analysis/validate_data.py --checksums --onpolicy
bash analysis/run_all.sh --no-pdf
# With LaTeX installed:
bash analysis/run_all.sh
```

CI validates inputs, tests analysis failure modes, checks Python and shell syntax,
rebuilds the artifacts, compares generated tables/tidy data, and compiles the PDF.
It does not install the GPU training stack or rerun the experiments.

## Changes to evidence and methods

Treat archived inputs as research evidence. Do not edit raw scores, delete failed
runs to improve results, or regenerate `data/checksums.sha256` to conceal an
unexplained difference. Document any intentional correction with its provenance.

Tables, result macros, and figure files are generated. Change the analysis source
and rebuild them together. Record methodological changes separately from cleanup;
keep the recorded analysis plan and later amendments distinguishable. Use new
run identities for changed training behavior and report all relevant seeds.

The repository's public metadata identifies the authors. It is not an anonymous
review artifact. Build and inspect any separate review package under the target
venue's requirements before submission.
