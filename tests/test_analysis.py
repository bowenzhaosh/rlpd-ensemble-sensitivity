"""Regression checks for the released evidence and analysis failure modes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rlpd_common import (ROOT, WASHU_RESULTS, dropout_pairs, frac_success,
                         load_manifest, parse_run_name, spearman_config_perm)
from validate_data import validate_fleet, verify_checksums


@pytest.fixture
def fleet(tmp_path):
    name = "pen-binary-v0_nq2_mq2_nodrop_s0"
    run = tmp_path / name
    shutil.copytree(WASHU_RESULTS / name, run)
    meta = parse_run_name(name)
    manifest = pd.DataFrame([{**meta, "max_steps": 1000000, "arm": "main"}])
    return tmp_path, run, manifest


def test_released_evidence():
    manifest = load_manifest()
    assert manifest.groupby("arm").size().to_dict() == {"main": 62, "tps": 24}
    assert validate_fleet(WASHU_RESULTS, manifest) == []
    op = load_manifest(((ROOT / "washu_runs_op.txt", False),))
    assert len(op) == 9
    assert validate_fleet(ROOT / "data/onpolicy-202606/results_op", op, onpolicy=True) == []
    assert verify_checksums() == []


@pytest.mark.parametrize("fault", ["missing_log", "missing_summary", "truncated", "duplicate_step",
                                  "missing_probe", "nonfinite_score", "wrong_score_scale",
                                  "summary_score", "summary_seed", "extra_run", "tagged_run",
                                  "extra_probe", "masked_probe", "wrong_ablation"])
def test_bad_evidence_fails(fleet, fault):
    results, run, manifest = fleet
    log_path = run / "online_log.csv"
    summary_path = run / "summary.json"
    if fault == "missing_log":
        log_path.unlink()
    elif fault == "missing_summary":
        summary_path.unlink()
    elif fault in {"summary_score", "summary_seed", "wrong_ablation"}:
        summary = json.loads(summary_path.read_text())
        if fault == "wrong_ablation":
            summary["bootstrap_mask"] = True
        else:
            summary["final_score" if fault == "summary_score" else "seed"] += 1
        summary_path.write_text(json.dumps(summary))
    elif fault == "extra_run":
        shutil.copytree(run, results / run.name.replace("_s0", "_s999"))
    elif fault == "tagged_run":
        shutil.copytree(run, results / run.name.replace("_s0", "_bmask_s0"))
    else:
        log = pd.read_csv(log_path)
        if fault == "truncated":
            log = log.iloc[:-1]
        elif fault == "duplicate_step":
            log.loc[1, "step"] = 0
        elif fault == "missing_probe":
            log.loc[log["step"] == 100000, "roughness"] = np.nan
        elif fault == "nonfinite_score":
            log.loc[1, "normalized_score"] = np.inf
        elif fault == "wrong_score_scale":
            log.loc[1, "normalized_score"] = 5
        elif fault == "extra_probe":
            columns = ["roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag"]
            log.loc[log["step"] == 805000, columns] = [1, 1, 1, 10]
        elif fault == "masked_probe":
            log.loc[log["step"] == 100000, "q_abs_mean_diag"] = 0
        log.to_csv(log_path, index=False)
    assert validate_fleet(results, manifest), f"{fault} was silently accepted"


def test_missing_manifest_run_fails(fleet):
    results, run, manifest = fleet
    shutil.rmtree(run)
    assert "Missing manifest run" in validate_fleet(results, manifest)[0]


def test_duplicate_manifest_fails(tmp_path):
    manifest = tmp_path / "runs.txt"
    manifest.write_text("pen-binary-v0,0,2,2,0,1000000\n" * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(((manifest, False),))


@pytest.mark.parametrize("constant", ["x", "y"])
def test_constant_input_never_reports_significance(constant):
    frame = pd.DataFrame({"config": range(5), "x": range(5), "y": range(5)})
    frame[constant] = 1
    result = spearman_config_perm(frame, "x", "y", ["config"], n_perm=20)
    assert np.isnan(result["rho"]) and np.isnan(result["p"])


def test_permutation_uses_config_medians():
    frame = pd.DataFrame({"config": range(5), "x": range(5), "y": range(5)})
    result = spearman_config_perm(frame, "x", "y", ["config"], n_perm=99)
    repeated = spearman_config_perm(pd.concat([frame] * 3), "x", "y", ["config"], n_perm=99)
    assert result == repeated
    assert result["rho"] == pytest.approx(1)
    assert 0 < result["p"] <= 1


def test_one_dropout_arm_produces_no_pairs():
    frame = pd.DataFrame([{"era": "washu", "status": "done", "env": "pen-binary-v0",
                           "mq": 2, "nq": 2, "tps": 0., "drop": 0., "seed": 0, "final_frac": .5}])
    assert dropout_pairs(frame).empty


@pytest.mark.parametrize("env,horizon", [("pen-binary-v0", 100), ("door-binary-v0", 200)])
def test_success_fraction_endpoints(env, horizon):
    assert frac_success(-100 * horizon, env) == 0
    assert frac_success(0, env) == 1


@pytest.mark.parametrize("args", [["--unknown"], ["--sync", "--local"]])
def test_build_rejects_invalid_flags_before_work(args):
    result = subprocess.run(["bash", str(ROOT / "analysis/run_all.sh"), *args], capture_output=True, text=True)
    assert result.returncode == 2 and "ERROR:" in result.stderr


def test_build_honors_missing_interpreter():
    result = subprocess.run(["bash", str(ROOT / "analysis/run_all.sh"), "--no-pdf"],
                            env={**os.environ, "RLPD_PY": "/nonexistent/rlpd-python"},
                            capture_output=True, text=True)
    assert result.returncode == 1 and "Python executable not found" in result.stderr


def test_missing_latex_is_an_error(tmp_path):
    # Supply only dirname on PATH, so even a machine with LaTeX tests this path.
    (tmp_path / "dirname").symlink_to(shutil.which("dirname"))
    result = subprocess.run(["/bin/bash", str(ROOT / "analysis/run_all.sh")],
                            env={**os.environ, "PATH": str(tmp_path), "RLPD_PY": sys.executable},
                            capture_output=True, text=True)
    assert result.returncode == 1 and "latexmk is required" in result.stderr


def test_shell_scripts_parse():
    scripts = sorted(ROOT.glob("*.sh")) + sorted(ROOT.glob("*.sbatch")) + sorted((ROOT / "analysis").glob("*.sh"))
    for path in scripts:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
