#!/usr/bin/env python3
"""Validate the released logs before generating publication artifacts.

Checks manifest membership, evaluation/probe coverage, summary metadata, and
agreement between summary scores and the underlying evaluations. This validates
the archived evidence; it does not establish deterministic training reruns.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rlpd_common import HORIZON, KEY, ROOT, JUNE_RESULTS, load_manifest, parse_run_name

PROBE_COLUMNS = ("roughness", "roughness_s001", "roughness_s01", "q_abs_mean_diag")


def validate_fleet(results, manifest, onpolicy=False):
    """Return all validation errors; never write to the input or output tree."""
    errors = []
    if manifest.duplicated(KEY).any():
        return ["Duplicate run keys in manifest"]
    expected = {tuple(row[k] for k in KEY): row for row in manifest.to_dict("records")}
    seen = set()
    if not results.is_dir():
        return [f"Results directory missing: {results}"]
    for run in sorted(p for p in results.iterdir() if p.is_dir()):
        meta = parse_run_name(run.name)
        if meta is None or meta["tag"] != "baseline":
            errors.append(f"{run.name}: unexpected or tagged run directory")
            continue
        key = tuple(meta[k] for k in KEY)
        if key not in expected:
            errors.append(f"{run.name}: run is absent from the manifest")
            continue
        if key in seen:
            errors.append(f"{run.name}: duplicate run key")
            continue
        seen.add(key)
        try:
            summary = json.loads((run / "summary.json").read_text())
            log = pd.read_csv(run / "online_log.csv")
            required = {"step", "normalized_score", "tag", *PROBE_COLUMNS}
            if onpolicy:
                required.update({"roughness_onpolicy", "q_abs_mean_diag_onpolicy"})
            missing = required - set(log.columns)
            if missing:
                raise ValueError(f"missing log columns: {sorted(missing)}")
            steps = pd.to_numeric(log["step"], errors="raise").to_numpy()
            expected_steps = np.arange(0, expected[key]["max_steps"] + 1, 5000)
            if not np.array_equal(steps, expected_steps):
                raise ValueError("evaluation steps are missing, duplicated, unordered, or truncated")
            if not log["tag"].eq("baseline").all():
                raise ValueError("log tag disagrees with baseline run name")
            scores = pd.to_numeric(log["normalized_score"], errors="raise").to_numpy()
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite evaluation score")
            if ((scores < -100 * HORIZON[meta["env"]] - 1e-6) | (scores > 1e-6)).any():
                raise ValueError("evaluation score outside the binary environment return scale")
            probes = log.loc[log["step"] % 50000 == 0, list(PROBE_COLUMNS)].to_numpy(dtype=float)
            if not np.isfinite(probes).all() or (probes < 0).any():
                raise ValueError("missing, nonfinite, or negative scheduled probe values")
            off_grid = log.loc[log["step"] % 50000 != 0, list(PROBE_COLUMNS)]
            if off_grid.notna().any().any():
                raise ValueError("unexpected probe values outside the 50k schedule")
            postzero = log.loc[(log["step"] > 0) & (log["step"] % 50000 == 0), "q_abs_mean_diag"]
            if (postzero < 1).any():
                raise ValueError("postzero Q scale below 1 would mask required normalized probes")
            if onpolicy:
                mask = (log["step"] > 0) & (log["step"] % 50000 == 0)
                probes_op = log.loc[mask, ["roughness_onpolicy", "q_abs_mean_diag_onpolicy"]].to_numpy(dtype=float)
                if not np.isfinite(probes_op).all() or (probes_op < 0).any():
                    raise ValueError("missing, nonfinite, or negative on-policy probe values")
            for field, value in {"env": meta["env"], "seed": meta["seed"],
                                 "nqs": meta["nq"], "tag": "baseline", "ln": True,
                                 "bootstrap_mask": False, "independent_targets": False,
                                 "critic_reset_step": 0}.items():
                if summary[field] != value:
                    raise ValueError(f"summary {field} disagrees with run name")
            if summary.get("target_smoothing_sigma", 0.0) != meta["tps"]:
                raise ValueError("summary TPS sigma disagrees with run name")
            for field, value in {"final_score": scores[-10:].mean(),
                                 "peak_score": scores.max()}.items():
                if not np.isclose(float(summary[field]), value, rtol=1e-10, atol=1e-6):
                    raise ValueError(f"summary {field} disagrees with raw evaluations")
            if summary["peak_step"] != int(steps[np.argmax(scores)]):
                raise ValueError("summary peak_step disagrees with raw evaluations")
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            errors.append(f"{run.name}: {exc}")
    for key in sorted(set(expected) - seen):
        errors.append(f"Missing manifest run: {dict(zip(KEY, key))}")
    return errors


def verify_checksums(root=ROOT):
    """Check the fixed release input inventory, including missing/extra entries."""
    errors = []
    inventory = root / "data" / "checksums.sha256"
    expected_paths = {
        *root.glob("data/june-2026/results/*/online_log.csv"),
        *root.glob("data/june-2026/results/*/summary.json"),
        *root.glob("data/onpolicy/results/*/online_log.csv"),
        *root.glob("data/onpolicy/results/*/summary.json"),
        root / "data/april/run_tracker.csv",
        root / "experiments/grid.txt", root / "experiments/tps.txt", root / "experiments/onpolicy.txt",
    }
    seen = set()
    try:
        for line in inventory.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            path = root / relative
            if path not in expected_paths or path in seen:
                errors.append(f"Unexpected/duplicate checksum entry: {relative}")
                continue
            seen.add(path)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"Checksum mismatch: {relative}")
    except (OSError, ValueError) as exc:
        errors.append(f"Invalid checksum inventory: {exc}")
    for path in sorted(expected_paths - seen):
        errors.append(f"Missing checksum entry: {path.relative_to(root)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", action="store_true", help="also verify the fixed release input hashes")
    parser.add_argument("--onpolicy", action="store_true", help="also validate the nine supplementary on-policy runs")
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        errors = validate_fleet(JUNE_RESULTS, manifest)
        if args.onpolicy:
            op = load_manifest(((ROOT / "experiments/onpolicy.txt", False),))
            errors += validate_fleet(ROOT / "data/onpolicy/results", op, onpolicy=True)
        if args.checksums:
            errors += verify_checksums()
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Validated {len(manifest)} main/TPS runs: manifest, scores, evaluation steps, and probes.")
    if args.onpolicy:
        print(f"Validated {len(op)} supplementary on-policy runs.")
    if args.checksums:
        print("Release input checksums match.")


if __name__ == "__main__":
    main()
