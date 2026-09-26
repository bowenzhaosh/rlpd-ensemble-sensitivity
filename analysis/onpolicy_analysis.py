#!/usr/bin/env python3
"""Compare on-policy and offline sharpness in the supplementary pen runs.

Each run measures both probes. Configurations are the analysis unit, with one
seed per configuration (nine total, eight excluding M=1).

Usage: python analysis/onpolicy_analysis.py [results_dir]
Default input: data/onpolicy/results.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = (Path(sys.argv[1]) if len(sys.argv) > 1
       else Path(__file__).resolve().parent.parent / "data" / "onpolicy"
       / "results")
H = {"pen-binary-v0": 100, "door-binary-v0": 200}
WIDE = {"rho": np.nan, "p": np.nan}
RE = re.compile(r"(?P<env>[a-z-]+-v\d+)_nq(?P<nq>\d+)_mq(?P<mq>\d+)"
                r"_(?P<drop>nodrop|drop[0-9.]+)_s(?P<seed>\d+)")


def frac(s, env):
    return 1.0 + s / (100.0 * H[env])


def parse(name):
    m = RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return dict(env=d["env"], nq=int(d["nq"]), mq=int(d["mq"]),
                drop=0.0 if d["drop"] == "nodrop" else float(d["drop"][4:]),
                seed=int(d["seed"]))


def perm_p(x, y, B=20000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    r = spearmanr(x, y)[0]
    rng = np.random.RandomState(seed)
    c = sum(abs(spearmanr(x, rng.permutation(y))[0]) >= abs(r) - 1e-12
            for _ in range(B))
    return r, (c + 1) / (B + 1)


def loco_range(x, y, cfgs):
    """Leave-one-config-out range of Spearman rho."""
    x, y, cfgs = np.asarray(x), np.asarray(y), np.asarray(cfgs)
    rs = [spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    return min(rs), max(rs)


def main():
    rows = []
    wide_rows = []
    for d in sorted(RES.glob("*")):
        sj, log = d / "summary.json", d / "online_log.csv"
        meta = parse(d.name)
        if meta is None or not sj.exists() or not log.exists():
            continue
        s = json.loads(sj.read_text())
        t = pd.read_csv(log)
        for c in ("roughness", "q_abs_mean_diag",
                  "roughness_onpolicy", "q_abs_mean_diag_onpolicy"):
            t[c] = pd.to_numeric(t.get(c), errors="coerce")
        fw = t[t["step"] > 800000]
        if fw["roughness"].notna().sum() == 0:
            continue
        w6 = t[t["step"] > 600000]
        wide_rows.append(dict(**meta, final_frac=frac(s["final_score"], meta["env"]),
                              sharp_op=(w6["roughness_onpolicy"]
                                        / w6["q_abs_mean_diag_onpolicy"] ** 2).median()))
        rows.append(dict(
            **meta, final_frac=frac(s["final_score"], meta["env"]),
            sharp_off=(fw["roughness"] / fw["q_abs_mean_diag"] ** 2).median(),
            sharp_op=(fw["roughness_onpolicy"]
                      / fw["q_abs_mean_diag_onpolicy"] ** 2).median(),
            rough_off=fw["roughness"].median(),
            rough_op=fw["roughness_onpolicy"].median(),
            q_off=fw["q_abs_mean_diag"].median(),
            q_op=fw["q_abs_mean_diag_onpolicy"].median()))
    df = pd.DataFrame(rows)
    wd = pd.DataFrame(wide_rows)
    if len(wd):
        wd = wd[(wd["mq"] == 2) & (wd["sharp_op"] > 0)]
        WIDE.update(dict(zip(("rho", "p"), perm_p(np.log(wd["sharp_op"]), wd["final_frac"])))
                    if len(wd) >= 4 else dict(rho=np.nan, p=np.nan))
    print(f"=== results_op: {len(df)}/9 runs with final-window data ===")
    if len(df) < 4:
        print("  too few runs yet; rerun when more land.")
        return
    cols = ["nq", "mq", "drop", "final_frac", "sharp_off", "sharp_op",
            "rough_off", "rough_op", "q_off", "q_op"]
    pd.set_option("display.width", 200)
    print(df[cols].sort_values(["mq", "nq", "drop"]).to_string(index=False))

    print("\n=== CONFIG-LEVEL: which sharpness tracks score better? "
          "(Spearman, permutation p) ===")
    for label, sub in (("ALL", df), ("excl M=1", df[df["mq"] == 2])):
        line = f"  {label:8s} (n={len(sub)}): "
        for col in ("sharp_off", "sharp_op"):
            d = sub.dropna(subset=[col, "final_frac"])
            d = d[d[col] > 0]
            if len(d) >= 4:
                r, p = perm_p(np.log(d[col]), d["final_frac"])
                lo, hi = loco_range(np.log(d[col]), d["final_frac"], d.index)
                line += f"{col} rho={r:+.2f} p={p:.3f} LOCO[{lo:+.2f},{hi:+.2f}]   "
        print(line)
    sub = df[df["mq"] == 2].dropna(subset=["rough_op", "final_frac"])
    r, p = perm_p(np.log(sub["rough_op"]), sub["final_frac"])
    print(f"  raw on-policy roughness (unnormalized) vs score, excl M=1: "
          f"rho={r:+.2f} p={p:.3f}")
    print(f"  wider final window (step > 600k), on-policy sharpness excl M=1: "
          f"rho={WIDE['rho']:+.2f} p={WIDE['p']:.3f}")

    print("\n=== INTERACTION (descriptive, single seed): dropout dlog-sharp "
          "@N=2 vs @N=10 — more negative @N2 = smooths more at small N ===")
    pen = df[(df["env"] == "pen-binary-v0") & (df["mq"] == 2)]
    for col in ("sharp_off", "sharp_op"):
        def dl(nq):
            a = pen[(pen["nq"] == nq) & (pen["drop"] == 0.0)][col]
            b = pen[(pen["nq"] == nq) & (pen["drop"] == 0.01)][col]
            return (np.log(b.values[0]) - np.log(a.values[0])
                    if len(a) and len(b) and a.values[0] > 0 and b.values[0] > 0
                    else np.nan)
        print(f"  {col}: dlog@N2={dl(2):+.2f}  dlog@N10={dl(10):+.2f}")

    print("\n=== where does the actor sit? (on-policy vs offline, ratios) ===")
    df2 = df.dropna(subset=["rough_op", "rough_off"])
    print(f"  roughness on/off ratio (median): "
          f"{(df2['rough_op'] / df2['rough_off']).median():.2f}x")
    print(f"  |Q| on/off ratio (median): "
          f"{(df2['q_op'] / df2['q_off']).median():.2f}x")


if __name__ == "__main__":
    main()
