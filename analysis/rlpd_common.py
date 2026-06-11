"""Shared constants, parsers, and statistics for the RLPD sharpness analysis.

PRE-REGISTERED ANALYSIS DISCIPLINE (written 2026-06-11, while the fleet was
12/62 complete and the TPS arm had not started — i.e. before seeing the data
these rules will be applied to):

  1. Aggregate across seeds with MEDIAN; report MIN-MAX bands. Never
     mean +/- SEM at n <= 5.
  2. Cross-config consistency = exact two-sided binomial sign test where each
     config contributes ONE unit (the direction of its seed-median contrast).
  3. Same-seed dropout contrasts on the same harness may be reported as
     seed-level pairs (sign test over seed x config pairs). TPS arms are NEVER
     per-seed-paired with baselines (TPS consumes one extra RNG split per
     critic update -> different trajectories at equal seed): distribution
     comparisons only (median/min-max + Mann-Whitney U).
  4. No single-seed claims. Configs with < 2 done seeds are excluded from
     tests and flagged provisional in tables.
  5. Sharpness used everywhere = roughness / q_abs_mean_diag^2 ("normalized
     sharpness"; Q-scale confound fix). Probe rows with q_abs_mean_diag < 1.0
     are masked (early-training scale degeneracy; in practice only step 0).
  6. sigma = 0.05 is the headline probe scale; {0.01, 0.1} form the
     robustness sweep (conclusions must hold in ordering across all three).
  7. Prospective test: Spearman rho between normalized sharpness at probe
     step t and final score, across done runs; primary = pen, all configs
     pooled, leave-one-config-out range reported; secondary = excluding the
     divergent (mq=1, nodrop) configs.
  8. "Final score" = summary.json final_score (mean of last 10 evals), fixed
     by the harness before launch. "Final sharpness" = median of the probes
     in the last 200k steps (5 probes).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
APRIL_TRACKER = ROOT / "data" / "april" / "run_tracker.csv"
WASHU_RESULTS = ROOT / "data" / "washu-202606" / "results"
OUT = ROOT / "analysis" / "out"
TIDY = OUT / "tidy"
PAPER_FIGS = ROOT / "paper" / "figures"
PAPER_TABLES = ROOT / "paper" / "tables"

# --- env constants ----------------------------------------------------------
# Episode horizons (RLPD Adroit binary setup): reward is -1 per step until the
# success condition, 0 afterwards, so return in [-H, 0] and
# 1 + return/H = fraction of the evaluation horizon spent in success.
HORIZON = {"pen-binary-v0": 100, "door-binary-v0": 200}

# online_log.csv "normalized_score" = per-episode return x 100 (harness
# fallback path for the binary envs), i.e. in [-100*H, 0].
def frac_success(score100, env):
    """Map harness score (return x 100) -> fraction-of-horizon-in-success in [0,1]."""
    return 1.0 + score100 / (100.0 * HORIZON[env])


# --- run-name parsing (June fleet) -------------------------------------------
# train_abc.py:     <env>_nq<N>_mq<M>_<nodrop|drop<p>>[_<tag>]_s<seed>
# train_abc_tps.py: <env>_nq<N>_mq<M>_<nodrop|drop<p>>_tps<sigma>[_<tag>]_s<seed>
_RUN_RE = re.compile(
    r"^(?P<env>[a-zA-Z0-9-]+?-v\d+)"
    r"_nq(?P<nq>\d+)_mq(?P<mq>\d+)"
    r"_(?P<droptag>nodrop|drop[0-9.]+)"
    r"(?:_tps(?P<tps>[0-9.]+))?"
    r"(?:_(?P<tag>.+?))?"
    r"_s(?P<seed>\d+)$"
)


def parse_run_name(name):
    m = _RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    drop = 0.0 if d["droptag"] == "nodrop" else float(d["droptag"][4:])
    return {
        "env": d["env"],
        "nq": int(d["nq"]),
        "mq": int(d["mq"]),
        "drop": drop,
        "tps": float(d["tps"]) if d["tps"] else 0.0,
        "tag": d["tag"] or "baseline",
        "seed": int(d["seed"]),
    }


# --- manifests ----------------------------------------------------------------
def load_manifest():
    """The 62+24 run fleet manifest from washu_runs*.txt -> DataFrame."""
    rows = []
    for fname, has_tps in (("washu_runs.txt", False), ("washu_runs_tps.txt", True)):
        for line in (ROOT / fname).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            rows.append({
                "env": f[0], "seed": int(f[1]), "nq": int(f[2]), "mq": int(f[3]),
                "drop": float(f[4]), "tps": float(f[5]) if has_tps else 0.0,
                "arm": "tps" if has_tps else "main",
            })
    return pd.DataFrame(rows)


KEY = ["env", "nq", "mq", "drop", "tps", "seed"]


# --- loaders -------------------------------------------------------------------
def load_washu_runs():
    """One row per June-fleet run dir. status: done (summary.json) / running."""
    rows = []
    if not WASHU_RESULTS.exists():
        return pd.DataFrame()
    for d in sorted(WASHU_RESULTS.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run_name(d.name)
        if meta is None:
            print(f"  [warn] unparseable run dir skipped: {d.name}")
            continue
        row = dict(meta, era="washu", run_dir=d.name)
        sj = d / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            row.update(
                status="done",
                final_score=s["final_score"],
                peak_score=s["peak_score"],
                wall_hours=s.get("wall_hours", np.nan),
            )
        else:
            row.update(status="running", final_score=np.nan, peak_score=np.nan,
                       wall_hours=np.nan)
        log = d / "online_log.csv"
        row["last_step"] = np.nan
        if log.exists():
            try:
                steps = pd.read_csv(log, usecols=["step"])["step"]
                row["last_step"] = int(steps.max())
            except Exception:
                pass
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["final_frac"] = [
            frac_success(s, e) if pd.notna(s) else np.nan
            for s, e in zip(df["final_score"], df["env"])
        ]
    return df


def load_washu_timeseries():
    """Long table: one row per (run, eval step) with score + probe columns."""
    frames = []
    if not WASHU_RESULTS.exists():
        return pd.DataFrame()
    for d in sorted(WASHU_RESULTS.iterdir()):
        log = d / "online_log.csv"
        meta = parse_run_name(d.name)
        if meta is None or not log.exists():
            continue
        try:
            t = pd.read_csv(log)
        except Exception as e:
            print(f"  [warn] bad log {d.name}: {e}")
            continue
        for k, v in meta.items():
            t[k] = v
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    ts = pd.concat(frames, ignore_index=True)
    for c in ("roughness", "roughness_s001", "roughness_s01",
              "q_abs_mean_diag", "normalized_score"):
        if c in ts.columns:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")
    ts["frac"] = [frac_success(s, e)
                  for s, e in zip(ts["normalized_score"], ts["env"])]
    # normalized sharpness (rule 5); mask q_abs < 1
    q2 = ts["q_abs_mean_diag"].where(ts["q_abs_mean_diag"] >= 1.0) ** 2
    for sig, col in (("s001", "roughness_s001"), ("s005", "roughness"),
                     ("s01", "roughness_s01")):
        ts[f"sharp_{sig}"] = ts[col] / q2
    return ts


def load_april_runs():
    """April-era curated tracker -> same schema as load_washu_runs (scores only)."""
    # Hand-edited tracker quirk: TODO rows carry one extra empty field
    # (13 instead of 12); drop the surplus empty so the row parses.
    def _fix(bad):
        return (bad[:7] + bad[8:]) if len(bad) == 13 else bad[:12]

    a = pd.read_csv(APRIL_TRACKER, engine="python", on_bad_lines=_fix)
    a = a.rename(columns={"dropout": "drop"})
    a["era"], a["tps"], a["tag"] = "april", 0.0, "baseline"
    a["status"] = a["status"].str.strip()
    a["final_frac"] = [
        frac_success(s, e) if pd.notna(s) and e in HORIZON else np.nan
        for s, e in zip(pd.to_numeric(a["final_score"], errors="coerce"), a["env"])
    ]
    return a


# --- statistics (rules 1-4) -----------------------------------------------------
def median_band(g):
    """median / min / max / n over a seed group of finals."""
    v = g.dropna()
    return pd.Series({"med": v.median(), "lo": v.min(), "hi": v.max(),
                      "n": int(v.size)})


def sign_test(n_pos, n_tot):
    """Exact two-sided binomial sign test, p(0.5)."""
    if n_tot == 0:
        return np.nan
    return stats.binomtest(n_pos, n_tot, 0.5).pvalue


def dropout_pairs(runs, mq=2):
    """Per-(env, nq, seed) paired contrast: final_frac(drop=.01) - final_frac(0).

    Same harness + same seed (rule 3 allows seed pairing for dropout).
    June (washu) era ONLY — April rows are corroboration-only and must never
    enter a paired claim (and would silently average into pivot cells).
    Returns tidy frame of pairs where BOTH arms are done.
    """
    r = runs[(runs["era"] == "washu") & (runs["status"] == "done")
             & (runs["mq"] == mq)
             & (runs["tps"] == 0.0) & runs["drop"].isin([0.0, 0.01])]
    piv = r.pivot_table(index=["env", "nq", "seed"], columns="drop",
                        values="final_frac")
    piv = piv.dropna(subset=[0.0, 0.01])
    piv["delta"] = piv[0.01] - piv[0.0]
    return piv.reset_index()


def spearman_loco(df, xcol, ycol, config_cols):
    """Overall Spearman rho + leave-one-config-out range. (rule 7)"""
    d = df.dropna(subset=[xcol, ycol])
    if len(d) < 4:
        return dict(rho=np.nan, p=np.nan, n=len(d), loco_lo=np.nan,
                    loco_hi=np.nan, n_configs=0)
    rho, p = stats.spearmanr(d[xcol], d[ycol])
    configs = list(d.groupby(config_cols).groups)
    locos = []
    for c in configs:
        mask = ~(d.set_index(config_cols).index == c)
        sub = d[np.asarray(mask)]
        if sub[xcol].nunique() > 2 and len(sub) >= 4:
            locos.append(stats.spearmanr(sub[xcol], sub[ycol])[0])
    return dict(rho=rho, p=p, n=len(d),
                loco_lo=min(locos) if locos else np.nan,
                loco_hi=max(locos) if locos else np.nan,
                n_configs=len(configs))


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U (rule 3 for TPS arms); returns p or NaN."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
