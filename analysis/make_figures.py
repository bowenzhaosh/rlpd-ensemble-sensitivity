#!/usr/bin/env python3
"""All paper figures from the tidy CSVs -> paper/figures/*.pdf.

Every figure tolerates partial data (fleet still running): panels render from
whatever seeds exist; a figure with no data at all becomes a labeled
PENDING placeholder so the LaTeX build never breaks.

Figure -> paper mapping:
  fig_headline.pdf     Fig 1  final score vs N, drop 0 vs 0.01, pen + door
  fig_curves.pdf       Fig 2  pen learning curves, N in {2,10} x dropout
  fig_sharp_track.pdf  Fig 3  (a) normalized sharpness vs steps (b) sharpness->score scatter
  fig_prospective.pdf  Fig 4  Spearman(sharp@t, final) vs probe step t, LOCO band
  fig_tps.pdf          Fig 5  TPS dose-response, N=2 vs N=10 (pen) + door check
  fig_minq.pdf         Fig 6  M=1 vs M=2: |Q| blowup, score, sharpness (pessimism vs sharpness)
  fig_sigma_robust.pdf App    sharpness->score scatter at sigma in {.01,.05,.1}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlpd_common import PAPER_FIGS, TIDY, spearman_loco

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
# Okabe-Ito (colorblind safe)
C_NODROP, C_DROP, C_N2, C_N10, C_GREY = ("#0072B2", "#D55E00", "#009E73",
                                          "#CC79A7", "#999999")
SINGLE = (3.3, 2.3)   # single-column-ish
DOUBLE = (6.6, 2.3)

ENVS = ["pen-binary-v0", "door-binary-v0"]
ENV_SHORT = {"pen-binary-v0": "pen", "door-binary-v0": "door"}
PROBE_STEPS = list(range(0, 1000001, 50000))
LAST_WINDOW = 800000  # "final sharpness" = median of probes > this (rule 8)


def placeholder(path, msg):
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.text(.5, .5, f"PENDING DATA\n{msg}", ha="center", va="center",
            fontsize=9, color="crimson", transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [pending] {path.name}: {msg}")


def jitter(seeds, width=0.18):
    s = np.asarray(seeds, dtype=float)
    return (s - s.mean()) / max(s.max() - s.min(), 1) * width if len(s) > 1 else s * 0


def med_band(ax, groups, color, label, marker="o"):
    """Plot median with min-max band over a {x: values} dict."""
    xs = sorted(groups)
    med = [np.median(groups[k]) for k in xs]
    lo = [np.min(groups[k]) for k in xs]
    hi = [np.max(groups[k]) for k in xs]
    ax.plot(xs, med, marker=marker, ms=3.5, color=color, label=label, lw=1.2)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- fig 1
def fig_headline(runs):
    path = PAPER_FIGS / "fig_headline.pdf"
    r = runs[(runs["era"] == "washu") & (runs["status"] == "done")
             & (runs["mq"] == 2) & (runs["tps"] == 0)
             & runs["drop"].isin([0.0, 0.01])]
    if r.empty:
        return placeholder(path, "no done mq=2 runs yet")
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    for ax, env in zip(axes, ENVS):
        d = r[r["env"] == env]
        for drop, color, label in ((0.0, C_NODROP, "no dropout"),
                                   (0.01, C_DROP, r"dropout $p$=0.01")):
            g = d[d["drop"] == drop]
            if g.empty:
                continue
            groups = {nq: list(v) for nq, v in
                      g.groupby("nq")["final_frac"]}
            med_band(ax, groups, color, label)
            for nq, sub in g.groupby("nq"):
                ax.scatter(nq + jitter(sub["seed"]), sub["final_frac"],
                           s=6, color=color, alpha=0.55, zorder=3, lw=0)
        ax.set_xlabel("ensemble size $N$")
        ax.set_xticks([2, 4, 6, 10])
        ax.set_title(ENV_SHORT[env])
        ns = d.groupby(["nq", "drop"]).size()
        ax.text(.02, .98, f"seeds/cell: {int(ns.min())}–{int(ns.max())}" if len(ns) else "",
                transform=ax.transAxes, va="top", fontsize=6, color=C_GREY)
    axes[0].set_ylabel("final score (frac. of horizon in success)")
    axes[0].legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    print(f"  fig_headline.pdf: {len(r)} runs")


# ---------------------------------------------------------------- fig 2
def fig_curves(ts):
    path = PAPER_FIGS / "fig_curves.pdf"
    t = ts[(ts["env"] == "pen-binary-v0") & (ts["mq"] == 2) & (ts["tps"] == 0)
           & ts["drop"].isin([0.0, 0.01]) & ts["nq"].isin([2, 10])]
    if t.empty:
        return placeholder(path, "no pen timeseries yet")
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, nq in zip(axes, (2, 10)):
        for drop, color, label in ((0.0, C_NODROP, "no dropout"),
                                   (0.01, C_DROP, r"dropout $p$=0.01")):
            d = t[(t["nq"] == nq) & (t["drop"] == drop)]
            if d.empty:
                continue
            agg = d.groupby("step")["frac"].agg(["median", "min", "max"])
            ax.plot(agg.index / 1e6, agg["median"], color=color, lw=1.2,
                    label=f"{label} (n={d['seed'].nunique()})")
            ax.fill_between(agg.index / 1e6, agg["min"], agg["max"],
                            color=color, alpha=0.15, lw=0)
        ax.set_title(f"pen, $N$={nq}")
        ax.set_xlabel("environment steps (M)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("score (frac. of horizon)")
    fig.savefig(path)
    plt.close(fig)
    print(f"  fig_curves.pdf: {t['seed'].nunique()} seeds")


# -------------------------------------------------------------- helpers
def final_sharp(ts, sharp_col="sharp_s005"):
    """Per-run final sharpness = median of probes in the last 200k (rule 8)."""
    probes = ts[ts[sharp_col].notna() & (ts["step"] > LAST_WINDOW)]
    return (probes.groupby(["env", "nq", "mq", "drop", "tps", "seed"])
            [sharp_col].median().rename("final_sharp").reset_index())


# ---------------------------------------------------------------- fig 3
def fig_sharp_track(ts, runs):
    path = PAPER_FIGS / "fig_sharp_track.pdf"
    t = ts[(ts["env"] == "pen-binary-v0") & (ts["mq"] == 2) & (ts["tps"] == 0)
           & ts["drop"].isin([0.0, 0.01]) & ts["nq"].isin([2, 10])
           & ts["sharp_s005"].notna()]
    done = runs[(runs["era"] == "washu") & (runs["status"] == "done")
                & (runs["tps"] == 0)]
    if t.empty or done.empty:
        return placeholder(path, "no probe data yet")
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)

    ax = axes[0]
    styles = {(2, 0.0): (C_N2, "-"), (2, 0.01): (C_N2, "--"),
              (10, 0.0): (C_N10, "-"), (10, 0.01): (C_N10, "--")}
    for (nq, drop), (color, ls) in styles.items():
        d = t[(t["nq"] == nq) & (t["drop"] == drop)]
        if d.empty:
            continue
        agg = d.groupby("step")["sharp_s005"].median()
        ax.plot(agg.index / 1e6, agg.values, color=color, ls=ls, lw=1.2,
                label=f"$N$={nq}, " + ("$p$=0.01" if drop else "no drop"))
    ax.set_yscale("log")
    ax.set_xlabel("environment steps (M)")
    ax.set_ylabel(r"normalized sharpness $\tilde S$ ($\sigma$=0.05)")
    ax.legend(ncol=2, loc="upper right")
    ax.set_title("(a) sharpness during training (pen, median over seeds)")

    ax = axes[1]
    fs = final_sharp(ts)
    sc = fs.merge(done, on=["env", "nq", "mq", "drop", "tps", "seed"])
    sc = sc[sc["final_sharp"].notna() & sc["final_frac"].notna()]
    if sc.empty:
        ax.text(.5, .5, "pending", ha="center", transform=ax.transAxes)
    else:
        for env, marker in (("pen-binary-v0", "o"), ("door-binary-v0", "s")):
            d = sc[sc["env"] == env]
            if d.empty:
                continue
            cmap = {0.0: C_NODROP, 0.01: C_DROP}
            ax.scatter(d["final_sharp"], d["final_frac"], s=14, marker=marker,
                       c=[cmap.get(x, C_GREY) for x in d["drop"]],
                       alpha=0.75, lw=0, label=ENV_SHORT[env])
            for _, row in d[d["mq"] == 1].iterrows():
                ax.annotate("M=1", (row["final_sharp"], row["final_frac"]),
                            fontsize=5.5, color=C_GREY, xytext=(2, 2),
                            textcoords="offset points")
        pen = sc[sc["env"] == "pen-binary-v0"]
        st = spearman_loco(pen, "final_sharp", "final_frac",
                           ["nq", "mq", "drop"])
        ax.set_xscale("log")
        ax.set_xlabel(r"final normalized sharpness $\tilde S$")
        ax.set_ylabel("final score")
        ax.legend(loc="lower left")
        ax.set_title(rf"(b) $\tilde S$ vs score; pen $\rho_s$={st['rho']:.2f}"
                     rf" (n={st['n']})")
    fig.savefig(path)
    plt.close(fig)
    print(f"  fig_sharp_track.pdf: scatter n={len(sc)}")


# ---------------------------------------------------------------- fig 4
def fig_prospective(ts, runs):
    path = PAPER_FIGS / "fig_prospective.pdf"
    done = runs[(runs["era"] == "washu") & (runs["status"] == "done")
                & (runs["tps"] == 0) & (runs["env"] == "pen-binary-v0")]
    probes = ts[(ts["env"] == "pen-binary-v0") & (ts["tps"] == 0)
                & ts["sharp_s005"].notna()]
    if done.empty or probes.empty:
        return placeholder(path, "needs done runs + probes")
    keys = ["env", "nq", "mq", "drop", "tps", "seed"]
    merged = probes.merge(done[keys + ["final_frac"]], on=keys)
    rows = []
    for t_step in PROBE_STEPS[1:]:  # skip step 0 (masked anyway)
        d = merged[merged["step"] == t_step]
        if len(d) < 6:
            continue
        st = spearman_loco(d, "sharp_s005", "final_frac", ["nq", "mq", "drop"])
        rows.append(dict(step=t_step, **st))
    if not rows:
        return placeholder(path, "not enough runs for prospective curve")
    pr = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(pr["step"] / 1e6, pr["rho"], color=C_NODROP, lw=1.3, marker="o",
            ms=3, label=r"Spearman $\rho_s$")
    ax.fill_between(pr["step"] / 1e6, pr["loco_lo"], pr["loco_hi"],
                    color=C_NODROP, alpha=0.18, lw=0,
                    label="leave-one-config-out range")
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.axvline(0.1, color=C_GREY, lw=0.6, ls=":")
    ax.set_xlabel("probe step (M)")
    ax.set_ylabel(r"$\rho_s$(sharpness@$t$, final score)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)
    pr.to_csv(TIDY / "prospective.csv", index=False)
    print(f"  fig_prospective.pdf: {len(pr)} probe points, "
          f"n_runs@100k={int(pr[pr['step'] == 100000]['n'].iloc[0]) if (pr['step'] == 100000).any() else '–'}")


# ---------------------------------------------------------------- fig 5
def fig_tps(runs):
    path = PAPER_FIGS / "fig_tps.pdf"
    r = runs[(runs["era"] == "washu") & (runs["status"] == "done")
             & (runs["mq"] == 2) & (runs["drop"] == 0.0)]
    tps_done = r[r["tps"] > 0]
    if tps_done.empty:
        return placeholder(path, "TPS arm not finished (78877)")
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=False)
    for ax, env, sigmas in ((axes[0], "pen-binary-v0", [0, .1, .2, .3]),
                            (axes[1], "door-binary-v0", [0, .2])):
        d = r[(r["env"] == env) & r["nq"].isin([2, 10]) & r["tps"].isin(sigmas)]
        for nq, color in ((2, C_N2), (10, C_N10)):
            g = d[d["nq"] == nq]
            if g.empty:
                continue
            groups = {s: list(v) for s, v in g.groupby("tps")["final_frac"]}
            med_band(ax, groups, color, f"$N$={nq}")
            for s, sub in g.groupby("tps"):
                ax.scatter(s + jitter(sub["seed"], .012), sub["final_frac"],
                           s=6, color=color, alpha=0.5, lw=0)
        ax.set_xlabel(r"target-policy smoothing $\sigma_{TPS}$")
        ax.set_title(ENV_SHORT[env])
        ax.set_xticks(sigmas)
    axes[0].set_ylabel("final score")
    axes[0].legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    print(f"  fig_tps.pdf: {len(tps_done)} TPS runs")


# ---------------------------------------------------------------- fig 6
def fig_minq(ts, runs):
    path = PAPER_FIGS / "fig_minq.pdf"
    t = ts[(ts["env"] == "pen-binary-v0") & (ts["nq"] == 2) & (ts["tps"] == 0)]
    if t[t["mq"] == 1].empty:
        return placeholder(path, "no M=1 runs synced yet")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), constrained_layout=True)
    arms = [(1, 0.0, "#882255", "M=1"), (1, 0.01, "#882255", "M=1, drop"),
            (2, 0.0, C_NODROP, "M=2"), (2, 0.01, C_NODROP, "M=2, drop")]
    for ax, col, ylab, logy in (
            (axes[0], "q_abs_mean_diag", r"$|\bar Q|$ on probe batch", True),
            (axes[1], "frac", "score", False),
            (axes[2], "sharp_s005", r"normalized sharpness $\tilde S$", True)):
        for mq, drop, color, label in arms:
            d = t[(t["mq"] == mq) & (t["drop"] == drop) & t[col].notna()]
            if d.empty:
                continue
            agg = d.groupby("step")[col].median()
            ax.plot(agg.index / 1e6, agg.values, color=color,
                    ls="--" if drop else "-", lw=1.1, label=label)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("steps (M)")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("pen, $N$=2: pessimism (M) moves $|Q|$, not normalized sharpness",
                 fontsize=8, y=1.04)
    fig.savefig(path)
    plt.close(fig)
    print("  fig_minq.pdf written")


# ------------------------------------------------------------- appendix
def fig_sigma_robust(ts, runs):
    path = PAPER_FIGS / "fig_sigma_robust.pdf"
    done = runs[(runs["era"] == "washu") & (runs["status"] == "done")
                & (runs["tps"] == 0) & (runs["env"] == "pen-binary-v0")]
    if done.empty:
        return placeholder(path, "no done runs yet")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), sharey=True)
    keys = ["env", "nq", "mq", "drop", "tps", "seed"]
    wrote = False
    for ax, (sig, col) in zip(axes, (("0.01", "sharp_s001"),
                                     ("0.05", "sharp_s005"),
                                     ("0.1", "sharp_s01"))):
        fs = final_sharp(ts, col).rename(columns={"final_sharp": "fs"})
        sc = fs.merge(done[keys + ["final_frac"]], on=keys).dropna(
            subset=["fs", "final_frac"])
        if sc.empty:
            continue
        wrote = True
        ax.scatter(sc["fs"], sc["final_frac"], s=10,
                   c=[C_DROP if d else C_NODROP for d in sc["drop"]],
                   alpha=0.75, lw=0)
        st = spearman_loco(sc, "fs", "final_frac", ["nq", "mq", "drop"])
        ax.set_xscale("log")
        ax.set_title(rf"$\sigma$={sig}: $\rho_s$={st['rho']:.2f} (n={st['n']})")
        ax.set_xlabel(r"final $\tilde S_\sigma$")
    if not wrote:
        plt.close(fig)
        return placeholder(path, "no probe data")
    axes[0].set_ylabel("final score")
    fig.savefig(path)
    plt.close(fig)
    print("  fig_sigma_robust.pdf written")


def main():
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(TIDY / "runs.csv")
    ts = pd.read_csv(TIDY / "timeseries.csv")
    print("figures:")
    fig_headline(runs)
    fig_curves(ts)
    fig_sharp_track(ts, runs)
    fig_prospective(ts, runs)
    fig_tps(runs)
    fig_minq(ts, runs)
    fig_sigma_robust(ts, runs)


if __name__ == "__main__":
    main()
