#!/usr/bin/env python3
"""Build tidy CSVs from both data eras + a fleet progress report.

Outputs (analysis/out/tidy/):
  runs.csv        one row per run, both eras (April rows: scores only)
  timeseries.csv  June fleet: one row per (run, eval step), probe cols included
  progress.json   fleet completion vs the 86-run manifest (drives the
                  provisional banner in the paper)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlpd_common import (KEY, TIDY, load_april_runs, load_manifest,
                         load_june_runs, load_june_timeseries)


def main():
    TIDY.mkdir(parents=True, exist_ok=True)

    june = load_june_runs()
    april = load_april_runs()
    ts = load_june_timeseries()
    manifest = load_manifest()

    runs = pd.concat([june, april], ignore_index=True) if len(june) else april
    runs.to_csv(TIDY / "runs.csv", index=False)
    ts.to_csv(TIDY / "timeseries.csv", index=False)

    # --- progress vs manifest -------------------------------------------------
    done = june[june["status"] == "done"][KEY] if len(june) else pd.DataFrame(columns=KEY)
    m = manifest.merge(done.assign(done=True), on=KEY, how="left")
    m["done"] = m["done"].eq(True)
    prog = {
        "main_done": int(m[m["arm"] == "main"]["done"].sum()),
        "main_total": int((m["arm"] == "main").sum()),
        "tps_done": int(m[m["arm"] == "tps"]["done"].sum()),
        "tps_total": int((m["arm"] == "tps").sum()),
        "complete": bool(m["done"].all()),
        "missing": m[~m["done"]][KEY + ["arm"]].to_dict("records"),
    }
    (TIDY / "progress.json").write_text(json.dumps(prog, indent=2))

    print(f"runs.csv:       {len(runs)} rows "
          f"({len(june)} june / {len(april)} april)")
    print(f"timeseries.csv: {len(ts)} rows")
    print(f"fleet: main {prog['main_done']}/{prog['main_total']}, "
          f"tps {prog['tps_done']}/{prog['tps_total']}")
    if len(june):
        bad = june[june["status"] == "done"]["final_frac"].isna().sum()
        if bad:
            print(f"  [warn] {bad} done runs with unparseable final score")


if __name__ == "__main__":
    main()
