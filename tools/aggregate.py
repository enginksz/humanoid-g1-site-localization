"""Aggregate summary.json over seeds -> results/table.md and results/aggregate.json.

  python3 tools/aggregate.py results results_s1 results_s2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["slab", "dynamic", "sliding", "corridor", "corridor_boards", "roof", "roof_boards", "outdoor"]
METHODS = ["fastlio2", "fmcw3d", "fmcw4d", "legekf", "fmcw3d_leg", "fmcw3d_leg_pg", "fmcw4d_leg_ng", "fmcw4d_leg"]
NAMES = {
    "fastlio2": "FAST-LIO2 (original code)",
    "fmcw3d": "3D LIO (FMCW-LIO, Doppler off)",
    "fmcw4d": "FMCW-LIO 4D (Doppler)",
    "legekf": "Leg-IMU EKF",
    "fmcw3d_leg": "3D LIO + leg",
    "fmcw3d_leg_pg": "3D LIO + leg, prediction-gated",
    "fmcw4d_leg_ng": "4D + leg, ungated",
    "fmcw4d_leg": "4D + leg + Doppler slip gate",
}


def main():
    dirs = [Path(d) for d in (sys.argv[1:] or ["results"])]
    sums = [json.loads((d / "summary.json").read_text()) for d in dirs if (d / "summary.json").exists()]
    agg = {}
    for s in SCENES:
        for m in METHODS:
            vals = [x[s][m] for x in sums if s in x and m in x[s] and not x[s][m].get("failed")]
            if not vals:
                continue
            ate = np.array([v["ate_rmse"] for v in vals])
            end = np.array([v["final_err_pct"] for v in vals])
            agg.setdefault(s, {})[m] = dict(n=len(vals), ate_mean=float(ate.mean()), ate_std=float(ate.std()),
                                           end_pct_mean=float(end.mean()), end_pct_std=float(end.std()))
    (ROOT / "results" / "aggregate.json").write_text(json.dumps(agg, indent=2))

    lines = [f"ATE RMSE [m] (SE3-aligned), mean ± std over {len(sums)} seeds. Best per scene in bold.\n",
             "| method | " + " | ".join(SCENES) + " |", "|---|" + "---|" * len(SCENES)]
    best = {s: min(agg[s], key=lambda m: agg[s][m]["ate_mean"]) for s in agg}
    for m in METHODS:
        cells = []
        for s in SCENES:
            r = agg.get(s, {}).get(m)
            if r is None:
                cells.append("–")
                continue
            c = f"{r['ate_mean']:.3f} ± {r['ate_std']:.3f}"
            cells.append(f"**{c}**" if best[s] == m else c)
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines += ["", "End-point drift [% of path], first-pose aligned (no loop closure), mean ± std.\n",
              "| method | " + " | ".join(SCENES) + " |", "|---|" + "---|" * len(SCENES)]
    for m in METHODS:
        cells = []
        for s in SCENES:
            r = agg.get(s, {}).get(m)
            cells.append("–" if r is None else f"{r['end_pct_mean']:.2f} ± {r['end_pct_std']:.2f}")
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    (ROOT / "results" / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
