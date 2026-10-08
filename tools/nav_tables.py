"""Markdown tables for the mapping-for-navigation study."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def map_table(scenes=("dynamic", "outdoor", "slab", "corridor")):
    rows = defaultdict(dict)
    for sc in scenes:
        f = ROOT / "results" / sc / "elevmap" / "metrics.json"
        if not f.exists():
            continue
        for name, r in json.loads(f.read_text()).items():
            rows[name][sc] = r
    out = ["| pose / dynamic handling | " + " | ".join(f"{s}: IoU / ghost m² / missed m²" for s in scenes) + " |",
           "|---|" + "---|" * len(scenes)]
    for name, d in rows.items():
        cells = []
        for s in scenes:
            r = d.get(s)
            cells.append("—" if r is None else f"{r['obst_iou']:.2f} / {r['ghost_m2']:.1f} / {r['missed_m2']:.1f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    det = ["", "| scene | Doppler point precision | recall |", "|---|---|---|"]
    for s in scenes:
        r = rows.get("gt__doppler", {}).get(s)
        if r and "dyn_precision" in r:
            det.append(f"| {s} | {r['dyn_precision']:.3f} | {r['dyn_recall']:.3f} |")
    return "\n".join(out + det)


def bench_table(scene="nav_clutter"):
    f = ROOT / "results" / "nav_bench" / f"{scene}_summary.json"
    res = json.loads(f.read_text()) if f.exists() else []
    if not res:   # partial: read per-run files
        for p in (ROOT / "results" / "nav_bench" / scene).glob("s*/*/nav_result.json"):
            res.append(json.loads(p.read_text()))
    by = defaultdict(list)
    for r in res:
        by[r["method"]].append(r)
    out = ["| method | runs | success | waypoints reached | collisions (mean) | falls | time [s] (succ.) | "
           "min clearance people [m] | person contacts (<0.5 m): all / robot-caused | planner ms (mean / p95) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rs in by.items():
        succ = [r for r in rs if r["completed"]]
        wp = np.mean([r["waypoints_reached"] / r["n_waypoints"] for r in rs])
        cp = [r["min_clear_people"] for r in rs if r["min_clear_people"] is not None]
        ms = [r["nav_ms"] for r in rs if r["nav_ms"]]
        ms95 = [r["nav_ms_p95"] for r in rs if r["nav_ms_p95"]]
        out.append(f"| {m} | {len(rs)} | {len(succ)}/{len(rs)} | {100 * wp:.0f}% | "
                   f"{np.mean([r['collisions'] for r in rs]):.1f} | {sum(r['fell'] for r in rs)} | "
                   f"{np.mean([r['duration'] for r in succ]):.0f} | " if succ else
                   f"| {m} | {len(rs)} | 0/{len(rs)} | {100 * wp:.0f}% | "
                   f"{np.mean([r['collisions'] for r in rs]):.1f} | {sum(r['fell'] for r in rs)} | — | ")
        out[-1] += (f"{np.min(cp):.2f} | " if cp else "— | ") + \
                   f"{sum(r.get('people_contacts', 0) for r in rs)} / {sum(r.get('robot_caused_contacts', 0) for r in rs)} | " + \
                   (f"{np.mean(ms):.0f} / {np.mean(ms95):.0f} |" if ms else "— |")
    return "\n".join(out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("maps", "all"):
        print(map_table())
    if what in ("bench", "all"):
        print()
        print(bench_table())
