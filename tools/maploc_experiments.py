"""All prior-map localization experiments -> results/maploc/summary.json

  OMP_NUM_THREADS=1 python3 tools/maploc_experiments.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAP = ROOT / "results/maploc/map_day0.npz"


def odom(scene, m="fmcw4d_leg"):
    return ROOT / "results" / scene / m / "poses_tum.txt"


def job(spec):
    from tools.maploc import run
    name, scene, kw = spec
    out = ROOT / "results/maploc/runs" / name
    try:
        r = run(ROOT / "data" / scene, kw.pop("odom", odom(scene)), out, MAP, **kw)
    except Exception as e:  # keep the batch going
        r = dict(error=repr(e)[:300])
    return name, r


def specs():
    S = []
    # 1) tracking from a known start, by amount of site change (+ ablations on day 7)
    for sc in ["slab", "slab_day1", "slab_day7"]:
        S.append((f"track_{sc}", sc, dict(init="gt", degeneracy=True)))
    S.append(("track_slab_day7_l2", "slab_day7", dict(init="gt", tukey_c=100.0)))
    S.append(("track_slab_day7_nodeskew", "slab_day7", dict(init="gt", do_deskew=False)))
    # 2) global localization from unknown pose, 5 start times, submap search vs particle filter
    for sc in ["slab_day1", "slab_day7"]:
        for ts in [3, 20, 40, 60, 80]:
            S.append((f"global_submap_{sc}_t{ts}", sc, dict(init="global", t_start=ts)))
            S.append((f"global_pf_{sc}_t{ts}", sc, dict(init="global", t_start=ts, global_method="pf")))
    # 3) kidnapped robot with a known gap (reboot / carried: no foot contact)
    for kd in [(40.0, 15.0), (70.0, 10.0)]:
        S.append((f"kidnap_known_{int(kd[0])}", "slab_day7", dict(init="gt", kidnap=kd)))
    # 4) silent odometry glitch (not announced): detection by the health monitor
    for g in [(30.0, 3.0, 0.0), (50.0, 0.0, 30.0), (75.0, 6.0, 0.0)]:
        S.append((f"glitch_{int(g[0])}_{g[1]:g}m_{g[2]:g}deg", "slab_day7", dict(init="gt", glitch=g)))
    # 5) long run (4 laps, ~255 m): map vs good and bad odometry
    for m in ["fmcw4d_leg", "legekf"]:
        S.append((f"long_{m}", "slab_day7_long", dict(init="gt", odom=odom("slab_day7_long", m))))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default=None, help="substring filter on run names")
    a = ap.parse_args()
    sp = [s for s in specs() if not a.only or any(o in s[0] for o in a.only.split(","))]
    out = ROOT / "results/maploc/summary.json"
    summary = json.loads(out.read_text()) if out.exists() else {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, r in ex.map(job, sp):
            summary[name] = r
            out.write_text(json.dumps(summary, indent=2))
            brief = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k in ("tracking_frac", "err_xy_rmse_tracking", "err_yaw_rmse_tracking", "fitness_median",
                              "time_to_localize_s", "lost_detect_delay_s", "time_to_relocalize_s",
                              "false_track_frac", "n_lost", "error")}
            print(f"{name:40s} {brief}", flush=True)


if __name__ == "__main__":
    main()
