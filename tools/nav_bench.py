"""Navigation benchmark: success / collisions / clearance per navigation method.

Methods
  blind          waypoint follower, no obstacle information
  reactive_gt    previous heading-fan steering on a GROUND-TRUTH costmap (sim geometry)
  mppi_none      LiDAR map (no dynamic handling) + navigation function + MPPI
  mppi_raycast   same, free-space carving clears moving objects
  mppi_doppler   same, FMCW Doppler removes moving objects and feeds the person/vehicle tracker

  python3 tools/nav_bench.py --scene nav_clutter --seeds 0 1 2 3 4 --jobs 6
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = {
    "blind": dict(),
    "reactive_gt": dict(costmap=True),
    "mppi_none": dict(nav="mppi", nav_dyn="none"),
    "mppi_raycast": dict(nav="mppi", nav_dyn="raycast"),
    "mppi_doppler": dict(nav="mppi", nav_dyn="doppler"),
    "mppi_persist": dict(nav="mppi", nav_dyn="persist"),
    "mppi_doppler_persist": dict(nav="mppi", nav_dyn="doppler+persist"),
    "mppi_doppler_raycast": dict(nav="mppi", nav_dyn="doppler+raycast"),
}


def static_raster(scene, seed, res=0.1):
    """True static obstacle footprint (no inflation) over the whole scene, for clearance."""
    import mujoco
    from sim import scenes as S
    from sim.costmap import CostmapConfig, GtCostmap
    from sim.record import G1_XML
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    sc = S.build(spec, scene, seed=seed)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w = sc.waypoints
    c = (w.min(0) + w.max(0)) / 2
    half = float(np.max(w.max(0) - w.min(0)) / 2 + 8)
    gc = GtCostmap(m, terrain=None, cfg=CostmapConfig(res=res, half=half, robot_radius=0.0))
    gc.update(d, (), xy=c)
    return gc.cost >= 0.99, gc.origin.copy(), res, sc


def run_one(job):
    scene, seed, method, out_root, vx, t_max = job
    import torch
    torch.set_num_threads(1)
    from scipy.ndimage import distance_transform_edt
    from sim.record import Recorder
    out = Path(out_root) / scene / f"s{seed}" / method
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(scene, seed=seed, vx_max=vx, **METHODS[method])
    meta = rec.run(out, t_max=t_max, verbose=False)
    (out / "lidar.bin").unlink(missing_ok=True)
    gt = np.load(out / "gt.npz")
    t, p = gt["t"], gt["p"][:, :2]
    occ, org, res, sc = static_raster(scene, seed)
    dist = distance_transform_edt(~occ) * res
    ij = np.clip(((p - org) / res).astype(int), 0, np.array(occ.shape) - 1)
    clr_static = dist[ij[:, 0], ij[:, 1]]
    # person contacts (< 0.5 m: robot ~0.25 m + person ~0.25 m). Simulated people never yield, so a
    # contact is "robot-caused" only if the robot was closing in on the person (> 0.1 m/s) at that moment.
    ts, ps = t[::5], p[::5]
    vr = np.gradient(ps, ts, axis=0)
    dyn = np.full(len(ts), np.inf)
    people_contacts = robot_contacts = 0
    for a in sc.actors:
        P = np.array([a.state(tt)[0][:2] for tt in ts])
        d = np.linalg.norm(P - ps, axis=1)
        dyn = np.minimum(dyn, d)
        close = d < 0.5
        starts = np.where(close & ~np.concatenate([[False], close[:-1]]))[0]
        for i in starts:
            u = (P[i] - ps[i]) / max(d[i], 1e-6)
            people_contacts += 1
            robot_contacts += int(vr[i] @ u > 0.1)
    plen = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    res_ = dict(scene=scene, seed=seed, method=method, completed=bool(meta["completed"]), fell=bool(meta["fell"]),
                collisions=int(meta["collisions"]), duration=float(meta["duration"]), path_length=plen,
                min_clear_static=float(clr_static.min()), p5_clear_static=float(np.percentile(clr_static, 5)),
                min_clear_people=float(dyn.min()) if np.isfinite(dyn).any() else None,
                people_contacts=people_contacts, robot_caused_contacts=robot_contacts,
                nav_ms=(meta.get("nav") or {}).get("ms_mean"), nav_ms_p95=(meta.get("nav") or {}).get("ms_p95"),
                waypoints_reached=int(rec.wp_idx - 1), n_waypoints=int(len(sc.waypoints) - 1))
    (out / "nav_result.json").write_text(json.dumps(res_, indent=1))
    return res_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nav_clutter")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--vx", type=float, default=0.8)
    ap.add_argument("--t-max", type=float, default=240.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "nav_bench"))
    a = ap.parse_args()
    jobs = [(a.scene, s, m, a.out, a.vx, a.t_max) for s in a.seeds for m in a.methods]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = []
        for r in ex.map(run_one, jobs):
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            res.append(r)
    Path(a.out, f"{a.scene}_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()


def recompute(scene="nav_clutter", root=ROOT / "results" / "nav_bench"):
    """Re-derive the person-contact metrics for every finished run from its gt.npz (same definition
    for all runs, including those recorded before the metric existed)."""
    from sim import scenes as S
    import mujoco
    from sim.record import G1_XML
    out = []
    for f in sorted((Path(root) / scene).glob("s*/*/nav_result.json")):
        r = json.loads(f.read_text())
        g = np.load(f.parent / "gt.npz")
        spec = mujoco.MjSpec.from_file(str(G1_XML))
        sc = S.build(spec, scene, seed=r["seed"])
        ts, ps = g["t"][::5], g["p"][::5, :2]
        vr = np.gradient(ps, ts, axis=0)
        dmin = np.inf
        pc = rc = 0
        for a in sc.actors:
            P = np.array([a.state(tt)[0][:2] for tt in ts])
            d = np.linalg.norm(P - ps, axis=1)
            dmin = min(dmin, float(d.min()))
            close = d < 0.5
            for i in np.where(close & ~np.concatenate([[False], close[:-1]]))[0]:
                u = (P[i] - ps[i]) / max(d[i], 1e-6)
                pc += 1
                rc += int(vr[i] @ u > 0.1)
        r.update(min_clear_people=dmin, people_contacts=pc, robot_caused_contacts=rc)
        f.write_text(json.dumps(r, indent=1))
        out.append(r)
    Path(root, f"{scene}_summary.json").write_text(json.dumps(out, indent=1))
    return out
