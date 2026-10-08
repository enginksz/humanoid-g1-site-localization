"""Figures for the mapping-for-navigation study (tools/elevmap.py, tools/nav_bench.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "results" / "figures"

LABEL = {
    "gt__none": "GT pose, accumulate only",
    "gt__raycast": "GT pose, raycast clearing",
    "gt__doppler": "GT pose, Doppler removal",
    "gt__doppler+raycast": "GT pose, Doppler + raycast",
    "fastlio2__raycast": "FAST-LIO2 pose, raycast clearing\n(classic LiDAR stack)",
    "fmcw3d__raycast": "FMCW-LIO 3D pose, raycast",
    "fmcw4d__doppler": "FMCW-LIO 4D pose, Doppler",
    "fmcw4d_leg__doppler": "4D + legs pose, Doppler\n(ours)",
    "fmcw4d_leg__doppler+raycast": "4D + legs, Doppler + raycast",
    "fmcw4d_leg__doppler+persist": "FMCW-LIO 4D + legs pose,\nDoppler + persistence (ours)",
    "gt__doppler+persist": "GT pose, Doppler + persistence",
}


def _load(scene, name):
    z = np.load(ROOT / "results" / scene / "elevmap" / f"map_{name}.npz")
    return {k: z[k] for k in z.files}


def map_panels(scene, names, out, crop=None, truth="oracle"):
    ref = _load(scene, truth)
    n = len(names)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    gt = np.load(ROOT / "data" / scene / "gt.npz")["p"]
    for ax, nm in zip(np.atleast_1d(axs), names):
        m = _load(scene, nm)
        lo, res = m["lo"], float(m["res"])
        from scipy.ndimage import distance_transform_edt
        obs, ob, rob = m["observed"], m["obstacle"], ref["obstacle"]
        both = obs & ref["observed"]
        dR = distance_transform_edt(~(rob & both)) * res
        dM = distance_transform_edt(~(ob & both)) * res
        img = np.ones(obs.shape + (3,))
        img[obs] = [0.93, 0.93, 0.91]
        img[ob] = [0.25, 0.25, 0.28]                                   # obstacle (within 0.3 m of truth)
        img[ob & both & (dR > 0.3)] = [0.85, 0.15, 0.15]               # ghost / false obstacle
        img[rob & both & (dM > 0.3)] = [0.15, 0.45, 0.85]              # missed obstacle
        ext = [lo[0], lo[0] + obs.shape[0] * res, lo[1], lo[1] + obs.shape[1] * res]
        ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest")
        ax.plot(gt[:, 0], gt[:, 1], color="#1BAF7A", lw=1.0)
        if crop:
            ax.set_xlim(crop[0], crop[1])
            ax.set_ylim(crop[2], crop[3])
        ax.set_title(LABEL.get(nm, nm), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for c, l in
                        [([0.25, 0.25, 0.28], "obstacle (correct)"), ([0.85, 0.15, 0.15], "ghost / false obstacle"),
                         ([0.15, 0.45, 0.85], "missed obstacle"), ("#1BAF7A", "robot path")]],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / out, dpi=140)
    plt.close(fig)


def bench_traj(scene="nav_clutter", seed=0, out="nav_bench_traj.png"):
    from tools.nav_bench import static_raster
    occ, org, res, sc = static_raster(scene, seed)
    base = ROOT / "results" / "nav_bench" / scene / f"s{seed}"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ext = [org[0], org[0] + occ.shape[0] * res, org[1], org[1] + occ.shape[1] * res]
    ax.imshow(np.where(occ, 0.25, 1.0).T, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    cols = dict(blind="#999999", reactive_gt="#EB6834", mppi_none="#9467bd", mppi_raycast="#2A78D6",
                mppi_doppler_raycast="#1BAF7A")
    for mth, c in cols.items():
        f = base / mth / "gt.npz"
        if not f.exists():
            continue
        p = np.load(f)["p"]
        r = json.loads((base / mth / "nav_result.json").read_text())
        names = dict(blind="blind follower", reactive_gt="previous: reactive on GT costmap",
                     mppi_none="LiDAR map, no dynamic handling", mppi_raycast="LiDAR map + raycast",
                     mppi_doppler_raycast="LiDAR map + Doppler + raycast (ours)")
        tag = f"{names.get(mth, mth)}: {'done' if r['completed'] else 'stuck'} ({r['duration']:.0f} s), {r['collisions']} coll."
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.6, label=tag)
    w = sc.waypoints
    ax.plot(w[:, 0], w[:, 1], "k--", lw=0.8, label="mission waypoints")
    for k, a in enumerate(sc.actors):
        ax.plot(a.path[:, 0], a.path[:, 1], ":", color="#d62728", lw=1.2, label="people's walking lines" if k == 0 else None)
    ax.set_xlim(w[:, 0].min() - 3, w[:, 0].max() + 3)
    ax.set_ylim(w[:, 1].min() - 4, w[:, 1].max() + 4)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("maps", "all"):
        map_panels("dynamic", ["gt__none", "gt__raycast", "gt__doppler"], "elevmap_dynamic_dyn.png",
                   crop=(-4, 32, -11, 13))
        map_panels("dynamic", ["fastlio2__raycast", "fmcw4d_leg__doppler+persist"], "elevmap_dynamic_stack.png",
                   crop=(-4, 32, -11, 13))
        map_panels("outdoor", ["fastlio2__raycast", "fmcw4d_leg__doppler+persist"], "elevmap_outdoor_stack.png")
    if what in ("bench", "all"):
        bench_traj()
