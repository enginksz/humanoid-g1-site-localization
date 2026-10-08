"""Figures for the report (static PNG, light theme).

  python3 tools/plots.py            -> results/figures/*.png

Color follows the method (fixed order, never cycled); GT is neutral ink.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import scenes as S  # noqa: E402
from sim.record import G1_XML  # noqa: E402
from tools.evaluate import associate, quat_to_R  # noqa: E402
from tools.seqio import read_meta, read_tum  # noqa: E402

FIG = ROOT / "results" / "figures"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# reference categorical palette, fixed slot per method
COLORS = {
    "fmcw3d": "#2a78d6",      # 1 blue
    "fmcw4d": "#eb6834",      # 2 orange
    "legekf": "#1baf7a",      # 3 aqua
    "fmcw3d_leg": "#eda100",  # 4 yellow
    "fmcw4d_leg": "#e87ba4",  # 5 magenta
    "fmcw4d_leg_ng": "#008300",  # 6 green
    "fmcw3d_leg_pg": "#4a3aa7",  # 7 violet
    "fastlio2": "#e34948",       # 8 red
}
LABELS = {
    "fmcw3d": "FMCW-LIO 3D (Doppler off)",
    "fmcw4d": "FMCW-LIO 4D (Doppler)",
    "legekf": "Leg-IMU EKF",
    "fmcw3d_leg": "3D LIO + leg",
    "fmcw4d_leg": "4D + leg + Doppler slip gate",
    "fmcw4d_leg_ng": "4D + leg, no gate",
    "fmcw3d_leg_pg": "3D LIO + leg, prediction gate",
    "fastlio2": "FAST-LIO2 (original)",
}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def aligned_first(est_path, gt_path):
    """Estimate expressed in the GT frame by aligning only the first pose."""
    te, pe, qe = read_tum(est_path)
    tg, pg, qg = read_tum(gt_path)
    ie, ig = associate(te, tg)
    te, pe, qe, pg = te[ie], pe[ie], qe[ie], pg[ig]
    Re, Rg = quat_to_R(qe), quat_to_R(qg[ig])
    R0 = Rg[0] @ Re[0].T
    return te, pe @ R0.T + (pg[0] - R0 @ pe[0]), pg


def scene_footprint(ax, scene):
    """Draw static scene boxes that intersect the LiDAR's height band."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    S.build(spec, scene)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for g in range(m.ngeom):
        if m.geom_group[g] != S.STATIC_GROUP or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        c, h, R = d.geom_xpos[g], m.geom_size[g], d.geom_xmat[g].reshape(3, 3)
        if c[2] - h[2] > 2.5 or c[2] + h[2] < 0.3:
            continue
        corners = np.array([[sx * h[0], sy * h[1], 0] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
        xy = (corners @ R.T + c)[:, :2]
        ax.fill(xy[:, 0], xy[:, 1], color="#d9d8d2", lw=0)
    for sl_patch in read_meta(ROOT / "data" / scene).get("slip_patches", []):
        cx, cy, hx, hy, _ = sl_patch
        ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ls="--", lw=1.0, ec=INK2))


def fig_trajectories(scene, methods):
    seq, res = ROOT / "data" / scene, ROOT / "results" / scene
    fig, ax = plt.subplots(figsize=(9, 5.2))
    scene_footprint(ax, scene)
    _, gp, _ = read_tum(seq / "gt_tum.txt")
    ax.plot(gp[:, 0], gp[:, 1], color=INK, lw=2.5, label="Ground truth", zorder=5)
    for m in methods:
        f = res / m / "poses_tum.txt"
        if not f.exists():
            continue
        _, p, _ = aligned_first(f, seq / "gt_tum.txt")
        ax.plot(p[:, 0], p[:, 1], color=COLORS[m], label=LABELS[m], zorder=6)
    gx, gy = gp[:, 0], gp[:, 1]
    pad = 6
    ax.set_xlim(gx.min() - pad, gx.max() + pad)
    ax.set_ylim(gy.min() - pad, gy.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{scene}: trajectories, first-pose aligned (no loop closure)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / f"traj_{scene}.png", dpi=130)
    plt.close(fig)


def fig_error_time(scene, methods, fname=None, title=None):
    seq, res = ROOT / "data" / scene, ROOT / "results" / scene
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for m in methods:
        f = res / m / "poses_tum.txt"
        if not f.exists():
            continue
        t, p, g = aligned_first(f, seq / "gt_tum.txt")
        ax.plot(t, np.linalg.norm(p - g, axis=1), color=COLORS[m], label=LABELS[m])
    meta = read_meta(seq)
    if meta.get("slip_patches"):
        _, gp, _ = read_tum(seq / "gt_tum.txt")
        tg = np.loadtxt(seq / "gt_tum.txt", usecols=0)
        on = np.zeros(len(tg), bool)
        for cx, cy, hx, hy, _ in meta["slip_patches"]:
            on |= (abs(gp[:, 0] - cx) < hx) & (abs(gp[:, 1] - cy) < hy)
        ax.fill_between(tg, 0, 1, where=on, transform=ax.get_xaxis_transform(), color="#ecebe6", lw=0,
                        label="on sliding sheet")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title(title or f"{scene}: position error over time (first-pose aligned)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / (fname or f"err_{scene}.png"), dpi=130)
    plt.close(fig)


def fig_summary(summary, scenes, methods):
    """ATE per scene; uses the multi-seed aggregate (mean, std whiskers) when present."""
    agg_f = ROOT / "results" / "aggregate.json"
    agg = json.loads(agg_f.read_text()) if agg_f.exists() else None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(methods)
    x = np.arange(len(scenes))
    for k, m in enumerate(methods):
        if agg:
            vals = [agg.get(s, {}).get(m, {}).get("ate_mean", np.nan) for s in scenes]
            err = [agg.get(s, {}).get(m, {}).get("ate_std", 0.0) for s in scenes]
        else:
            vals = [summary.get(s, {}).get(m, {}).get("ate_rmse", np.nan) for s in scenes]
            err = None
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, width=w * 0.9, color=COLORS[m], label=LABELS[m],
               yerr=err, error_kw=dict(ecolor=INK2, lw=1, capsize=0))
    ax.set_yscale("log")
    ax.set_xticks(x, scenes)
    ax.set_ylabel("ATE RMSE [m] (log)")
    n = next(iter(next(iter(agg.values())).values()))["n"] if agg else 1
    ax.set_title(f"Absolute trajectory error by scene (SE3-aligned, mean ± std over {n} seeds)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "summary_ate.png", dpi=130)
    plt.close(fig)


def fig_slip_gate(scene):
    """Leg-vs-Doppler Mahalanobis distance against GT sheet contact."""
    seq, res = ROOT / "data" / scene, ROOT / "results" / scene
    f = res / "fmcw4d_leg" / "leg_updates.csv"
    if not f.exists():
        return
    # rows without stance samples are short (no gate_src column); keep stance rows only
    rows = [l.split(",") for l in f.read_text().splitlines()[1:]]
    rows = np.array([[float(x) for x in r] for r in rows if len(r) == 8 and float(r[1]) > 0])
    names = ["t", "n", "m2", "vbx", "vby", "vbz", "accepted", "gate_src"]
    a = {k: rows[:, i] for i, k in enumerate(names)}
    meta = read_meta(seq)
    gt = np.load(seq / "gt.npz")
    on = np.zeros(len(gt["t"]), bool)
    for cx, cy, hx, hy, _ in meta["slip_patches"]:
        on |= (abs(gt["p"][:, 0] - cx) < hx) & (abs(gt["p"][:, 1] - cy) < hy)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(gt["t"], 0, 1, where=on, transform=ax.get_xaxis_transform(), color="#ecebe6", lw=0,
                    label="robot on sliding sheet (GT)")
    acc = a["accepted"] > 0
    ax.scatter(a["t"][acc], np.clip(a["m2"][acc], 1e-2, 1e4), s=10, color=COLORS["fmcw4d_leg"], label="leg update accepted")
    ax.scatter(a["t"][~acc], np.clip(a["m2"][~acc], 1e-2, 1e4), s=14, color=INK, marker="x", label="rejected (slip)")
    ax.axhline(11.34, color=INK2, lw=1, ls="--")
    ax.text(a["t"].max(), 11.34 * 1.3, "chi2 99% gate", ha="right", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("leg vs Doppler  d²")
    ax.set_title(f"{scene}: Doppler slip gate", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / f"slipgate_{scene}.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    scenes = [s for s in ["slab", "dynamic", "sliding", "corridor", "corridor_boards", "roof", "roof_boards", "outdoor"] if s in summary]
    main_methods = ["fastlio2", "fmcw3d", "fmcw4d", "fmcw3d_leg", "fmcw4d_leg"]
    fig_summary(summary, scenes, main_methods)
    for s in scenes:
        fig_trajectories(s, main_methods)
        fig_error_time(s, main_methods)
        if read_meta(ROOT / "data" / s).get("slip_patches"):
            fig_slip_gate(s)
            fig_error_time(s, ["fmcw3d_leg", "fmcw4d_leg_ng", "fmcw4d_leg"], fname=f"err_gate_{s}.png",
                           title=f"{s}: effect of the slip gate")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
