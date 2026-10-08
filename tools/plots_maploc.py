"""Figures for prior-map localization and closed-loop navigation -> results/figures/"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plots import FIG, INK, INK2, scene_footprint  # noqa: E402  (also sets the style)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed slot order


def log(name):
    return np.genfromtxt(ROOT / "results/maploc/runs" / name / "maploc_log.csv", delimiter=",", names=True)


def fig_days():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for i, (n, lab) in enumerate([("track_slab", "day 0 (map)"), ("track_slab_day1", "day 1"),
                                  ("track_slab_day7", "day 7")]):
        a = log(n)
        a1.plot(a["t"], a["err_xy"] * 100, color=C[i], label=lab, lw=1.5)
        a2.plot(a["t"], a["fitness"], color=C[i], label=lab, lw=1.5)
    a1.set_ylabel("xy error [cm]")
    a2.set_ylabel("ICP inlier ratio")
    a2.set_xlabel("time [s]")
    a1.set_title("Localization in the day-0 map while the site changes", loc="left")
    a1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "maploc_days.png", dpi=130)
    plt.close(fig)


def fig_glitch(name="glitch_30_3m_0deg", t_g=30.0):
    a = log(name)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(a["t"], np.maximum(a["err_xy"], 1e-3), color=C[0], lw=1.5, label="map-localizer xy error")
    lost = a["t"][a["lost"] > 0]
    hand = a["t"][a["handover"] > 0]
    ax.axvline(t_g, color=INK2, ls="--", lw=1)
    ax.text(t_g + 1, 0.95, "odometry glitch (+3 m)", color=INK2, fontsize=9, va="top", transform=ax.get_xaxis_transform())
    for t in lost:
        ax.axvline(t, color=INK, lw=1)
    for t in hand:
        ax.axvline(t, color=C[2], lw=1.5)
    ax.plot([], [], color=INK, lw=1, label="declared lost")
    ax.plot([], [], color=C[2], lw=1.5, label="relocalized (handover)")
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("xy error [m]")
    ax.set_title("Silent odometry failure: detection and relocalization", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "maploc_glitch.png", dpi=130)
    plt.close(fig)


def fig_closed_loop(scene, runs, fname):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    scene_footprint(ax, scene if scene != "slab_day7" else "slab")
    from sim import scenes as S
    import mujoco
    from sim.record import G1_XML
    sp = mujoco.MjSpec.from_file(str(G1_XML))
    route = S.build(sp, scene).waypoints
    ax.plot(route[:, 0], route[:, 1], color=INK, lw=1, ls="--", label="planned route")
    for i, (run, lab) in enumerate(runs):
        g = np.load(ROOT / "results/closed_loop" / run / "gt.npz")
        ax.plot(g["p"][:, 0], g["p"][:, 1], color=C[i], lw=2, label=lab)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{scene}: where the robot actually walked, navigating on its own estimate", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_days()
    fig_glitch()
    fig_closed_loop("roof", [("roof_3d", "3D LIO"), ("roof_4d", "4D (Doppler)"), ("roof_4d_leg", "4D + leg"),
                             ("roof_3d_map", "3D LIO + prior map")], "closed_loop_roof.png")
    fig_closed_loop("dynamic", [("dynamic_3d", "3D LIO (103 collisions)"), ("dynamic_4d_leg", "4D + leg"),
                                ("dynamic_3d_map", "3D LIO + prior map")], "closed_loop_dynamic.png")
    print("ok")
