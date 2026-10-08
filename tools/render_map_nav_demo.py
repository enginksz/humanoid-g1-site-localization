#!/usr/bin/env python3
"""Demo video: previous reactive follower vs. LiDAR map + Doppler + navigation function + MPPI.

Left: chase camera (MuJoCo). Right: what the robot's own navigation stack sees, from its own
sensors only - the online map (obstacles / observed ground), Doppler-flagged moving points,
person/vehicle tracks with predicted motion, MPPI samples, the selected trajectory and the
navigation-function path to the next waypoint.

  python3 tools/render_map_nav_demo.py --scene nav_clutter --seed 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.nav import extract_path  # noqa: E402
from tools.render_nav_demo import DemoRecorder, _label  # noqa: E402

W = 640          # each panel is W x W


class MapDemoRecorder(DemoRecorder):
    def __init__(self, *a, writer=None, header=(), static_raster=None, **kw):
        super().__init__(*a, **kw)
        self.writer = writer
        self.header = list(header)
        self.static = static_raster
        self.trail = []
        self.fig, self.ax = plt.subplots(figsize=(W / 100, W / 100), dpi=100)
        self.fig.subplots_adjust(0, 0, 1, 1)

    def _right_panel(self, t):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#f4f4f2")
        p = self.d.qpos[:2].copy()
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.trail.append(p)
        half = 8.0
        if self.nav is not None and self.nav.last:
            L = self.nav.last
            w = L["win"]
            o, res = w["origin"], w["res"]
            n = w["obstacle"].shape
            ext = [o[0], o[0] + n[0] * res, o[1], o[1] + n[1] * res]
            img = np.ones(n + (3,))
            img[w["observed"]] = [0.90, 0.90, 0.88]
            infl = np.clip(L["cost"], 0, 1)
            img = img * (1 - 0.35 * infl[..., None]) + 0.35 * infl[..., None] * np.array([1.0, 0.85, 0.55])
            img[w["obstacle"]] = [0.22, 0.22, 0.25]
            ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest")
            dp = self.nav.mapper.last_dyn_pts
            if len(dp):
                ax.scatter(dp[::3, 0], dp[::3, 1], s=3, c="#e3342f", lw=0, zorder=4)
            if L.get("rollouts") is not None:
                for r_ in L["rollouts"][:16]:
                    ax.plot(r_[:, 0], r_[:, 1], color="#5aa9e6", lw=0.6, alpha=0.6, zorder=5)
            if L.get("best") is not None:
                ax.plot(L["best"][:, 0], L["best"][:, 1], color="#f28c28", lw=2.2, zorder=6)
            path = extract_path(L["D"], L["d_origin"], L["d_res"], p, n_max=120)
            if len(path) > 1:
                ax.plot(path[:, 0], path[:, 1], "--", color="#1BAF7A", lw=1.6, zorder=5)
            C, rr = L["tracks"]
            for c_, r_ in zip(C, rr):
                ax.add_patch(plt.Circle(c_[0], r_, fill=False, color="#e3342f", lw=1.5, zorder=7))
                ax.annotate("", xy=c_[min(9, len(c_) - 1)], xytext=c_[0],
                            arrowprops=dict(arrowstyle="->", color="#e3342f", lw=1.4), zorder=7)
            title = "robot's own map (LiDAR + Doppler)  ·  MPPI"
        elif self.static is not None:
            occ, org, res = self.static
            ext = [org[0], org[0] + occ.shape[0] * res, org[1], org[1] + occ.shape[1] * res]
            ax.imshow(np.where(occ, 0.25, 0.95).T, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
            title = "ground-truth costmap (sim geometry)  ·  reactive steering"
        else:
            title = ""
        tr = np.array(self.trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#1BAF7A", lw=1.2, alpha=0.8, zorder=3)
        wp = self.scene.waypoints
        goal = wp[min(self.wp_idx, len(wp) - 1)]
        ax.plot(*goal, marker="*", ms=16, color="#7b3fe4", zorder=8)
        ax.add_patch(plt.Polygon([p + 0.45 * np.array([np.cos(yaw), np.sin(yaw)]),
                                  p + 0.28 * np.array([np.cos(yaw + 2.4), np.sin(yaw + 2.4)]),
                                  p + 0.28 * np.array([np.cos(yaw - 2.4), np.sin(yaw - 2.4)])],
                                 color="#111111", zorder=9))
        ax.set_xlim(p[0] - half, p[0] + half)
        ax.set_ylim(p[1] - half, p[1] + half)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, lw=0))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _on_step(self, t):
        n0 = len(self.frames)
        super()._on_step(t)                    # renders the camera frame into self.frames
        if len(self.frames) == n0:
            return
        cam = self.frames.pop()
        lines = self.header + [f"t = {t:5.1f} s   waypoint {min(self.wp_idx, len(self.scene.waypoints) - 1)}"
                               f"/{len(self.scene.waypoints) - 1}   collisions {self.collisions}"]
        cam = _label(cam, lines)
        frame = np.concatenate([cam, self._right_panel(t)], axis=1)
        self.writer.stdin.write(frame.astype(np.uint8).tobytes())


def open_writer(path, fps):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{2 * W}x{W}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)


def card(writer, lines, seconds, fps):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (2 * W, W), (24, 26, 30))
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    y = W // 2 - 30 * len(lines)
    for i, ln in enumerate(lines):
        d.text((70, y), ln, fill=(240, 240, 240) if i else (120, 200, 160), font=f1 if i == 0 else f2)
        y += 56 if i == 0 else 38
    fr = np.asarray(img).tobytes()
    for _ in range(int(seconds * fps)):
        writer.stdin.write(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nav_clutter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vx", type=float, default=0.8)
    ap.add_argument("--fps", type=int, default=10, help="capture rate (sim time)")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed-up")
    ap.add_argument("--old-seconds", type=float, default=60.0)
    ap.add_argument("--new-seconds", type=float, default=260.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "nav_demo" / "map_nav_demo.mp4"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    play = int(round(a.fps * a.speed))
    wr = open_writer(out, play)
    from tools.nav_bench import static_raster
    occ, org, res, _ = static_raster(a.scene, a.seed)

    card(wr, ["Humanoid navigation on its own LiDAR map",
              "Unitree G1 (pretrained walking policy) in simulation  ·  head FMCW LiDAR with Doppler",
              "Mission: U-trap, 1 m gap, pallets, people crossing and walking head-on",
              f"Video plays at {a.speed:.0f}x"], 4, play)
    card(wr, ["1 / 2   Previous: reactive steering",
              "Heading fan on a GROUND-TRUTH costmap (read from the simulator)",
              "No global plan -> trapped in the U-shaped formwork"], 3, play)
    rec = MapDemoRecorder(a.scene, seed=a.seed, vx_max=a.vx, costmap=True, fps=a.fps, width=W, height=W,
                          writer=wr, static_raster=(occ, org, res),
                          header=["SIMULATION  ·  previous method", "reactive heading fan on GT costmap"])
    m1 = rec.run(out.parent / "_tmp_old", t_max=a.old_seconds, verbose=False)
    card(wr, ["2 / 2   This work: map + Doppler + navigation function + MPPI",
              "Map built ONLY from the robot's LiDAR (no simulator geometry)",
              "Doppler removes walking people from the map and tracks them",
              "Global: Dijkstra navigation function (no local minima)  ·  Local: MPPI on (vx, wz)"], 4, play)
    rec2 = MapDemoRecorder(a.scene, seed=a.seed, vx_max=a.vx, nav="mppi", nav_dyn="doppler+raycast",
                           fps=a.fps, width=W, height=W, writer=wr,
                           header=["SIMULATION  ·  this work", "own LiDAR map + Doppler + nav. function + MPPI"])
    m2 = rec2.run(out.parent / "_tmp_new", t_max=a.new_seconds, verbose=False)
    card(wr, ["Result (this seed)",
              f"previous: {'completed' if m1['completed'] else 'stuck'} after {m1['duration']:.0f} s, "
              f"{m1['collisions']} collisions",
              f"this work: {'completed' if m2['completed'] else 'not completed'} in {m2['duration']:.0f} s, "
              f"{m2['collisions']} collisions",
              "Benchmark over 5 seeds and all methods: see the report"], 5, play)
    wr.stdin.close()
    wr.wait()
    for d in ("_tmp_old", "_tmp_new"):
        (out.parent / d / "lidar.bin").unlink(missing_ok=True)
    print(out, m1["completed"], m2["completed"], m2["duration"])


if __name__ == "__main__":
    main()
