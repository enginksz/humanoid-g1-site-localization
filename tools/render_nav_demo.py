#!/usr/bin/env python3
"""Offscreen A/B nav demo → MP4/GIF (Hybrid plan Phase 3).

Renders a tracking camera while the G1 walks blind then with GT costmap.
Label in-frame: SIMULATION · free-space from GT (not a learned model).

  python3 -m tools.render_nav_demo --scene dynamic --seconds 25 --vx 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.record import SIM_DT, Recorder, quat_wxyz_to_R  # noqa: E402


def _pretty(spec):
    sky = spec.add_texture()
    sky.name = "demo_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = [0.55, 0.70, 0.90], [0.92, 0.94, 0.97]
    sky.width, sky.height = 256, 1536
    sun = spec.worldbody.add_light()
    sun.name = "demo_sun"
    sun.pos = [10, 0, 40]
    sun.dir = [0.3, 0.2, -1]
    sun.diffuse = [0.75, 0.72, 0.68]
    sun.castshadow = True
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720


class DemoRecorder(Recorder):
    """Recorder that yields RGB frames at a fixed rate for offscreen capture."""

    def __init__(self, *a, fps=12, width=640, height=360, cam="chase", **kw):
        super().__init__(*a, spec_hook=_pretty, **kw)
        self.fps = fps
        self.width, self.height = width, height
        self.cam_mode = cam
        self.frames: list[np.ndarray] = []
        self._renderer = None
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._every = max(1, int(round(1.0 / (fps * SIM_DT))))
        self._opt = mujoco.MjvOption()
        # Hide overhead slabs/beams (group 5) so the chase cam can see the robot.
        mujoco.mj_forward(self.m, self.d)
        for gid in range(self.m.ngeom):
            z = float(self.d.geom_xpos[gid][2])
            hz = float(self.m.geom_size[gid][2]) if self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else 0.0
            if z - hz > 2.0:          # entirely above ~2 m
                self.m.geom_group[gid] = 5
        self._opt.geomgroup[:] = 1
        self._opt.geomgroup[5] = 0    # overhead off
        self._opt.geomgroup[3] = 1    # dynamic actors on

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, height=self.height, width=self.width)

    def _update_cam(self):
        """Chase / side camera kept under the upper slab (~3.3 m)."""
        p = self.d.xpos[self.pelvis]
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        self._cam.lookat[:] = [float(p[0]), float(p[1]), float(p[2]) + 0.2]
        if self.cam_mode == "top":
            # still under-slab: steep but not above the ceiling
            self._cam.distance = 6.0
            self._cam.elevation = -55.0
            self._cam.azimuth = yaw_deg + 90.0
        else:
            self._cam.distance = 4.0
            self._cam.elevation = -18.0
            self._cam.azimuth = yaw_deg + 195.0   # slightly off-axis behind

    def _on_step(self, t):
        super()._on_step(t)
        step = int(round(t / SIM_DT))
        if step % self._every:
            return
        self._ensure_renderer()
        self._update_cam()
        self._renderer.update_scene(self.d, self._cam, scene_option=self._opt)
        # costmap markers into the scene for a few cells
        if self.costmap is not None and self.costmap.cost is not None:
            scn = self._renderer.scene
            z0 = float(self.d.qpos[2]) - 0.55
            eye = np.eye(3).reshape(-1)
            n0 = scn.ngeom
            for x, y, c in self.costmap.world_cells(max_cost=0.2):
                if scn.ngeom >= scn.maxgeom - 2 or scn.ngeom - n0 > 250:
                    break
                rgba = np.array([0.95, 0.15, 0.15, 0.55] if c > 0.5 else [0.95, 0.75, 0.1, 0.45], np.float32)
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.07, 0, 0]), np.array([x, y, z0]), eye, rgba)
                scn.ngeom += 1
        rgb = self._renderer.render()
        self.frames.append(rgb.copy())


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn HUD text with a simple bitmap-free approach via numpy bars + optional PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([8, 8, 520, 8 + 22 * (len(lines) + 1)], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    y = 14
    for line in lines:
        draw.text((16, y), line, fill=(240, 240, 240, 255), font=font)
        y += 22
    return np.asarray(img)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # try imageio / cv2 / ffmpeg raw
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
        return
    except Exception:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
        return
    except Exception:
        pass
    # fallback: PNG sequence + ffmpeg
    seq = path.with_suffix("") 
    seq.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        from PIL import Image
        Image.fromarray(f).save(seq / f"f{i:05d}.png")
    import subprocess
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(seq / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])


def _write_gif(frames: list[np.ndarray], path: Path, fps: int):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    # subsample for gif size
    step = max(1, fps // 6)
    imgs = [Image.fromarray(f).resize((640, 360)) for f in frames[::step]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / max(fps // step, 1)), loop=0)


def run_segment(scene, costmap, seed, vx, seconds, fps, width, height, cam="chase") -> tuple[list[np.ndarray], dict]:
    rec = DemoRecorder(scene, seed=seed, vx_max=vx, costmap=costmap, recover=True,
                       fps=fps, width=width, height=height, cam=cam)
    tag = "costmap" if costmap else "blind"
    out = ROOT / "results" / "nav_demo" / f"_tmp_{tag}"
    t0 = time.time()
    meta = rec.run(out, t_max=seconds, verbose=False)
    (out / "lidar.bin").unlink(missing_ok=True)
    label_mode = "GT COSTMAP (nav stack)" if costmap else "BLIND FOLLOWER (nav stack)"
    labeled = []
    for i, fr in enumerate(rec.frames):
        t = i / fps
        lines = [
            "SIMULATION · Unitree G1 · localization+nav track",
            f"{label_mode} · vx_max={vx:.1f} m/s",
            "Free-space here is read from the simulator (GT geometry)",
            f"t={t:5.1f}s  collisions={rec.collisions}  falls={len(rec.falls)}",
        ]
        labeled.append(_label(fr, lines))
    stats = dict(
        costmap=costmap, duration=meta["duration"], completed=meta["completed"],
        collisions=rec.collisions, n_falls=len(rec.falls), wall_s=round(time.time() - t0, 1),
        n_frames=len(labeled),
    )
    if rec._renderer is not None:
        rec._renderer.close()
    return labeled, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dynamic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="sim seconds per segment")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--cam", choices=["chase", "top"], default="chase",
                    help="chase=behind+above (default); top=steep overhead")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "nav_demo")
    a = ap.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)

    all_frames = []
    stats = []
    for use_cm in (False, True):
        print(f"rendering {'costmap' if use_cm else 'blind'} …", flush=True)
        frames, st = run_segment(a.scene, use_cm, a.seed, a.vx, a.seconds, a.fps,
                                 a.width, a.height, cam=a.cam)
        stats.append(st)
        all_frames.extend(frames)
        # 0.5 s black separator
        if not use_cm:
            blank = np.zeros_like(frames[0])
            blank = _label(blank, ["NEXT: GT free-space costmap"])
            all_frames.extend([blank] * max(1, a.fps // 2))

    mp4 = a.out / f"{a.scene}_ab_vx{a.vx:.1f}.mp4"
    gif = a.out / f"{a.scene}_ab_vx{a.vx:.1f}.gif"
    _write_mp4(all_frames, mp4, a.fps)
    _write_gif(all_frames, gif, a.fps)
    summary = dict(scene=a.scene, vx=a.vx, seconds=a.seconds, stats=stats,
                   mp4=str(mp4), gif=str(gif), note="SIM · GT free-space (Hybrid Phase 3)")
    (a.out / f"{a.scene}_ab_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
