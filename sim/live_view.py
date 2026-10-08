"""Watch a closed-loop run live: MuJoCo viewer (+ optional ROS 2 / RViz2 stream).

The robot walks on its OWN estimate (see sim/closed_loop.py). Drawn in the viewer:
  green  : ground-truth trail          orange : estimated trail (what the robot believes)
  points : last LiDAR scan, coloured by Doppler (red = approaching, blue = receding,
           grey = static after ego-motion; i.e. moving workers light up)
Top-left text: estimator, time, true error, filter sigma, map mode, collisions.

  ./scripts/live.sh roof 3d                      # 3D LIO only, watch it drift
  ./scripts/live.sh roof 3d --map results/maploc/map_roof.npz
  ./scripts/live.sh dynamic 4d_leg --ros         # + RViz2 (configs/live.rviz)

Keys in the viewer: space pauses (MuJoCo default), close the window to stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from . import scenes as S
from .closed_loop import ESTIMATORS, ClosedLoopRecorder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREEN = np.array([0.10, 0.75, 0.30, 1.0], np.float32)
ORANGE = np.array([0.95, 0.45, 0.10, 1.0], np.float32)
N_POINTS = 800
EYE = np.eye(3).reshape(-1)
SPHERE, CAPSULE = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE
RGBA_APPROACH = np.array([0.9, 0.15, 0.1, 1], np.float32)
RGBA_RECEDE = np.array([0.15, 0.35, 0.95, 1], np.float32)
RGBA_STATIC = np.array([0.35, 0.35, 0.35, 0.6], np.float32)
RGBA_COST = np.array([0.95, 0.15, 0.15, 0.55], np.float32)
RGBA_COST_SOFT = np.array([0.95, 0.75, 0.10, 0.45], np.float32)
SZ_MOVING, SZ_STATIC = np.array([0.05, 0, 0]), np.array([0.025, 0, 0])
SZ_COST = np.array([0.06, 0, 0])
ZERO3 = np.zeros(3)


class LiveRecorder(ClosedLoopRecorder):
    def __init__(self, *a, speed=1.0, ros=None, **kw):
        super().__init__(*a, **kw)
        self.speed = speed
        self.viewer = None
        self.ros = ros
        self.gt_trail, self.est_trail = [], []
        self.scan_world = np.zeros((0, 4))
        self._disp_buf = []
        self.t_wall0 = None
        self.paused_t = 0.0

    # ---- data taps
    def _cast_columns(self, cols, t_off):
        out = super()._cast_columns(cols, t_off)
        if len(out):
            # display copy, computed with the sensor pose/velocity at the firing instant:
            # world position, and Doppler minus the ego-motion part = the target's own radial
            # speed (static ground stays grey even while the robot tumbles; workers light up)
            sid = self.sid_lidar
            R, p = self.d.site_xmat[sid].reshape(3, 3), self.d.site_xpos[sid]
            v_w, _ = self._site_vel(sid)
            dirs = out[:, :3] / np.linalg.norm(out[:, :3], axis=1, keepdims=True)
            own = out[:, 3] + dirs @ (R.T @ v_w)
            self._disp_buf.append(np.c_[out[:, :3] @ R.T + p, own])
        return out

    def _on_scan(self, t0, pts):
        super()._on_scan(t0, pts)
        sid = self.sid_lidar
        R, p = self.d.site_xmat[sid].reshape(3, 3), self.d.site_xpos[sid]
        if self._disp_buf:
            W = np.concatenate(self._disp_buf)
            self._disp_buf = []
            self.scan_world = W if len(W) <= N_POINTS else W[np.random.default_rng(0).choice(len(W), N_POINTS, replace=False)]
        if self.ros is not None:
            est = self.last_est["Tw"] if self.last_est is not None else None
            T_gt = np.eye(4)
            T_gt[:3, :3] = self.d.site_xmat[self.sid_imu].reshape(3, 3)
            T_gt[:3, 3] = self.d.site_xpos[self.sid_imu]
            self.ros.publish(t0, T_gt, est, np.c_[pts[:, :3] @ R.T + p, pts[:, 3]],
                             costmap=self.costmap)

    def _on_step(self, t):
        super()._on_step(t)
        step = int(round(t / 0.002))
        if step % 20 or self.viewer is None:         # 25 Hz display
            return
        if not self.viewer.is_running():
            raise KeyboardInterrupt
        if step % 50 == 0:
            self.gt_trail.append(self.d.site_xpos[self.sid_imu] - [0, 0, 0.6])
            if self.last_est is not None:
                self.est_trail.append(self.last_est["Tw"][:3, 3] - [0, 0, 0.6])
        self._draw(t)
        # real-time pacing
        if self.t_wall0 is None:
            self.t_wall0 = time.time() - t / self.speed
        lag = (self.t_wall0 + t / self.speed) - time.time()
        if lag > 0:
            time.sleep(lag)

    # ---- drawing
    def _draw(self, t):
        v = self.viewer
        with v.lock():
            scn = v.user_scn
            scn.ngeom = 0
            cap = scn.maxgeom - 5

            def add(gtype, size, pos, rgba):
                if scn.ngeom >= cap:
                    return
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, size, pos, EYE, rgba)
                scn.ngeom += 1

            def line(a, b, rgba, w=0.035):
                if scn.ngeom >= cap:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, CAPSULE, ZERO3, ZERO3, EYE, rgba)
                mujoco.mjv_connector(g, CAPSULE, w, a, b)
                scn.ngeom += 1

            for trail, col in ((self.gt_trail, GREEN), (self.est_trail, ORANGE)):
                for a, b in zip(trail[:-1], trail[1:]):
                    line(a, b, col)
            for x, y, z, dop in self.scan_world:
                if abs(dop) > 0.25:              # moving w.r.t. the world (after ego-motion)
                    add(SPHERE, SZ_MOVING, np.array((x, y, z)), RGBA_APPROACH if dop < 0 else RGBA_RECEDE)
                else:
                    add(SPHERE, SZ_STATIC, np.array((x, y, z)), RGBA_STATIC)
            # GT costmap overlay (Hybrid: ground-truth free-space)
            if self.costmap is not None:
                z0 = float(self.d.qpos[2]) - 0.55
                n_drawn = 0
                for x, y, c in self.costmap.world_cells(max_cost=0.15):
                    if n_drawn >= 400:
                        break
                    rgba = RGBA_COST if c > 0.5 else RGBA_COST_SOFT
                    add(SPHERE, SZ_COST, np.array((x, y, z0)), rgba)
                    n_drawn += 1
        err = sig = np.nan
        mode = "-"
        if self.last_est is not None:
            err = np.linalg.norm(self.last_est["Tw"][:2, 3] - self.d.site_xpos[self.sid_imu][:2])
            sig = self.last_est["sigma_pos"]
        if self.loc is not None:
            mode = self.loc.mode
        cost_s = f"{self.last_cost:.2f}" if self.costmap is not None else "-"
        left = "estimator\nsim time\ntrue xy error\nfilter sigma\nmap localizer\ncollisions\nvx cmd\nGT cost"
        right = (f"{self.est_kind}{' + map' if self.loc is not None else ''}\n{t:6.1f} s\n{err:6.2f} m\n"
                 f"{sig * 100:6.1f} cm\n{mode}\n{self.collisions}\n{self.cmd[0]:5.2f} m/s\n{cost_s}")
        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
        v.sync()


def _pretty(spec):
    """Daylight + sky for the viewer only (LiDAR/IMU are unaffected)."""
    sky = spec.add_texture()
    sky.name = "live_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = [0.55, 0.70, 0.90], [0.92, 0.94, 0.97]
    sky.width, sky.height = 256, 1536
    sun = spec.worldbody.add_light()
    sun.name = "live_sun"
    sun.pos = [10, 0, 40]
    sun.dir = [0.3, 0.2, -1]
    sun.diffuse = [0.75, 0.72, 0.68]
    sun.castshadow = True
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dynamic", choices=S.SCENES)
    ap.add_argument("--est", default="4d_leg", choices=list(ESTIMATORS))
    ap.add_argument("--map", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor")
    ap.add_argument("--ros", action="store_true", help="also publish to ROS 2 (source /opt/ros/humble first)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--vx", type=float, default=1.0, help="max walking speed command [m/s]")
    ap.add_argument("--no-recover", dest="recover", action="store_false",
                    help="do not stand the robot back up after a fall")
    ap.add_argument("--costmap", action="store_true",
                    help="GT free-space costmap steering + overlay")
    ap.add_argument("--hold", type=int, default=1, help="keep the window open at the end")
    a = ap.parse_args()
    torch.set_num_threads(1)
    ros = None
    if a.ros:
        from tools.ros_live import RosLive
        ros = RosLive(map_path=a.map)
    out = Path(a.out or ROOT / "results/live" / f"{a.scene}_{a.est}{'_map' if a.map else ''}{'_cm' if a.costmap else ''}")
    rec = LiveRecorder(a.scene, est=a.est, map_path=a.map, speed=a.speed, ros=ros, spec_hook=_pretty,
                       vx_max=a.vx, recover=a.recover, costmap=a.costmap)
    rec.start_estimator(out)
    with mujoco.viewer.launch_passive(rec.m, rec.d, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1          # moving workers / machines (hidden by default)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(rec.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 9.0, -35.0, 200.0
        rec.viewer = viewer
        try:
            rec.run(out, t_max=a.t_max, verbose=False)
        except KeyboardInterrupt:
            pass
        rec.est.close()
        (out / "lidar.bin").unlink(missing_ok=True)
        print(f"done: sim {rec.d.time:.1f} s, collisions={rec.collisions}", flush=True)
        while a.hold and viewer.is_running():   # keep the final picture until the window is closed
            time.sleep(0.1)
        viewer.close()
    if ros is not None:
        ros.shutdown()
        # a DDS/rclpy background thread keeps the interpreter alive at exit; leave hard
        sys.stdout.flush()
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
