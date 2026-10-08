"""Closed-loop navigation: the G1 walks its route using its OWN state estimate.

FMCW-LIO runs as a live process (tools/online_est.py) fed with the simulated
IMU, leg velocity and LiDAR as they are produced; the waypoint follower steers
with the estimate. The estimate is anchored once at start-up to the robot's
known start pose (docking / survey marker), after that ground truth is never used.

Health policy ("stop"): if the estimator's own uncertainty grows past a limit,
the robot stops (fail-safe) instead of walking on with a bad pose.

  python3 -m sim.closed_loop --scene corridor --est 4d_leg --out results/closed_loop/corridor_4d_leg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import scenes as S
from .record import IMU_POS, LIDAR_PITCH, LIDAR_POS, Recorder, quat_wxyz_to_R, rot_y

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.legodo import LegVelocityOnline  # noqa: E402
from tools.online_est import OnlineEstimator  # noqa: E402
from tools import maploc as ML  # noqa: E402

ESTIMATORS = {
    "3d": dict(doppler=False, leg=False),
    "4d": dict(doppler=True, leg=False),
    "3d_leg": dict(doppler=False, leg=True, extra=("leg/chi2_gate=1e9",)),
    "4d_leg": dict(doppler=True, leg=True),
    # geometry-only LiDAR + legs, with a realistic (not over-confident) point noise so the
    # leg velocity can actually correct the weakly observed axis (no Doppler needed)
    "3d_leg_cov": dict(doppler=False, leg=True, extra=("leg/chi2_gate=1e9", "update/point_cov=0.05")),
}


def cross_track(p, route):
    """Distance from points p (N,2) to the route polyline."""
    best = np.full(len(p), np.inf)
    for a, b in zip(route[:-1], route[1:]):
        ab = b - a
        u = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - (a + u[:, None] * ab), axis=1))
    return best


class ClosedLoopRecorder(Recorder):
    def __init__(self, scene, est="4d_leg", health="none", sigma_pos_max=0.5, sigma_yaw_max=5.0, map_path=None, **kw):
        super().__init__(scene, **kw)
        # optional prior-map localizer between the LIO and the follower
        self.loc = ML.MapLocalizer(ML.PriorMap(map_path)) if map_path else None
        self.scans = {}
        self.T_odom_prev = None
        self.map_log = []
        self.est_kind = est
        self.health = health
        self.sigma_pos_max, self.sigma_yaw_max = sigma_pos_max, sigma_yaw_max
        self.lvo = LegVelocityOnline()
        self.est = None
        self.anchor = None
        self.last_est = None
        self.est_log = []
        self.stopped_at = None
        self.collisions = 0
        self._in_collision = False
        self._nav_scans = {}
        self._Tw_prev = None
        # robot_bodies / floor already set on Recorder

    def start_estimator(self, out_dir):
        meta = dict(R_bl=rot_y(LIDAR_PITCH).reshape(-1).tolist(), p_bl_b=(LIDAR_POS - IMU_POS).tolist())
        cfg = ESTIMATORS[self.est_kind]
        self.est = OnlineEstimator(meta, Path(out_dir) / "estimator", cfg["doppler"], cfg["leg"], cfg.get("extra", ()))

    # ---- hooks
    def _on_imu(self, t, imu_row, q, dq, F):
        self.est.imu(t, imu_row[1:4], imu_row[4:7])
        v, sig, n = self.lvo.update(t, imu_row[1:4], q, dq, F)
        self.est.leg(t, v, sig, n)

    def _on_scan(self, t0, pts):
        self.est.scan(t0, pts)
        if self.loc is not None:
            self.scans[round(t0 + 0.098, 3)] = pts
            for k in [k for k in self.scans if k < t0 - 1.0]:
                del self.scans[k]
        for e in self.est.poll():
            self._take(e)

    def _take(self, e):
        if self.anchor is None:
            # one-time anchoring to the known start pose (the only GT use)
            sid = self.sid_imu
            T_gt = np.eye(4)
            T_gt[:3, :3] = self.d.site_xmat[sid].reshape(3, 3)
            T_gt[:3, 3] = self.d.site_xpos[sid]
            self.anchor = T_gt @ np.linalg.inv(e["T"])
        Tw = self.anchor @ e["T"]
        if self.loc is not None:
            Tw = self._map_update(e, Tw)
            if Tw is None:
                return
        self._nav_map_scan(e["t"], Tw)
        self.last_est = dict(e, Tw=Tw)
        self.est_log.append([e["t"], *Tw[:3, 3], np.arctan2(Tw[1, 0], Tw[0, 0]), e["sigma_pos"], e["sigma_yaw_deg"]])

    def _map_update(self, e, Tw_odom):
        """Prior-map localization on top of the live LIO (corrects drift, monitors health)."""
        key = min(self.scans, key=lambda k: abs(k - e["t"])) if self.scans else None
        if key is None or abs(key - e["t"]) > 0.03:
            return self.loc.T if self.loc.mode == ML.TRACK and self.T_odom_prev is not None else Tw_odom
        pts = self.scans.pop(key)
        T_odom = e["T"]
        if self.T_odom_prev is None:
            self.loc.T, self.loc.mode = Tw_odom, ML.TRACK         # known start pose
            dT = np.eye(4)
        else:
            dT = np.linalg.inv(self.T_odom_prev) @ T_odom
        self.T_odom_prev = T_odom
        R_tilt = ML.rz(-ML.yaw_of(T_odom[:3, :3])) @ T_odom[:3, :3]
        sc = dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], t_off=pts[:, 4])
        R_bl, p_bl = rot_y(LIDAR_PITCH), LIDAR_POS - IMU_POS
        info = self.loc.step(ML.deskew(sc, R_bl, p_bl, dT), dT, R_tilt, T_odom)
        self.map_log.append([e["t"], 1 if self.loc.mode == ML.TRACK else 0, info.get("fitness", np.nan),
                             int(info.get("lost", False))])
        if self.loc.mode != ML.TRACK:
            return None                    # lost: no pose -> the follower stands still
        return self.loc.T.copy()

    # ---- navigation map on the ESTIMATED pose (no ground truth)
    def _nav_scan(self, t0, pts):
        if self.nav is not None:
            self._nav_scans[round(t0 + 0.098, 3)] = pts
            for k in [k for k in self._nav_scans if k < t0 - 1.0]:
                del self._nav_scans[k]

    def _nav_map_scan(self, t_end, Tw):
        """Integrate the scan that ends at t_end, deskewed between the previous and this estimate."""
        if self.nav is None:
            return
        Tprev, self._Tw_prev = self._Tw_prev, Tw
        if not self._nav_scans:
            return
        key = min(self._nav_scans, key=lambda k: abs(k - t_end))
        if abs(key - t_end) > 0.03:
            return
        pts = self._nav_scans.pop(key)
        R_bl, p_bl = rot_y(LIDAR_PITCH), LIDAR_POS - IMU_POS
        T0 = Tprev if Tprev is not None else Tw
        from scipy.spatial.transform import Rotation, Slerp
        sl = Slerp([0.0, 0.1], Rotation.from_matrix(np.stack([T0[:3, :3], Tw[:3, :3]])))

        def pose_fn(toff):
            a = np.clip(np.asarray(toff, np.float64), 0.0, 0.1)
            Rb = sl(a).as_matrix()
            pb = T0[:3, 3][None] + (a / 0.1)[:, None] * (Tw[:3, 3] - T0[:3, 3])[None]
            return pb + Rb @ p_bl, Rb @ R_bl
        self.nav.on_scan(t_end, pts, pose_fn)

    def _nav_zground(self):
        if self.last_est is None:
            return None
        return float(self.last_est["Tw"][2, 3]) - 0.78

    def _nav_pose(self):
        if self.loc is not None and self.loc.mode != ML.TRACK and self.T_odom_prev is not None:
            return None                    # wait for relocalization
        if self.last_est is None:
            return None
        e = self.last_est
        if self.health == "stop" and (e["sigma_pos"] > self.sigma_pos_max or e["sigma_yaw_deg"] > self.sigma_yaw_max):
            if self.stopped_at is None:
                self.stopped_at = e["t"]
            self.done = True              # fail-safe: stop the mission
            return None
        Tw = e["Tw"]
        # IMU frame is at the pelvis origin, so its xy is the pelvis xy
        return Tw[:2, 3].copy(), np.arctan2(Tw[1, 0], Tw[0, 0])

    def _on_step(self, t):
        super()._on_step(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="corridor", choices=S.SCENES)
    ap.add_argument("--est", default="4d_leg", choices=list(ESTIMATORS))
    ap.add_argument("--health", default="none", choices=["none", "stop"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=300.0)
    ap.add_argument("--map", default=None, help="prior map (tools/maploc.py build-map)")
    ap.add_argument("--vx", type=float, default=1.0, help="max walking speed command [m/s]")
    ap.add_argument("--recover", action="store_true", help="assisted stand-up after falls, then continue")
    ap.add_argument("--nav", default=None, choices=[None, "mppi"],
                    help="LiDAR map + navigation function + MPPI on the estimated pose")
    ap.add_argument("--nav-dyn", default="doppler", choices=["doppler", "none", "raycast", "persist", "doppler+persist", "doppler+raycast"])
    ap.add_argument("--costmap", action="store_true",
                    help="GT free-space costmap steering (Hybrid: geom+terrain GT, not learned)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out)
    rec = ClosedLoopRecorder(a.scene, est=a.est, health=a.health, seed=a.seed, map_path=a.map,
                             vx_max=a.vx, recover=a.recover, costmap=a.costmap, nav=a.nav, nav_dyn=a.nav_dyn)
    rec.start_estimator(out)
    meta = rec.run(out, t_max=a.t_max, verbose=False)
    rec.est.close()
    gt = np.load(out / "gt.npz")
    route = rec.scene.waypoints
    ct = cross_track(gt["p"][:, :2], route)
    el = np.array(rec.est_log)
    k = np.clip(np.searchsorted(gt["t"], el[:, 0]), 0, len(gt["t"]) - 1)
    est_err = np.linalg.norm(el[:, 1:3] - gt["p"][k, :2], axis=1) if len(el) else np.array([np.nan])
    res = dict(scene=a.scene, est=a.est, health=a.health, completed=meta["completed"], fell=meta["fell"],
               duration=meta["duration"], path_length=meta["path_length"],
               goal_err=float(np.linalg.norm(gt["p"][-1, :2] - route[-1])),
               cross_track_rms=float(np.sqrt(np.mean(ct ** 2))), cross_track_max=float(ct.max()),
               est_err_final=float(est_err[-1]), est_err_max=float(np.nanmax(est_err)),
               collisions=rec.collisions, stopped_at=rec.stopped_at,
               # is the reported uncertainty honest? (fraction of time true error is within 3 sigma)
               within_3sigma=float(np.mean(est_err <= 3 * np.maximum(el[:, 5], 1e-3))) if len(el) else float("nan"),
               sigma_pos_final=float(el[-1, 5]) if len(el) else float("nan"),
               n_falls=len(rec.falls), falls=[dict(t=round(f[0], 2), recovered=(round(f[1], 2) if f[1] else None))
                                              for f in rec.falls], vx_max=a.vx, costmap=a.costmap,
               nav=meta.get("nav"))
    if rec.map_log:
        ml = np.array(rec.map_log)
        res.update(map=str(a.map), map_tracking_frac=float(ml[:, 1].mean()), map_lost=int(ml[:, 3].sum()),
                   map_fitness_median=float(np.nanmedian(ml[:, 2])))
    np.savetxt(out / "nav_est.csv", el, delimiter=",", header="t,x,y,z,yaw,sigma_pos,sigma_yaw_deg", comments="")
    (out / "closed_loop.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    (out / "lidar.bin").unlink(missing_ok=True)       # keep disk usage down


if __name__ == "__main__":
    main()
