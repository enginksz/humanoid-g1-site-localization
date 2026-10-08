"""Record a G1 humanoid walking through a construction-site scene.

Outputs one sequence directory:
  imu.bin      float64 records (t, wx, wy, wz, ax, ay, az), body = IMU frame
  lidar.bin    per scan: float64 t_begin, uint32 n, n x (x, y, z, doppler, t_off) float32
               points in LiDAR frame at their own firing time (true motion distortion);
               doppler = -dir . v_lidar for static points (FMCW-LIO convention)
  legs.npz     t, q(12), dq(12), contact force per foot (GT), foot slip speed (GT)
  gt.npz       t, p_wb, q_wb (wxyz), v_wb (world), w_b (body) for the IMU frame
  gt_tum.txt   TUM poses of the IMU frame (for evo)
  meta.json    extrinsics, sensor + noise parameters, scene description

Usage:  python3 -m sim.record --scene slab --out data/slab
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import scenes as S
from .costmap import GtCostmap

ROOT = Path(__file__).resolve().parents[1]
RL_GYM = ROOT / "third_party" / "unitree_rl_gym"
G1_XML = RL_GYM / "resources/robots/g1_description/g1_12dof.xml"
POLICY = RL_GYM / "deploy/pre_train/g1/motion.pt"

SIM_DT = 0.002
CTRL_DECIM = 10            # 50 Hz policy
IMU_DECIM = 2              # 250 Hz IMU / encoders
KPS = np.array([100, 100, 100, 150, 40, 40] * 2, float)
KDS = np.array([2, 2, 2, 4, 2, 2] * 2, float)
DEFAULT_Q = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, float)

# IMU mounted in the pelvis, LiDAR in the head (pitched 8 deg down to see the floor)
IMU_POS = np.array([0.0, 0.0, 0.0])
LIDAR_POS = np.array([0.08, 0.0, 0.46])
LIDAR_PITCH = np.deg2rad(8.0)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).copy())
    return q


# ----------------------------------------------------------------- sensors
class LidarModel:
    """Rolling FMCW LiDAR, Aeva Aeries II-like: 120 x 30 deg FoV, 10 Hz.

    Columns sweep left->right through the scan period, so each column is cast
    at the simulation step in which it would actually be fired.
    """

    def __init__(self, kind="aeva", rng=None):
        self.kind = kind
        if kind == "aeva":
            self.az = np.deg2rad(np.linspace(60, -60, 240))
            self.el = np.deg2rad(np.linspace(-15, 15, 48))
        elif kind == "ouster":   # 360 deg spinning, 32 beams (hypothetical FMCW variant)
            self.az = np.deg2rad(np.linspace(180, -180, 512, endpoint=False))
            self.el = np.deg2rad(np.linspace(-22.5, 22.5, 32))
        else:
            raise ValueError(kind)
        self.period = 0.1
        self.max_range = 120.0
        self.min_range = 0.4
        self.range_sigma = 0.02
        self.doppler_sigma = 0.03
        self.dropout = 0.02
        self.rng = rng or np.random.default_rng(0)
        A, E = np.meshgrid(self.az, self.el, indexing="ij")          # (ncol, nrow)
        self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)

    def meta(self):
        return dict(kind=self.kind, n_cols=len(self.az), n_rows=len(self.el), period=self.period,
                    fov_az_deg=float(np.rad2deg(self.az.max() - self.az.min())),
                    fov_el_deg=float(np.rad2deg(self.el.max() - self.el.min())),
                    range_sigma=self.range_sigma, doppler_sigma=self.doppler_sigma,
                    dropout=self.dropout, max_range=self.max_range)


class Recorder:
    def __init__(self, scene_name, lidar_kind="aeva", seed=0, imu_grade="mems", laps=1, spec_hook=None,
                 vx_max=1.0, recover=False, costmap=False, nav=None, nav_dyn="doppler",
                 heading_deadband=0.75, yaw_rate_max=1.0, arrive_taper=0.5):
        self.rng = np.random.default_rng(seed)
        spec = mujoco.MjSpec.from_file(str(G1_XML))
        self.scene = S.build(spec, scene_name, seed=seed)
        if laps > 1:   # closed routes only: repeat the loop, stops keep their index within each lap
            w = self.scene.waypoints
            n = len(w) - 1
            self.scene.waypoints = np.concatenate([w[:1]] + [w[1:]] * laps)
            self.scene.stops = {k + lap * n: v for lap in range(laps) for k, v in self.scene.stops.items()}
        pelvis = spec.body("pelvis")
        imu_site = pelvis.add_site()
        imu_site.name = "imu"
        imu_site.pos = list(IMU_POS)
        lid = pelvis.add_site()
        lid.name = "lidar"
        lid.pos = list(LIDAR_POS)
        lid.quat = list(R_to_quat_wxyz(rot_y(LIDAR_PITCH)))
        for kind, nm in [(mujoco.mjtSensor.mjSENS_GYRO, "gyro"), (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc")]:
            s = spec.add_sensor()
            s.name = nm
            s.type = kind
            s.objtype = mujoco.mjtObj.mjOBJ_SITE
            s.objname = "imu"
        if spec_hook is not None:
            spec_hook(spec)              # e.g. lights/sky for the live viewer (no effect on sensors)
        self.m = spec.compile()
        self.m.opt.timestep = SIM_DT
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.policy = torch.jit.load(str(POLICY))
        self.lidar = LidarModel(lidar_kind, self.rng)

        m = self.m
        self.sid_imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.sid_lidar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.adr_gyro = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
        self.adr_acc = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "acc")]
        self.foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                            for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        for a in self.scene.actors:
            a.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.name)
            a.mocap_id = m.body_mocapid[a.body_id]
        self.actor_by_body = {a.body_id: a for a in self.scene.actors}
        for sl in self.scene.sliders:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sl.name + "_x")
            sl.qpos_adr, sl.qvel_adr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
            sl.act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, sl.name + "_servo")
            sl.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sl.name)
        # robot: free joint + 12 leg joints come first in qpos/qvel/ctrl
        self.QJ = slice(7, 19)
        self.DQJ = slice(6, 18)
        # LiDAR sees static scene + actors only (not the robot)
        self.geomgroup = np.array([0, 0, 1, 1, 0, 0], np.uint8)

        # IMU error model (per-sample white noise at 250 Hz + constant bias + random walk)
        if imu_grade == "mems":
            self.imu_noise = dict(gyro_sigma=0.004, acc_sigma=0.03,
                                  gyro_bias=self.rng.normal(0, 0.004, 3),
                                  acc_bias=self.rng.normal(0, 0.04, 3),
                                  gyro_rw=2e-5, acc_rw=2e-4)
        else:
            self.imu_noise = dict(gyro_sigma=0.0, acc_sigma=0.0, gyro_bias=np.zeros(3),
                                  acc_bias=np.zeros(3), gyro_rw=0.0, acc_rw=0.0)

        # walking speed and fall handling
        self.vx_max = vx_max
        self.recover = recover
        self.heading_deadband = heading_deadband
        self.yaw_rate_max = yaw_rate_max
        self.arrive_taper = arrive_taper
        self.rec_state = None
        self.falls = []                 # (t_fall, t_recovered or None)
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.robot_mass = float(self.m.body_subtreemass[self.pelvis])
        # GT free-space costmap (Hybrid plan: control uses GT, not a learned model)
        self.costmap = None
        self.last_cost = 0.0
        self._costmap_t = -1.0
        if costmap:
            terrain = getattr(self.scene, "terrain", None)
            self.costmap = GtCostmap(self.m, terrain=terrain)
        # map-based navigation (sim/nav.py): LiDAR map + navigation function + MPPI
        self.nav = None
        self._nav_cmd_t = -1.0
        self._scan_poses = []
        if nav:
            from .nav import MapNavigator
            wps = self.scene.waypoints
            T = getattr(self.scene, "terrain", None)
            z_lo, z_hi = (-4.0, 6.0) if T is not None else (-1.5, 4.5)
            lo = [wps[:, 0].min() - 22, wps[:, 1].min() - 22, z_lo]
            hi = [wps[:, 0].max() + 22, wps[:, 1].max() + 22, z_hi]
            self.nav = MapNavigator(lo, hi, v_max=vx_max, dyn=("doppler" if "doppler" in nav_dyn else "none"),
                                    raycast=("raycast" in nav_dyn), seed=seed,
                                    persist=(1.5 if "persist" in nav_dyn else 0.0))
        # controller state
        self.action = np.zeros(12)
        self.target_q = DEFAULT_Q.copy()
        self.obs = np.zeros(47, np.float32)
        self.cmd = np.zeros(3)
        self.wp_idx = 1
        self.stop_until = None
        self.done = False
        # collision accounting (any non-floor contact with the robot)
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.robot_bodies = {b for b in range(self.m.nbody) if self.m.body_rootid[b] == root}
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.collisions = 0
        self._in_collision = False

    # --------------------------------------------------------- control
    def _policy_step(self, step):
        d = self.d
        w, x, y, z = d.qpos[3:7]
        grav = np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])
        ph = (step * SIM_DT) % 0.8 / 0.8
        o = self.obs
        o[:3] = d.qvel[3:6] * 0.25
        o[3:6] = grav
        o[6:9] = self.cmd * np.array([2.0, 2.0, 0.25])
        o[9:21] = d.qpos[self.QJ] - DEFAULT_Q
        o[21:33] = d.qvel[self.DQJ] * 0.05
        o[33:45] = self.action
        o[45:47] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(o)[None]).numpy()[0]
        self.target_q = self.action * 0.25 + DEFAULT_Q

    def _follow(self, t, t_start):
        """Waypoint follower on ground-truth pose (navigation is not under test)."""
        if t < t_start or self.done:
            self.cmd[:] = 0
            return
        if self.stop_until is not None:
            if t < self.stop_until:
                self.cmd[:] = 0
                return
            self.stop_until = None
        wps = self.scene.waypoints
        nav = self._nav_pose()
        if nav is None:                    # no pose yet (estimator initializing): stand
            self.cmd[:] = 0
            return
        p, yaw = nav
        goal = wps[self.wp_idx]
        dvec = goal - p
        dist = np.linalg.norm(dvec)
        if dist < 0.35:
            if self.wp_idx in self.scene.stops:
                self.stop_until = t + self.scene.stops[self.wp_idx]
            self.wp_idx += 1
            if self.wp_idx >= len(wps):
                self.done = True
            self.cmd[:] = 0
            return
        if self.nav is not None:
            if t - self._nav_cmd_t >= 0.1 - 1e-9:      # planner at 10 Hz, command held in between
                self._nav_cmd_t = t
                vx, wz = self.nav.command(t, np.asarray(p, float), float(yaw), goal.astype(float),
                                          z_ground=self._nav_zground())
                self.cmd[:] = [vx, 0.0, wz]
            return
        err = np.arctan2(dvec[1], dvec[0]) - yaw
        err = (err + np.pi) % (2 * np.pi) - np.pi
        vx_scale = 1.0
        db = self.heading_deadband
        # GT costmap: query at true pelvis pose; refresh ~20 Hz (dynamics move)
        if self.costmap is not None:
            gt_xy = self.d.qpos[:2].copy()
            R_gt = quat_wxyz_to_R(self.d.qpos[3:7])
            gt_yaw = float(np.arctan2(R_gt[1, 0], R_gt[0, 0]))
            if t - self._costmap_t >= 0.05:
                self.costmap.update(self.d, self.scene.actors, xy=gt_xy, yaw=gt_yaw)
                self._costmap_t = t
            vx_scale, yaw_off, self.last_cost = self.costmap.steer(gt_xy, gt_yaw, goal)
            err = err + yaw_off
            err = (err + np.pi) % (2 * np.pi) - np.pi
            if vx_scale < 0.05:
                db = max(db, 0.85)
        # slope-aware crawl: pretrained walk policy tips on grade changes
        T = getattr(self.scene, "terrain", None)
        if T is not None:
            xy = self.d.qpos[:2]
            s_here = float(T.slope_deg(float(xy[0]), float(xy[1])))
            # look 0.8 m ahead along commanded heading
            ahead = xy + 0.8 * np.array([np.cos(yaw + err), np.sin(yaw + err)])
            s_ahead = float(T.slope_deg(float(ahead[0]), float(ahead[1])))
            z0 = float(T.height(float(xy[0]), float(xy[1])))
            z1 = float(T.height(float(ahead[0]), float(ahead[1])))
            grade = abs(z1 - z0) / 0.8
            smax = max(s_here, s_ahead)
            if grade > 0.10 or smax > 8.0:          # steep / crest transition
                vx_scale = min(vx_scale, 0.28)
                db = max(db, 0.70)
            elif grade > 0.07 or smax > 5.5:
                vx_scale = min(vx_scale, 0.42)
                db = max(db, 0.60)
            elif grade > 0.045 or smax > 3.5:
                vx_scale = min(vx_scale, 0.62)
        yaw_lim = self.yaw_rate_max * (1.6 if vx_scale < 0.05 else 1.0)
        wz = np.clip(2.2 * err, -yaw_lim, yaw_lim)
        vm = self.vx_max * vx_scale
        if abs(err) > db or vx_scale < 1e-3:
            vx = 0.0
        else:
            taper = min(1.0, dist / self.arrive_taper + 0.5)
            vx = np.clip(vm * (1 - abs(err) / db), 0.0, vm) * taper
        self.cmd[:] = [vx, 0.0, wz]

    # --------------------------------------------------------- fall recovery
    def _ground(self, x, y):
        T = getattr(self.scene, "terrain", None)
        return float(T.height(x, y)) if T is not None else 0.0

    def _fallen(self):
        d = self.d
        R = quat_wxyz_to_R(d.qpos[3:7])
        return (d.qpos[2] - self._ground(*d.qpos[:2]) < 0.45) or R[2, 2] < np.cos(np.deg2rad(50))

    def _recovery_step(self, t):
        """Assisted stand-up: a virtual harness (PD wrench on the pelvis + gravity
        compensation) lifts the robot upright where it fell, legs servo to the default
        stance, then the harness lets go and the walking policy resumes.
        A real robot needs a learned get-up controller; this keeps the run going so the
        estimator's behaviour through a fall and a restart can be studied."""
        d, rs = self.d, self.rec_state
        el = t - rs["t0"]
        R = quat_wxyz_to_R(d.qpos[3:7])
        z_stand = self._ground(*rs["xy"]) + 0.78
        ramp = min(el / 1.5, 1.0)
        p_tgt = np.array([*rs["xy"], rs["z0"] + (z_stand - rs["z0"]) * ramp])
        c, s_ = np.cos(rs["yaw"]), np.sin(rs["yaw"])
        R_tgt = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
        rv = np.zeros(3)
        mujoco.mju_quat2Vel(rv, R_to_quat_wxyz(R_tgt @ R.T), 1.0)
        v, w = d.qvel[0:3], d.qvel[3:6]
        w_world = R @ w                                # free-joint angular velocity is in the body frame
        release = float(np.clip((el - rs["t_release"]) / 0.6, 0, 1)) if rs["t_release"] else 0.0
        gain = 1.0 - release
        F = gain * (3000 * (p_tgt - d.xpos[self.pelvis]) - 600 * v + np.array([0, 0, self.robot_mass * 9.81]))
        Tq = gain * (600 * rv - 80 * w_world)
        d.xfrc_applied[self.pelvis, :3] = F
        d.xfrc_applied[self.pelvis, 3:] = Tq
        upright = R[2, 2] > np.cos(np.deg2rad(5)) and abs(d.qpos[2] - z_stand) < 0.05 and np.linalg.norm(v) < 0.1
        if rs["t_release"] is None and el > 1.8 and upright:
            rs["t_release"] = el
        if rs["t_release"] is not None and release >= 1.0:
            d.xfrc_applied[self.pelvis] = 0
            self.rec_state = None
            self.action[:] = 0
            self.falls[-1] = (self.falls[-1][0], t)
            return False
        return True

    # hooks for closed-loop subclasses (sim/closed_loop.py)
    def _nav_pose(self):
        """(xy, yaw) the waypoint follower steers with. Default: ground truth pelvis."""
        R = quat_wxyz_to_R(self.d.qpos[3:7])
        return self.d.qpos[:2].copy(), np.arctan2(R[1, 0], R[0, 0])

    def _on_imu(self, t, imu_row, q, dq, F):
        pass

    def _on_scan(self, t0, pts):
        pass

    def _nav_zground(self):
        """Robot's ground height for the map window (base: true pelvis height - stance height)."""
        return float(self.d.qpos[2]) - 0.78

    def _nav_scan(self, t0, pts):
        """Feed the navigation map. Base recorder: true LiDAR pose of each firing step."""
        if self.nav is None or not self._scan_poses:
            return
        toffs = np.array([sp[0] for sp in self._scan_poses])
        Ps = np.array([sp[1] for sp in self._scan_poses])
        Rs = np.array([sp[2] for sp in self._scan_poses])

        def pose_fn(tq):
            k = np.clip(np.searchsorted(toffs, tq, side="right") - 1, 0, len(toffs) - 1)
            return Ps[k], Rs[k]
        self.nav.on_scan(t0 + 0.1, pts, pose_fn)

    def _on_step(self, t):
        if int(round(t / 0.002)) % 10:
            return
        d, m = self.d, self.m
        hit = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            g_other = c.geom2 if b1 in self.robot_bodies else c.geom1 if b2 in self.robot_bodies else None
            if g_other is None or g_other == self.floor:
                continue
            if m.geom_bodyid[g_other] in self.robot_bodies:
                continue
            hit = True
            break
        if hit and not self._in_collision:
            self.collisions += 1
        self._in_collision = hit

    # --------------------------------------------------------- sensing
    def _imu_sample(self, t):
        d, n = self.d, self.imu_noise
        n["gyro_bias"] = n["gyro_bias"] + self.rng.normal(0, n["gyro_rw"], 3)
        n["acc_bias"] = n["acc_bias"] + self.rng.normal(0, n["acc_rw"], 3)
        g = d.sensordata[self.adr_gyro:self.adr_gyro + 3] + n["gyro_bias"] + self.rng.normal(0, n["gyro_sigma"], 3)
        a = d.sensordata[self.adr_acc:self.adr_acc + 3] + n["acc_bias"] + self.rng.normal(0, n["acc_sigma"], 3)
        return np.concatenate([[t], g, a])

    def _site_vel(self, sid):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
        return res[3:], res[:3]            # linear (world), angular (world)

    def _cast_columns(self, cols, t_off):
        """Cast a block of LiDAR columns at the current state; returns (k,5) float32."""
        L = self.lidar
        d = self.d
        R_wl = d.site_xmat[self.sid_lidar].reshape(3, 3)
        p_wl = d.site_xpos[self.sid_lidar].copy()
        if self.nav is not None:
            self._scan_poses.append((t_off, p_wl, R_wl.copy()))
        v_l, w_l = self._site_vel(self.sid_lidar)
        dirs_l = L.dirs[cols].reshape(-1, 3)
        dirs_w = dirs_l @ R_wl.T
        n = len(dirs_w)
        geomid = np.full(n, -1, np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(self.m, d, p_wl, dirs_w.reshape(-1).copy(), self.geomgroup, 1, -1,
                           geomid, dist, None, n, L.max_range)
        ok = (geomid >= 0) & (dist > L.min_range) & (self.rng.random(n) > L.dropout)
        if not ok.any():
            return np.zeros((0, 5), np.float32)
        dirs_w, dirs_l, dist, gid = dirs_w[ok], dirs_l[ok], dist[ok], geomid[ok]
        # relative radial velocity: dir . (v_target - v_sensor); v_sensor is the
        # LiDAR origin velocity (rotation does not move the origin)
        v_tgt = np.zeros_like(dirs_w)
        bodies = self.m.geom_bodyid[gid]
        for sl in self.scene.sliders:
            sel = bodies == sl.body_id
            if sel.any():
                v_tgt[sel] = [d.qvel[sl.qvel_adr], 0.0, 0.0]
        if self.actor_by_body:
            for bid, actor in self.actor_by_body.items():
                sel = bodies == bid
                if sel.any():
                    v_tgt[sel] = actor.cur_vel
        doppler = np.einsum("ij,ij->i", dirs_w, v_tgt - v_l)
        doppler += self.rng.normal(0, L.doppler_sigma, len(doppler))
        r = dist + self.rng.normal(0, L.range_sigma, len(dist))
        pts = dirs_l * r[:, None]
        out = np.empty((len(r), 5), np.float32)
        out[:, :3] = pts
        out[:, 3] = doppler
        out[:, 4] = t_off
        return out

    # --------------------------------------------------------- main loop
    def run(self, out_dir, t_stand=2.5, t_max=400.0, verbose=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        m, d, L = self.m, self.d, self.lidar
        mujoco.mj_forward(m, d)
        steps_per_scan = int(round(L.period / SIM_DT))
        ncol = len(L.az)
        col_edges = np.linspace(0, ncol, steps_per_scan + 1).astype(int)

        imu_rows, leg_t, leg_q, leg_dq, leg_f, leg_slip = [], [], [], [], [], []
        gt_t, gt_p, gt_q, gt_v, gt_w = [], [], [], [], []
        lidar_f = open(out / "lidar.bin", "wb")
        scan_buf, scan_t0 = [], None
        n_scans = n_pts = 0
        wall0 = time.time()
        fell = False

        step = 0
        while True:
            t = step * SIM_DT              # time of the state that forward() is evaluated at
            for a in self.scene.actors:
                pos, vel, yaw = a.state(t)
                d.mocap_pos[a.mocap_id] = pos
                d.mocap_quat[a.mocap_id] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                a.cur_vel = vel
            recovering = self.rec_state is not None and self._recovery_step(t)
            if recovering:
                self.cmd[:] = 0
                self.target_q = DEFAULT_Q.copy()
            else:
                self._follow(t, t_stand)
            for sl in self.scene.sliders:
                d.ctrl[sl.act_id] = sl.target(t)
            d.ctrl[:12] = (self.target_q - d.qpos[self.QJ]) * KPS - d.qvel[self.DQJ] * KDS
            mujoco.mj_step(m, d)           # forward() at state t, then integrate
            # d.xpos / sensordata / cvel now describe the state at time t

            self._on_step(t)
            if step % IMU_DECIM == 0:
                imu_rows.append(self._imu_sample(t))
                R_wb = d.site_xmat[self.sid_imu].reshape(3, 3)
                v_b, w_b = self._site_vel(self.sid_imu)
                gt_t.append(t)
                gt_p.append(d.site_xpos[self.sid_imu].copy())
                gt_q.append(R_to_quat_wxyz(R_wb))
                gt_v.append(v_b)
                gt_w.append(R_wb.T @ w_b)
                leg_t.append(t)
                leg_q.append(d.qpos[self.QJ].copy() + self.rng.normal(0, 1e-3, 12))   # encoder noise
                leg_dq.append(d.qvel[self.DQJ].copy() + self.rng.normal(0, 2e-2, 12))
                f, slip = np.zeros(2), np.zeros(2)
                for k, bid in enumerate(self.foot_bodies):
                    cf = d.cfrc_ext[bid]
                    f[k] = np.linalg.norm(cf[3:])
                    vv = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vv, 0)
                    slip[k] = np.linalg.norm(vv[3:5]) if f[k] > 20 else 0.0
                leg_f.append(f)
                leg_slip.append(slip)
                self._on_imu(t, imu_rows[-1], leg_q[-1], leg_dq[-1], f)

            # LiDAR: this step fires columns [col_edges[k], col_edges[k+1])
            k = step % steps_per_scan
            if k == 0:
                scan_t0 = t
                scan_buf = []
                self._scan_poses = []
            cols = np.arange(col_edges[k], col_edges[k + 1])
            if len(cols):
                scan_buf.append(self._cast_columns(cols, t - scan_t0))
            if k == steps_per_scan - 1:
                pts = np.concatenate(scan_buf) if scan_buf else np.zeros((0, 5), np.float32)
                lidar_f.write(struct.pack("<dI", scan_t0, len(pts)))
                lidar_f.write(pts.astype(np.float32).tobytes())
                self._on_scan(scan_t0, pts)
                self._nav_scan(scan_t0, pts)
                n_scans += 1
                n_pts += len(pts)

            if step % CTRL_DECIM == 0 and self.rec_state is None:
                self._policy_step(step)

            step += 1
            if self.rec_state is None and self._fallen():
                fell = True
                if verbose:
                    print(f"  robot fell at t={t:.2f}s pos={d.qpos[:3].round(2)}")
                if not self.recover or len(self.falls) >= 10:
                    break
                R = quat_wxyz_to_R(d.qpos[3:7])
                self.falls.append((t, None))
                self.rec_state = dict(t0=t, xy=d.qpos[:2].copy(), z0=float(d.xpos[self.pelvis][2]),
                                      yaw=float(np.arctan2(R[1, 0], R[0, 0])), t_release=None)
            if (self.done and (step % steps_per_scan == 0)) or t > t_max:
                break
            if verbose and step % 5000 == 0:
                print(f"  t={t:6.1f}s wp={self.wp_idx}/{len(self.scene.waypoints)} "
                      f"pos={d.qpos[:3].round(2)} scans={n_scans} wall={time.time() - wall0:.0f}s", flush=True)
        lidar_f.close()

        imu = np.array(imu_rows)
        imu.astype(np.float64).tofile(out / "imu.bin")
        np.savez_compressed(out / "legs.npz", t=np.array(leg_t), q=np.array(leg_q), dq=np.array(leg_dq),
                            foot_force=np.array(leg_f), foot_slip=np.array(leg_slip))
        gt_q = np.array(gt_q)
        np.savez_compressed(out / "gt.npz", t=np.array(gt_t), p=np.array(gt_p), q=gt_q,
                            v=np.array(gt_v), w=np.array(gt_w))
        with open(out / "gt_tum.txt", "w") as f:
            for tt, p, q in zip(gt_t, gt_p, gt_q):
                f.write(f"{tt:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f} {q[0]:.7f}\n")

        R_bl = rot_y(LIDAR_PITCH)
        p_bl_b = LIDAR_POS - IMU_POS
        gtp = np.array(gt_p)
        meta = dict(
            scene=self.scene.name, description=self.scene.description,
            duration=float(gt_t[-1]), n_scans=n_scans, mean_points_per_scan=n_pts / max(n_scans, 1),
            path_length=float(np.linalg.norm(np.diff(gtp[:, :2], axis=0), axis=1).sum()),
            fell=fell, completed=self.done, t_stand=t_stand,
            falls=[dict(t=f[0], recovered=f[1]) for f in self.falls],
            collisions=self.collisions, vx_max=self.vx_max,
            costmap=self.costmap is not None,
            nav=(None if self.nav is None else dict(dyn=self.nav.mapper.dyn, raycast=self.nav.mapper.raycast,
                                                    ms_mean=1e3 * float(np.mean(self.nav.timing or [0])),
                                                    ms_p95=1e3 * float(np.percentile(self.nav.timing or [0], 95)))),
            imu_rate=1.0 / (SIM_DT * IMU_DECIM), lidar=L.meta(),
            R_bl=R_bl.reshape(-1).tolist(), p_bl_b=p_bl_b.tolist(),
            imu_noise={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.imu_noise.items()},
            slip_patches=self.scene.slip_patches,
            actors=[a.name for a in self.scene.actors],
            wall_time=time.time() - wall0,
        )
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"  done: {n_scans} scans, {n_pts / max(n_scans, 1):.0f} pts/scan, "
                  f"path {meta['path_length']:.1f} m, {meta['duration']:.1f} s sim, {meta['wall_time']:.0f} s wall")
        return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slab", choices=S.SCENES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lidar", default="aeva", choices=["aeva", "ouster"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--t-stand", type=float, default=2.5, help="seconds standing before walking")
    ap.add_argument("--laps", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(1)
    Recorder(args.scene, args.lidar, args.seed, laps=args.laps).run(args.out, t_stand=args.t_stand, t_max=args.t_max)


if __name__ == "__main__":
    main()
