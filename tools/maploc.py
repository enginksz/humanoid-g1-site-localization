"""Localization in a prior map of a changing construction site.

Architecture (what a site robot runs every day):

  LIO odometry (smooth, drifts) --delta--> [ TRACK: robust point-to-plane ICP vs prior map ]
                                              |  health: inlier ratio, ICP jump, degeneracy
                                              v  lost?
                                            [ GLOBAL: particle filter (x, y, yaw) on a 3D
                                              likelihood field, roll/pitch from gravity ]
                                              |  converged (particle spread small)
                                              +--> back to TRACK

The prior map is the "day 0" survey: day-0 LiDAR scans accumulated with
ground-truth poses (in practice: a SLAM map aligned to site/BIM coordinates).
Output poses are in the map frame = GT world frame, so errors need no alignment.

  python3 tools/maploc.py build-map data/slab results/maploc/map_day0.npz
  python3 tools/maploc.py run data/slab_day7 results/slab_day7/fmcw4d_leg/poses_tum.txt out_dir \
          --map results/maploc/map_day0.npz [--init gt|global] [--kidnap 40,15]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.record import quat_wxyz_to_R  # noqa: E402
from tools.seqio import iter_scans, read_meta, read_tum  # noqa: E402

TRACK, GLOBAL = "TRACK", "GLOBAL"


def pose_mat(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(q)
    T[:3, 3] = p
    return T


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def voxel(P, size):
    k = np.floor(P / size).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[i]


def scan_to_body(s, R_bl, p_bl, rmin=0.8, rmax=60.0):
    xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(r > rmin) & (r < rmax)]
    return xyz @ R_bl.T + p_bl


def deskew(s, R_bl, p_bl, dT, period=0.1, rmin=0.8, rmax=60.0):
    """Move every point to the scan-end body frame, assuming constant velocity over
    the last odometry increment dT (scan-end to scan-end, 0.1 s)."""
    xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    ok = (r > rmin) & (r < rmax)
    xyz, tau = xyz[ok] @ R_bl.T + p_bl, s["t_off"][ok].astype(np.float64)
    a = (tau[-1] - tau) / period if len(tau) else tau     # fraction of dT still to go
    rv = Rotation.from_matrix(dT[:3, :3]).as_rotvec()
    # point at time tau, expressed in the end frame: T_end^-1 T_tau = (dT^a)^-1
    R = Rotation.from_rotvec(-a[:, None] * rv[None, :])
    return R.apply(xyz - a[:, None] * dT[:3, 3][None, :])


# ------------------------------------------------------------------ prior map
def build_map(seq, out, voxel_size=0.1, stride=2, grid_res=0.15):
    seq = Path(seq)
    meta = read_meta(seq)
    R_bl, p_bl = np.array(meta["R_bl"]).reshape(3, 3), np.array(meta["p_bl_b"])
    tg, pg, qg = read_tum(seq / "gt_tum.txt")
    pts = []
    for i, (t0, s) in enumerate(iter_scans(seq)):
        if i % stride:
            continue
        # per-column timestamps: put each point at its own firing time (GT deskew)
        t = t0 + s["t_off"].astype(np.float64)
        k = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1)
        xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        ok = (r > 0.8) & (r < 60)
        xyz, k = xyz[ok] @ R_bl.T + p_bl, k[ok]
        for kk in np.unique(k):
            sel = k == kk
            pts.append(xyz[sel] @ quat_wxyz_to_R(qg[kk]).T + pg[kk])
        if i % 100 == 0:
            pts = [voxel(np.concatenate(pts), voxel_size)]
    P = voxel(np.concatenate(pts), voxel_size)
    lo = P.min(0) - 2.0
    hi = P.max(0) + 2.0
    shape = np.ceil((hi - lo) / grid_res).astype(int) + 1
    occ = np.ones(shape, bool)
    idx = np.floor((P - lo) / grid_res).astype(int)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = False
    edt = distance_transform_edt(occ) * grid_res
    np.savez_compressed(out, points=P.astype(np.float32), edt=edt.astype(np.float16), lo=lo, res=grid_res,
                        z_body=float(np.median(pg[:, 2])))
    print(f"map: {len(P)} points, grid {shape}, saved {out}")


class PriorMap:
    def __init__(self, path):
        z = np.load(path)
        self.P = z["points"].astype(np.float64)
        self.edt = z["edt"].astype(np.float32)
        self.lo, self.res = z["lo"], float(z["res"])
        self.z_body = float(z["z_body"])
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.P))
        self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=20))
        self.N = np.asarray(self.pcd.normals).copy()
        self.tree = cKDTree(self.P)            # built once (open3d rebuilds per ICP call)
        # free space at body height, for spreading particles
        zi = int(round((self.z_body - self.lo[2]) / self.res))
        sl = self.edt[:, :, zi]
        ok = np.argwhere(sl > 0.45)
        self.free_xy = ok * self.res + self.lo[:2]
        # global-search candidates: free cells inside the structure's footprint
        # (the floor plane extends far beyond the building and is not a candidate)
        st = self.P[self.P[:, 2] > 0.3]
        blo, bhi = np.percentile(st[:, :2], 1, axis=0), np.percentile(st[:, :2], 99, axis=0)
        inb = np.all((self.free_xy > blo) & (self.free_xy < bhi), axis=1)
        self.free_xy = self.free_xy[inb]
        g = np.unique(np.round(self.free_xy / 0.5), axis=0) * 0.5      # 0.5 m coarse grid
        keep = self.dist(np.c_[g, np.full(len(g), self.z_body)]) > 0.45
        self.cand_xy = g[keep]

    def dist(self, X):
        """Distance to the nearest map point for world points X (..., 3)."""
        g = ((X - self.lo) / self.res).reshape(-1, 3).T
        d = map_coordinates(self.edt, g, order=0, mode="nearest")
        return d.reshape(X.shape[:-1])


# ------------------------------------------------------------------ localizer
class MapLocalizer:
    def __init__(self, pmap: PriorMap, n_particles=3000, seed=0, tukey_c=0.3, global_method="submap"):
        self.m = pmap
        self.rng = np.random.default_rng(seed)
        self.N = n_particles
        self.mode = GLOBAL
        self.T = np.eye(4)
        self.bad = 0
        self.tukey_c = tukey_c
        self.global_method = global_method
        self.sub_buf = []          # (T_odom, points) for the global-localization submap
        self.cands = None          # multi-hypothesis verification: list of dicts(T, fits)
        self.baseline = None       # running inlier-ratio baseline while healthy
        self.n_global_updates = 0
        # ICP: robust point-to-plane; Tukey kernel tolerates the site changes
        self.icp_est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.3))

    # ---------------- global (particle filter)
    def init_global(self):
        i = self.rng.integers(0, len(self.m.free_xy), self.N)
        self.X = np.c_[self.m.free_xy[i] + self.rng.uniform(-0.07, 0.07, (self.N, 2)),
                       self.rng.uniform(-np.pi, np.pi, self.N)]
        self.w = np.full(self.N, 1.0 / self.N)
        self.mode = GLOBAL
        self.n_global_updates = 0
        self.sub_buf = []
        self.baseline = None
        self.cands = None

    def _pf_motion(self, d_xy, d_yaw):
        n = self.N
        c, s = np.cos(self.X[:, 2]), np.sin(self.X[:, 2])
        dx = d_xy[0] * (1 + self.rng.normal(0, 0.05, n)) + self.rng.normal(0, 0.02, n)
        dy = d_xy[1] * (1 + self.rng.normal(0, 0.05, n)) + self.rng.normal(0, 0.02, n)
        self.X[:, 0] += c * dx - s * dy
        self.X[:, 1] += s * dx + c * dy
        self.X[:, 2] += d_yaw + self.rng.normal(0, np.deg2rad(0.8), n)

    def _pf_measure(self, pts_tilted):
        """pts_tilted: body points rotated by roll/pitch only (yaw-free), (M,3)."""
        c, s = np.cos(self.X[:, 2]), np.sin(self.X[:, 2])
        x = c[:, None] * pts_tilted[None, :, 0] - s[:, None] * pts_tilted[None, :, 1] + self.X[:, 0:1]
        y = s[:, None] * pts_tilted[None, :, 0] + c[:, None] * pts_tilted[None, :, 1] + self.X[:, 1:2]
        z = np.broadcast_to(pts_tilted[None, :, 2] + self.m.z_body, x.shape)
        d = self.m.dist(np.stack([x, y, z], -1))
        sig = 0.25
        ll = np.log(0.85 * np.exp(-0.5 * (d / sig) ** 2) + 0.15).sum(1)
        ll /= max(len(pts_tilted) / 15.0, 1.0)            # tempering: points are not independent
        logw = np.log(self.w + 1e-300) + ll
        logw -= logw.max()
        self.w = np.exp(logw)
        self.w /= self.w.sum()

    def _pf_resample(self):
        neff = 1.0 / np.sum(self.w ** 2)
        if neff < self.N / 2:
            pos = (self.rng.random() + np.arange(self.N)) / self.N
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.N - 1)
            self.X = self.X[idx]
            # a few random particles keep the filter able to recover
            k = int(0.02 * self.N)
            j = self.rng.integers(0, len(self.m.free_xy), k)
            self.X[:k] = np.c_[self.m.free_xy[j], self.rng.uniform(-np.pi, np.pi, k)]
            self.w = np.full(self.N, 1.0 / self.N)
        return neff

    def _pf_estimate(self):
        w = self.w
        mx, my = np.sum(w * self.X[:, 0]), np.sum(w * self.X[:, 1])
        cy, sy = np.sum(w * np.cos(self.X[:, 2])), np.sum(w * np.sin(self.X[:, 2]))
        yaw = np.arctan2(sy, cy)
        sxy = np.sqrt(np.sum(w * ((self.X[:, 0] - mx) ** 2 + (self.X[:, 1] - my) ** 2)))
        syaw = np.sqrt(max(-2 * np.log(max(np.hypot(cy, sy), 1e-9)), 0))
        return mx, my, yaw, sxy, syaw

    # ---------------- tracking (ICP)
    def _icp(self, pts_body, T0, iters=12, tukey_c=None):
        """Robust point-to-plane ICP (Gauss-Newton on SE3, Tukey weights).

        Returns the pose, the inlier fraction (point distance < 0.2 m), the inlier
        RMSE and the 3x3 translation block of the Hessian (for degeneracy)."""
        tukey_c = tukey_c or self.tukey_c
        T = T0.copy()
        H = np.zeros((6, 6))
        for it in range(iters):
            dmax = 1.0 if it < 4 else 0.4
            Pw = pts_body @ T[:3, :3].T + T[:3, 3]
            d, idx = self.m.tree.query(Pw, distance_upper_bound=dmax)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            p, q, n = Pw[ok], self.m.P[idx[ok]], self.m.N[idx[ok]]
            r = np.einsum("ij,ij->i", p - q, n)
            w = np.where(np.abs(r) < tukey_c, (1 - (r / tukey_c) ** 2) ** 2, 0.0) if tukey_c < 50 else np.ones_like(r)
            J = np.hstack([np.cross(p, n), n])           # d r / d(dtheta, dt), left perturbation
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
            dT = np.eye(4)
            dT[:3, :3] = Rotation.from_rotvec(dx[:3]).as_matrix()
            dT[:3, 3] = dx[3:]
            T = dT @ T
            if np.linalg.norm(dx) < 1e-4:
                break
        Pw = pts_body @ T[:3, :3].T + T[:3, 3]
        d, _ = self.m.tree.query(Pw, distance_upper_bound=0.2)
        inl = np.isfinite(d)
        fit = float(inl.mean())
        rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf
        self.last_H = H
        return T, fit, rmse

    def _degeneracy(self, *_):
        """Smallest/largest eigenvalue ratio of the translation block of the last ICP Hessian."""
        Ht = self.last_H[3:, 3:]
        ev = np.linalg.eigvalsh(Ht)
        return float(ev[0] / max(ev[-1], 1e-9))

    def _level(self, P_body, R_tilt, iters=60):
        """Refine roll/pitch and body height from the floor plane (RANSAC).

        The LIO world frame is the robot's initial body frame (gravity is a state),
        so its 'tilt' is off by the start-up attitude; a few degrees put far floor
        points into the structure band. Returns (R_tilt_levelled, body_height)."""
        Pt = P_body @ R_tilt.T
        C = Pt[(Pt[:, 2] < -0.3) & (Pt[:, 2] > -2.0)]
        if len(C) < 50:
            return R_tilt, self.m.z_body
        best_n, best_d, best_k = None, None, 0
        for _ in range(iters):
            a, b, c = C[self.rng.choice(len(C), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            if n[2] < 0:
                n = -n
            if n[2] < np.cos(np.deg2rad(15)):
                continue
            d = -n @ a
            k = np.sum(np.abs(C @ n + d) < 0.04)
            if k > best_k:
                best_n, best_d, best_k = n, d, k
        if best_n is None:
            return R_tilt, self.m.z_body
        inl = C[np.abs(C @ best_n + best_d) < 0.04]
        cen = inl.mean(0)
        _, _, Vt = np.linalg.svd(inl - cen)          # least-squares refit
        n = Vt[2] if Vt[2][2] > 0 else -Vt[2]
        v = np.cross(n, [0, 0, 1.0])
        sn, cs = np.linalg.norm(v), n[2]
        R_fix = np.eye(3) if sn < 1e-9 else (np.eye(3) + _skew(v) + _skew(v) @ _skew(v) * (1 - cs) / sn ** 2)
        h = -(R_fix @ cen)[2]                          # body height above the floor
        return R_fix @ R_tilt, float(h)

    def _structure(self, P_tilted_or_world, z_offset):
        """Mask of points in the 0.25-2.8 m band above the floor: columns, walls, stacks.
        Floor and ceiling match everywhere and would drown out the discriminative part."""
        z = P_tilted_or_world[:, 2] + z_offset
        return (z > 0.25) & (z < 2.8)

    def _structure_fit(self, P_body, T, R_tilt=None):
        Pw = P_body @ T[:3, :3].T + T[:3, 3]
        Pw = Pw[self._structure(Pw, 0.0)]
        if len(Pw) < 20:
            return 0.0
        d, _ = self.m.tree.query(Pw, distance_upper_bound=0.2)
        return float(np.isfinite(d).mean())

    # ---------------- global: submap + correlative search + ambiguity test
    def _correlative(self, P_tilted, xy, yaws, sig=0.3, zb=None):
        """Score = mean point likelihood for every (xy, yaw) hypothesis."""
        scores = np.empty((len(yaws), len(xy)))
        for i, a in enumerate(yaws):
            c, s_ = np.cos(a), np.sin(a)
            px = c * P_tilted[:, 0] - s_ * P_tilted[:, 1]
            py = s_ * P_tilted[:, 0] + c * P_tilted[:, 1]
            X = np.stack([xy[:, 0:1] + px[None], xy[:, 1:2] + py[None],
                          np.broadcast_to(P_tilted[None, :, 2] + (self.m.z_body if zb is None else zb),
                                          (len(xy), len(px)))], -1)
            d = self.m.dist(X)
            scores[i] = np.exp(-0.5 * (d / sig) ** 2).mean(1)
        return scores

    def _global_submap(self, T_odom, R_tilt, info):
        Tc_inv = np.linalg.inv(T_odom)
        P = np.concatenate([(Tc_inv @ Tj)[:3, :3] @ pj.T + (Tc_inv @ Tj)[:3, 3:4] for Tj, pj in self.sub_buf], 1).T
        P = voxel(P, 0.3)
        travel = np.linalg.norm((Tc_inv @ self.sub_buf[0][0])[:3, 3])
        info.update(submap_scans=len(self.sub_buf), submap_travel=travel)
        R_tilt, zb = self._level(P, R_tilt)
        info.update(level_height=zb)
        Pt = (P @ R_tilt.T)
        Pt = Pt[self._structure(Pt, zb)]
        if len(Pt) < 30:
            return (0.0, None), (0.0, None), []
        M = Pt[self.rng.choice(len(Pt), min(250, len(Pt)), replace=False)]
        # coarse: 0.5 m x 5 deg over the whole site, wide kernel so near-misses still score
        yaws = np.deg2rad(np.arange(-180, 180, 5.0))
        S = self._correlative(M, self.m.cand_xy, yaws, sig=0.5, zb=zb)
        flat = np.argsort(S, axis=None)[::-1][:60]
        hyps = []
        for f in flat:
            iy, ix = np.unravel_index(f, S.shape)
            hyps.append((self.m.cand_xy[ix], yaws[iy]))
        # fine: +-0.3 m, +-5 deg around each coarse peak
        dxy = np.stack(np.meshgrid(np.arange(-0.3, 0.31, 0.1), np.arange(-0.3, 0.31, 0.1)), -1).reshape(-1, 2)
        fine = []
        for xy0, a0 in hyps:
            fy = a0 + np.deg2rad(np.arange(-5, 5.1, 1.25))
            Sf = self._correlative(M, xy0 + dxy, fy, zb=zb)
            iy, ix = np.unravel_index(np.argmax(Sf), Sf.shape)
            fine.append((Sf[iy, ix], xy0 + dxy[ix], fy[iy]))
        fine.sort(key=lambda h: -h[0])
        # non-max suppression -> distinct modes, verify each with ICP on the submap
        modes = []
        for sc, xy, a in fine:
            if all(np.hypot(*(xy - m_[1])) > 1.5 or abs((a - m_[2] + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(20)
                   for m_ in modes):
                modes.append((sc, xy, a))
            if len(modes) == 5:
                break
        Psub = P if len(P) < 3000 else P[self.rng.choice(len(P), 3000, replace=False)]
        verified = []
        for sc, xy, a in modes:
            T0 = np.eye(4)
            T0[:3, :3] = rz(a) @ R_tilt
            T0[:3, 3] = [xy[0], xy[1], zb]
            T, fit, rmse = self._icp(Psub, T0)
            verified.append((self._structure_fit(Psub, T), T))
        verified.sort(key=lambda v: -v[0])
        if not verified:
            return (0.0, None), (0.0, None), []
        best = verified[0]
        second = next((v for v in verified[1:] if np.linalg.norm(v[1][:2, 3] - best[1][:2, 3]) > 1.0
                       or abs(yaw_of(v[1][:3, :3]) - yaw_of(best[1][:3, :3])) > np.deg2rad(10)), (0.0, None))
        info.update(global_best=best[0], global_second=second[0])
        return best, second, verified

    def _verify_candidates(self, sub, dT_odom, info, min_scans=20, max_scans=60, margin=0.06):
        """Track all candidate poses with odometry + ICP; prune by mean inlier ratio."""
        for c in self.cands:
            T, fit, _ = self._icp(sub, c["T"] @ dT_odom, iters=6)
            c["T"] = T
            c["fits"].append(self._structure_fit(sub, T))
        means = [np.mean(c["fits"][-20:]) for c in self.cands]
        best = int(np.argmax(means))
        keep = [c for c, m_ in zip(self.cands, means) if m_ >= means[best] - margin and m_ > 0.3]
        self.cands = keep
        n = len(self.cands[0]["fits"]) if self.cands else 0
        info.update(n_candidates=len(self.cands))
        if not self.cands:
            self.cands = None                          # all died: search again
            return info
        means = sorted([np.mean(c["fits"][-20:]) for c in self.cands], reverse=True)
        lead = means[0] - (means[1] if len(means) > 1 else 0.0)
        if (len(self.cands) == 1 and n >= min_scans) or (n >= max_scans and lead > 0.02):
            win = max(self.cands, key=lambda c: np.mean(c["fits"][-20:]))
            self.T, self.mode, self.bad = win["T"], TRACK, 0
            self.baseline = float(np.mean(win["fits"][-20:]))
            self.cands = None
            info["handover"] = True
        elif n >= 2 * max_scans:
            self.cands = None                          # still ambiguous: start over with a fresh submap
        return info

    # ---------------- one step per LiDAR scan
    def step(self, pts_body, dT_odom, R_tilt, T_odom=None):
        """pts_body: scan in body frame; dT_odom: odometry increment (body frame);
        R_tilt: body attitude with yaw removed (roll/pitch from the gravity-aligned odometry)."""
        info = {}
        sub = voxel(pts_body, 0.3)
        if self.mode == GLOBAL and self.global_method == "submap":
            self.n_global_updates += 1
            self.sub_buf.append((T_odom.copy(), sub))
            if len(self.sub_buf) > 80:                    # sliding 8 s window
                self.sub_buf.pop(0)
            info.update(fitness=np.nan)
            travel = np.linalg.norm((np.linalg.inv(T_odom) @ self.sub_buf[0][0])[:3, 3])
            if self.cands is not None:
                return self._verify_candidates(sub, dT_odom, info)
            if self.n_global_updates % 5 == 0 and (travel > 2.0 or len(self.sub_buf) >= 40):
                best, second, verified = self._global_submap(T_odom, R_tilt, info)
                # do not trust a single snapshot in a repetitive column grid: keep every
                # plausible mode alive and let the robot's motion disambiguate them
                good = [v for v in verified if v[0] > 0.35]
                if good:
                    self.cands = [dict(T=v[1], fits=[v[0]]) for v in good]
            return info
        if self.mode == GLOBAL:
            d_yaw = yaw_of(dT_odom[:3, :3])
            self._pf_motion(dT_odom[:2, 3], d_yaw)
            R_tilt, zb = self._level(sub, R_tilt)
            St = sub @ R_tilt.T
            St = St[self._structure(St, zb)]
            if len(St) >= 20:
                M = St[self.rng.choice(len(St), min(150, len(St)), replace=False)]
                self._pf_measure(M)
            neff = self._pf_resample()
            self.n_global_updates += 1
            mx, my, yaw, sxy, syaw = self._pf_estimate()
            info.update(neff=neff, spread_xy=sxy, spread_yaw=np.degrees(syaw))
            self.T = np.eye(4)
            self.T[:3, :3] = rz(yaw) @ R_tilt
            self.T[:3, 3] = [mx, my, self.m.z_body]
            if self.n_global_updates >= 5 and sxy < 0.3 and syaw < np.deg2rad(3):
                T, fit, rmse = self._icp(sub, self.T)
                if self._structure_fit(sub, T) > 0.35:
                    self.T, self.mode, self.bad = T, TRACK, 0
                    info["handover"] = True
            info.update(fitness=np.nan)
            return info
        # TRACK
        T_pred = self.T @ dT_odom
        T, fit, rmse = self._icp(sub, T_pred)
        jump = np.linalg.norm(T[:3, 3] - T_pred[:3, 3])
        info.update(fitness=fit, rmse=rmse, jump=jump)
        # health: absolute floor + drop against the running baseline (site changes lower
        # the inlier ratio slowly; a wrong lock or a jump lowers it abruptly)
        floor = 0.35 if self.baseline is None else max(0.35, self.baseline - 0.15)
        healthy = fit > floor and jump < 0.6
        info["fit_floor"] = floor
        if healthy:
            self.baseline = fit if self.baseline is None else 0.97 * self.baseline + 0.03 * fit
        self.bad = 0 if healthy else self.bad + 1
        self.T = T if healthy else T_pred      # coast on odometry while unhealthy
        if self.bad >= 5:
            info["lost"] = True
            self.init_global()
        return info


# ------------------------------------------------------------------ runner
def run(seq, odom_tum, out_dir, map_path, init="gt", kidnap=None, n_particles=3000, t_start=None, seed=0,
        degeneracy=False, tukey_c=0.3, do_deskew=True, global_method="submap", glitch=None):
    seq, out_dir = Path(seq), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_meta(seq)
    R_bl, p_bl = np.array(meta["R_bl"]).reshape(3, 3), np.array(meta["p_bl_b"])
    to, po, qo = read_tum(odom_tum)
    tg, pg, qg = read_tum(seq / "gt_tum.txt")
    pmap = PriorMap(map_path)
    loc = MapLocalizer(pmap, n_particles, seed, tukey_c, global_method)
    glitch_done = False
    rows, poses = [], []
    T_odom_prev = None
    started = False
    odom_reset = False
    t_wall = time.time()
    for t0, s in iter_scans(seq):
        t = t0 + 0.098
        k = np.searchsorted(to, t)
        if k >= len(to) or (k > 0 and abs(to[k - 1] - t) < abs(to[k] - t)):
            k -= 1
        if k < 0 or abs(to[k] - t) > 0.06:
            continue
        if t_start is not None and t < t_start:
            continue
        T_odom = pose_mat(po[k], qo[k])
        kidnapped = kidnap is not None and kidnap[0] <= t < kidnap[0] + kidnap[1]
        if kidnapped:
            odom_reset = True                          # localization offline (robot carried, reboot...)
            continue
        known_gap = odom_reset and T_odom_prev is not None
        if T_odom_prev is None or odom_reset:
            dT = np.eye(4)                             # odometry restarted: no motion info across the gap
            odom_reset = False
        else:
            dT = np.linalg.inv(T_odom_prev) @ T_odom
        if glitch is not None and not glitch_done and t >= glitch[0]:
            # silent odometry failure: a bogus jump the localizer is not told about
            G = np.eye(4)
            G[:3, :3] = rz(np.deg2rad(glitch[2]))
            G[0, 3] = glitch[1]
            dT = dT @ G
            glitch_done = True
        T_odom_prev = T_odom
        R_tilt = rz(-yaw_of(T_odom[:3, :3])) @ T_odom[:3, :3]
        pts = deskew(s, R_bl, p_bl, dT) if do_deskew else scan_to_body(s, R_bl, p_bl)
        if not started:
            started = True
            if init == "gt":
                kk = np.searchsorted(tg, t)
                loc.T = pose_mat(pg[kk], qg[kk])
                loc.mode = TRACK
            else:
                loc.init_global()
        if known_gap:
            loc.init_global()                          # the system knows localization was interrupted
        info = loc.step(pts, dT, R_tilt, T_odom)
        kg = min(np.searchsorted(tg, t), len(tg) - 1)
        Tg = pose_mat(pg[kg], qg[kg])
        e_xy = np.linalg.norm(loc.T[:2, 3] - Tg[:2, 3])
        e_z = abs(loc.T[2, 3] - Tg[2, 3])
        e_yaw = np.degrees(abs((yaw_of(loc.T[:3, :3]) - yaw_of(Tg[:3, :3]) + np.pi) % (2 * np.pi) - np.pi))
        deg = loc._degeneracy(voxel(pts, 0.3), loc.T) if (degeneracy and loc.mode == TRACK) else np.nan
        rows.append([t, 1 if loc.mode == TRACK else 0, info.get("fitness", np.nan), info.get("rmse", np.nan),
                     info.get("jump", np.nan), info.get("spread_xy", np.nan), int(info.get("lost", False)),
                     int(info.get("handover", False)), e_xy, e_z, e_yaw, deg])
        q = np.empty(4)
        import mujoco
        mujoco.mju_mat2Quat(q, loc.T[:3, :3].reshape(-1).copy())
        poses.append([t, *loc.T[:3, 3], q[1], q[2], q[3], q[0]])
    a = np.array(rows)
    hdr = "t,tracking,fitness,rmse,jump,spread_xy,lost,handover,err_xy,err_z,err_yaw_deg,degeneracy"
    np.savetxt(out_dir / "maploc_log.csv", a, delimiter=",", header=hdr, comments="", fmt="%.5f")
    np.savetxt(out_dir / "poses_tum.txt", np.array(poses), fmt="%.6f")
    tr = a[:, 1] > 0
    good = tr & (a[:, 8] < 0.5)
    res = dict(n=len(a), wall_s=time.time() - t_wall, tracking_frac=float(tr.mean()),
               err_xy_rmse_tracking=float(np.sqrt(np.mean(a[tr, 8] ** 2))) if tr.any() else None,
               err_xy_p95_tracking=float(np.percentile(a[tr, 8], 95)) if tr.any() else None,
               err_yaw_rmse_tracking=float(np.sqrt(np.mean(a[tr, 10] ** 2))) if tr.any() else None,
               fitness_median=float(np.nanmedian(a[tr, 2])) if tr.any() else None,
               n_lost=int(a[:, 6].sum()), n_handover=int(a[:, 7].sum()),
               false_track_frac=float((tr & (a[:, 8] > 1.0)).mean()))
    if init == "global" or kidnap:
        # (with a known gap the localizer re-enters global mode on resume)
        t0 = a[0, 0] if init == "global" else kidnap[0] + kidnap[1]
        after = a[a[:, 0] >= t0]
        first_good = after[(after[:, 1] > 0) & (after[:, 8] < 0.3)]
        res["time_to_localize_s"] = float(first_good[0, 0] - t0) if len(first_good) else None
    if glitch:
        after = a[a[:, 0] >= glitch[0]]
        lost = after[after[:, 6] > 0]
        res["lost_detect_delay_s"] = float(lost[0, 0] - glitch[0]) if len(lost) else None
        good = after[(after[:, 1] > 0) & (after[:, 8] < 0.3) & (after[:, 0] > (lost[0, 0] if len(lost) else 1e9))]
        res["time_to_relocalize_s"] = float(good[0, 0] - glitch[0]) if len(good) else None
        res["max_err_after_glitch"] = float(after[:, 8].max())
    (out_dir / "maploc_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-map")
    b.add_argument("seq")
    b.add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("seq")
    r.add_argument("odom")
    r.add_argument("out")
    r.add_argument("--map", required=True)
    r.add_argument("--init", default="gt", choices=["gt", "global"])
    r.add_argument("--kidnap", default=None, help="t_start,duration")
    r.add_argument("--t-start", type=float, default=None)
    r.add_argument("--particles", type=int, default=3000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--degeneracy", action="store_true")
    r.add_argument("--tukey", type=float, default=0.3, help=">=50 means plain least squares")
    r.add_argument("--no-deskew", action="store_true")
    r.add_argument("--global-method", default="submap", choices=["submap", "pf"])
    r.add_argument("--glitch", default=None, help="t,dx_m,dyaw_deg silent odometry jump")
    a = ap.parse_args()
    if a.cmd == "build-map":
        build_map(a.seq, a.out)
    else:
        kid = tuple(float(x) for x in a.kidnap.split(",")) if a.kidnap else None
        print(json.dumps(run(a.seq, a.odom, a.out, a.map, a.init, kid, a.particles, a.t_start, a.seed,
                             a.degeneracy, a.tukey, not a.no_deskew, a.global_method,
                             tuple(float(x) for x in a.glitch.split(',')) if a.glitch else None), indent=2))


if __name__ == "__main__":
    main()
