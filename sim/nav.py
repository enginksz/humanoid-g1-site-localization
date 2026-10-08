"""Map-based navigation for the G1: online LiDAR map -> costmap -> navigation function + MPPI.

Everything here uses only robot-side information: the head LiDAR scans (with Doppler) and
a pose source (ground truth in open loop, the live estimator in closed loop). Nothing is
read from the simulator's geometry.

  OnlineMapper   voxel hit/miss map updated per scan (tools/elevmap.py), Doppler dynamic-point
                 removal, 2.5D layers in a robot-centred window
  DynTracker     Doppler-flagged points -> clusters -> constant-velocity Kalman tracks
  NavFunction    Dijkstra cost-to-go from the goal over the window costmap (8-connected).
                 A navigation function has no local minima, so U-traps are escaped.
  MPPI           sampling MPC on a unicycle (vx, wz) = the walking policy's command interface.
                 Running cost: costmap + predicted track positions + control effort;
                 terminal cost: navigation function. Warm-started, 20 Hz.
"""
from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.elevmap import BAND_HI, BAND_LO, doppler_dynamic_mask  # noqa: E402


# ---------------------------------------------------------------- online map
class OnlineMapper:
    def __init__(self, lo, hi, res=0.15, dyn="doppler", raycast=False, max_range=20.0, seed=0,
                 persist=0.0, recent=0.35):
        self.res = res
        self.lo = np.asarray(lo, float)
        self.dims = np.ceil((np.asarray(hi) - self.lo) / res).astype(int) + 1
        n = int(np.prod(self.dims))
        self.hits = np.zeros(n, np.uint16)
        self.miss = np.zeros(n, np.uint16)
        # running sum of per-scan mean z in each voxel -> sub-voxel ground height. Without it the
        # floor straddles a voxel boundary under a few cm of pose z error and shows up as 15 cm "steps"
        self.zsum = np.zeros(n, np.float32)
        self.dyn = dyn
        self.raycast = raycast
        self.max_range = max_range
        self.rng = np.random.default_rng(seed)
        self.last_dyn_pts = np.zeros((0, 3))
        self.n_scans = 0
        # layered map: persistent static layer + short-lived "recent" layer
        self.persist_scans = int(round(persist / 0.1))
        self.first = np.full(n, 65535, np.uint16)
        self.last = np.zeros(n, np.uint16)
        self.recent_n = max(1, int(round(recent / 0.1)))
        self.recent = []                         # world points of the last few scans

    def _flat(self, P):
        k = np.floor((P - self.lo) / self.res).astype(np.int64)
        ok = np.all((k >= 0) & (k < self.dims), 1)
        k = k[ok]
        return np.unique(k[:, 0] * self.dims[1] * self.dims[2] + k[:, 1] * self.dims[2] + k[:, 2])

    def integrate(self, pts, p_lw, R_wl_fn):
        """pts (n,5) LiDAR frame [x y z doppler t_off]; R_wl_fn(t_off) -> (p_lw (n,3), R_wl (n,3,3))."""
        pl = pts[:, :3].astype(np.float64)
        r = np.linalg.norm(pl, axis=1)
        keep = (r > 0.5) & (r < self.max_range)
        pl, dop, toff = pl[keep], pts[keep, 3].astype(np.float64), pts[keep, 4].astype(np.float64)
        if len(pl) < 50:
            return
        P0, R = R_wl_fn(toff)
        pw = P0 + np.einsum("nij,nj->ni", R, pl)
        drop = np.zeros(len(pw), bool)
        if self.dyn == "doppler":
            drop, _ = doppler_dynamic_mask(pl, dop, pw, self.rng, t=toff)
        self.last_dyn_pts = pw[drop]
        P = pw[~drop]
        kk = np.floor((P - self.lo) / self.res).astype(np.int64)
        okk = np.all((kk >= 0) & (kk < self.dims), 1)
        fl = kk[okk, 0] * self.dims[1] * self.dims[2] + kk[okk, 1] * self.dims[2] + kk[okk, 2]
        idx, inv = np.unique(fl, return_inverse=True)
        zmean = np.bincount(inv, weights=P[okk, 2]) / np.bincount(inv)
        sat = self.hits[idx] >= 65000
        self.zsum[idx[~sat]] += zmean[~sat].astype(np.float32)
        self.hits[idx] = np.minimum(self.hits[idx].astype(np.uint32) + 1, 65000).astype(np.uint16)
        k = min(self.n_scans, 65534)
        self.first[idx] = np.minimum(self.first[idx], k)
        self.last[idx] = k
        if self.persist_scans or self.dyn == "doppler":
            # recent layer: everything seen in the last ~0.35 s, INCLUDING the Doppler-removed people,
            # is an obstacle now (never stored). Removing people from the map must not remove them
            # from collision checking.
            self.recent.append(pw)
            self.recent = self.recent[-self.recent_n:]
        if self.raycast:
            o = P0[len(P0) // 2]
            Q = P[::3]
            d = Q - o
            L = np.linalg.norm(d, axis=1)
            ok = L > 0.3 + self.res
            d, L = d[ok] / L[ok, None], L[ok] - 0.3
            n = np.ceil(L / self.res).astype(int)
            if n.sum():
                ray = np.repeat(np.arange(len(L)), n)
                step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
                mi = self._flat(o + d[ray] * (step[:, None] * self.res))
                mi = mi[~np.isin(mi, idx, assume_unique=True)]     # never miss a voxel hit this scan
                self.miss[mi] = np.minimum(self.miss[mi].astype(np.uint32) + 1, 65000).astype(np.uint16)
        self.n_scans += 1

    def window(self, cx, cy, half, z_ref=None):
        """2.5D layers in the xy window [cx-half, cx+half]^2 (map cells).

        z_ref: the robot's ground height. Ground is searched only in [z_ref-1.5, z_ref+1.0]
        so an overhead slab seen where the floor is not (yet) observed is not taken as ground.
        """
        nx, ny, nz = self.dims
        i0 = int(np.clip((cx - half - self.lo[0]) / self.res, 0, nx - 1))
        i1 = int(np.clip((cx + half - self.lo[0]) / self.res, 1, nx))
        j0 = int(np.clip((cy - half - self.lo[1]) / self.res, 0, ny - 1))
        j1 = int(np.clip((cy + half - self.lo[1]) / self.res, 1, ny))
        H = self.hits.reshape(self.dims)[i0:i1, j0:j1].astype(np.float32)
        M = self.miss.reshape(self.dims)[i0:i1, j0:j1].astype(np.float32)
        occ = (H >= 2) & (H / np.maximum(H + M, 1) > 0.5)
        if self.persist_scans:
            F = self.first.reshape(self.dims)[i0:i1, j0:j1].astype(np.int32)
            Lst = self.last.reshape(self.dims)[i0:i1, j0:j1].astype(np.int32)
            occ_ground = occ                      # ground needs no persistence
            occ = occ & ((Lst - F) >= self.persist_scans)
        else:
            occ_ground = occ
        zi = np.arange(nz)[None, None, :]
        if z_ref is None:
            gsel = occ_ground
        else:
            k_lo = int((z_ref - 1.5 - self.lo[2]) / self.res)
            k_hi = int((z_ref + 1.0 - self.lo[2]) / self.res)
            gsel = occ_ground & (zi >= k_lo) & (zi <= k_hi)
        has_g = gsel.any(2)
        first = np.argmax(gsel, axis=2)
        Zs = self.zsum.reshape(self.dims)[i0:i1, j0:j1]
        hf = np.take_along_axis(H, first[..., None], 2)[..., 0]
        ground = np.take_along_axis(Zs, first[..., None], 2)[..., 0] / np.maximum(hf, 1)
        ground[~has_g] = np.nan
        # obstacle band above the cell's ground (or above the robot's ground if not seen)
        k_ref = first if z_ref is None else np.where(has_g, first, int((z_ref - self.lo[2]) / self.res))
        band = (zi >= k_ref[..., None] + int(round(BAND_LO / self.res))) & \
               (zi <= k_ref[..., None] + int(round(BAND_HI / self.res)))
        obst = (occ & band).any(2)
        # recent layer: anything in the body band seen in the last ~0.3 s (crossing people)
        if self.recent and z_ref is not None:
            R = np.concatenate(self.recent)
            ii = ((R[:, 0] - self.lo[0]) / self.res).astype(int) - i0
            jj = ((R[:, 1] - self.lo[1]) / self.res).astype(int) - j0
            ok = (ii >= 0) & (jj >= 0) & (ii < obst.shape[0]) & (jj < obst.shape[1])
            gz = np.where(has_g, ground, z_ref)
            ii, jj, zz = ii[ok], jj[ok], R[ok, 2]
            inb = (zz > gz[ii, jj] + BAND_LO) & (zz < gz[ii, jj] + BAND_HI)
            obst[ii[inb], jj[inb]] = True
        observed = has_g | obst
        origin = self.lo[:2] + np.array([i0, j0]) * self.res
        return dict(origin=origin, res=self.res, ground=ground, obstacle=obst, observed=observed)


# ---------------------------------------------------------------- dynamic tracks
@dataclass
class Track:
    x: np.ndarray                 # [x, y, vx, vy]
    P: np.ndarray
    radius: float
    t_seen: float
    hits: int = 1
    id: int = 0


class DynTracker:
    def __init__(self, gate=1.5, q=1.5, r=0.15):
        self.tracks: list[Track] = []
        self.gate, self.q, self.r = gate, q, r
        self.t = None
        self._next = 0

    def update(self, t, dyn_pts):
        if self.t is not None:
            dt = t - self.t
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            Q = self.q * np.diag([dt ** 3 / 3, dt ** 3 / 3, dt, dt])
            for tr in self.tracks:
                tr.x = F @ tr.x
                tr.P = F @ tr.P @ F.T + Q
        self.t = t
        dets = []
        if len(dyn_pts) >= 5:
            g = np.floor(dyn_pts[:, :2] / 0.3).astype(int)
            g -= g.min(0)
            img = np.zeros(g.max(0) + 1, bool)
            img[g[:, 0], g[:, 1]] = True
            lab, n = cc_label(img, np.ones((3, 3)))
            pl = lab[g[:, 0], g[:, 1]]
            for k in range(1, n + 1):
                P = dyn_pts[pl == k]
                if len(P) < 5:
                    continue
                c = P[:, :2].mean(0)
                rad = float(np.percentile(np.linalg.norm(P[:, :2] - c, axis=1), 90)) + 0.1
                dets.append((c, min(max(rad, 0.3), 3.0)))
        used = set()
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        Rm = self.r ** 2 * np.eye(2)
        for tr in sorted(self.tracks, key=lambda tr: -tr.hits):
            best, bd = None, self.gate
            for i, (c, rad) in enumerate(dets):
                if i in used:
                    continue
                dd = np.linalg.norm(c - tr.x[:2])
                if dd < bd:
                    best, bd = i, dd
            if best is None:
                continue
            used.add(best)
            c, rad = dets[best]
            S = H @ tr.P @ H.T + Rm
            K = tr.P @ H.T @ np.linalg.inv(S)
            tr.x = tr.x + K @ (c - H @ tr.x)
            tr.P = (np.eye(4) - K @ H) @ tr.P
            tr.radius = 0.7 * tr.radius + 0.3 * rad
            tr.t_seen, tr.hits = t, tr.hits + 1
        for i, (c, rad) in enumerate(dets):
            if i not in used:
                self.tracks.append(Track(np.array([c[0], c[1], 0, 0.0]), np.diag([0.1, 0.1, 2.0, 2.0]),
                                         rad, t, id=self._next))
                self._next += 1
        self.tracks = [tr for tr in self.tracks if t - tr.t_seen < 0.8]

    def predicted(self, horizon, dt):
        """(n_tracks, T, 2) predicted centres and (n_tracks,) radii (confirmed tracks only)."""
        trs = [tr for tr in self.tracks if tr.hits >= 2]
        if not trs:
            return np.zeros((0, horizon, 2)), np.zeros(0)
        ts = (np.arange(horizon) + 1) * dt
        C = np.stack([tr.x[:2][None] + ts[:, None] * np.clip(tr.x[2:4], -3, 3)[None] for tr in trs])
        return C, np.array([tr.radius for tr in trs])


# ---------------------------------------------------------------- costmap + navigation function
@dataclass
class CostCfg:
    robot_radius: float = 0.30
    inflate: float = 0.6           # cost decays to 0 over this distance beyond the radius
    unknown_cost: float = 0.15
    max_slope_deg: float = 14.0    # lethal (area-averaged); tuned on the open site: graded road p95 = 6 deg
    plan_res: float = 0.15


def build_costmap(win, cfg: CostCfg):
    """Cell cost in [0, 1]; 1 = lethal. Obstacles inflated by the robot radius."""
    obst = win["obstacle"].copy()
    g = win["ground"]
    # slope from the mapped ground (only between observed neighbours)
    gx = np.abs(np.diff(g, axis=0, append=np.nan))
    gy = np.abs(np.diff(g, axis=1, append=np.nan))
    step = np.fmax(np.nan_to_num(gx, nan=0), np.nan_to_num(gy, nan=0))
    # slope from a local plane (ground smoothed over 0.75 m), then averaged over ~1 m: a single
    # cell cannot tell a graded road from rough ground (true road slope p99 = 8.9 deg, off-road
    # median 6.0 deg), the area average can (road p95 6.0 deg, off-road median 6.0 deg)
    from scipy.ndimage import uniform_filter
    m = np.isfinite(g).astype(float)
    mk = uniform_filter(m, 5)
    gs = uniform_filter(np.nan_to_num(g), 5) / np.maximum(mk, 1e-6)
    gs[mk < 0.3] = np.nan
    sx = np.zeros_like(g)
    sy = np.zeros_like(g)
    sx[1:-1, :] = (gs[2:, :] - gs[:-2, :]) / (2 * win["res"])
    sy[:, 1:-1] = (gs[:, 2:] - gs[:, :-2]) / (2 * win["res"])
    sl = np.degrees(np.arctan(np.hypot(np.nan_to_num(sx), np.nan_to_num(sy))))
    sl_ok = np.isfinite(sx) & np.isfinite(sy)
    slope_deg = uniform_filter(np.where(sl_ok, sl, 0.0), 7) / np.maximum(uniform_filter(sl_ok.astype(float), 7), 1e-6)
    obst |= step > 0.25                                 # a step the robot cannot take
    d = distance_transform_edt(~obst) * win["res"]
    cost = np.clip(1.0 - (d - cfg.robot_radius) / cfg.inflate, 0, 1) ** 2
    cost[d <= cfg.robot_radius] = 1.0
    # traversability for the walking policy: free below 3 deg, rising cost, lethal from max_slope_deg
    slope_c = np.clip((slope_deg - 3.0) / (cfg.max_slope_deg - 3.0), 0, 1)
    cost = np.maximum(cost, maximum_filter(slope_c, size=3))
    cost[~win["observed"]] = np.maximum(cost[~win["observed"]], cfg.unknown_cost)
    return cost


def nav_function(cost, res, goal_xy, origin, plan_res=0.15, lethal=0.95, start_xy=None):
    """Dijkstra cost-to-go [m-equivalent] from the goal over the costmap (8-connected).

    Runs at the costmap resolution with scipy's C Dijkstra: a coarser, max-pooled grid closes
    1 m gaps (robot diameter + inflation leave one cell), which was observed in closed loop.
    Returns (field, origin, res)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    f = max(1, int(round(plan_res / res)))
    nx, ny = cost.shape[0] // f, cost.shape[1] // f
    C = cost[:nx * f, :ny * f].reshape(nx, f, ny, f).max(axis=(1, 3)).copy()
    pres = res * f
    if start_xy is not None:      # the robot's own cells are never lethal (escape when inflated)
        si = ((np.asarray(start_xy) - origin) / pres).astype(int)
        r = max(1, int(round(0.3 / pres)))
        sl = (slice(max(si[0] - r, 0), si[0] + r + 1), slice(max(si[1] - r, 0), si[1] + r + 1))
        C[sl] = np.minimum(C[sl], 0.9)
    gi = np.clip(((goal_xy - origin) / pres).astype(int), 0, [nx - 1, ny - 1])
    if C[gi[0], gi[1]] >= lethal:   # goal outside / blocked: nearest free cell to the true goal
        free = np.argwhere(C < lethal)
        if len(free) == 0:
            return None, origin, pres
        cc = origin + (free + 0.5) * pres
        gi = free[np.argmin(np.linalg.norm(cc - goal_xy, axis=1))]
    idx = np.arange(nx * ny).reshape(nx, ny)
    ok = C < lethal
    rows, cols, w = [], [], []
    for di, dj, L in [(1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.414), (1, -1, 1.414)]:
        a = (slice(0, nx - di), slice(max(-dj, 0), ny - max(dj, 0)))
        b = (slice(di, nx), slice(max(dj, 0), ny + min(dj, 0)))
        m = ok[a] & ok[b]
        wa = L * pres * (1.0 + 6.0 * 0.5 * (C[a] + C[b]))
        rows.append(idx[a][m]); cols.append(idx[b][m]); w.append(wa[m])
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                   shape=(nx * ny, nx * ny)).tocsr()
    D = dijkstra(G, directed=False, indices=int(idx[gi[0], gi[1]])).reshape(nx, ny)
    if start_xy is not None:
        si = np.clip(((np.asarray(start_xy) - origin) / pres).astype(int), 0, [nx - 1, ny - 1])
        if not np.isfinite(D[si[0], si[1]]):
            # selected goal cell is not reachable from the robot (e.g. an isolated free pocket at the
            # window edge): sub-goal = the REACHABLE cell closest to the true goal
            R = dijkstra(G, directed=False, indices=int(idx[si[0], si[1]])).reshape(nx, ny)
            reach = np.argwhere(np.isfinite(R))
            if len(reach) > 1:
                cc = origin + (reach + 0.5) * pres
                gi = reach[np.argmin(np.linalg.norm(cc - goal_xy, axis=1))]
                D = dijkstra(G, directed=False, indices=int(idx[gi[0], gi[1]])).reshape(nx, ny)
    gxy = origin + (gi + 0.5) * pres
    D += np.linalg.norm(gxy - goal_xy)
    return D, origin, pres


def descent_field(D):
    """Unit direction towards the lowest-cost 8-neighbour of every cell (discrete steepest descent).

    np.gradient over a field with lethal (inf) cells points away from walls rather than along the
    path; in a narrow corridor every cell borders a wall, so the gradient steered the guide sample
    and the heading critic into the walls."""
    nx, ny = D.shape
    Dp = np.pad(np.where(np.isfinite(D), D, np.inf), 1, constant_values=np.inf)
    best = Dp[1:-1, 1:-1].copy()
    gx = np.zeros_like(D)
    gy = np.zeros_like(D)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = Dp[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]
            better = nb < best
            best = np.where(better, nb, best)
            n = np.hypot(di, dj)
            gx = np.where(better, di / n, gx)
            gy = np.where(better, dj / n, gy)
    return gx, gy


def carrot(D, origin, pres, xy, dist=1.2):
    """Point ~dist metres down the navigation function from xy (None if unavailable)."""
    p = extract_path(D, origin, pres, np.asarray(xy, float), n_max=int(dist / pres) + 2)
    return p[-1] if len(p) >= 2 else None


def extract_path(D, origin, pres, start_xy, n_max=200):
    i, j = ((start_xy - origin) / pres).astype(int)
    nx, ny = D.shape
    path = []
    for _ in range(n_max):
        if not (0 <= i < nx and 0 <= j < ny):
            break
        path.append(origin + (np.array([i, j]) + 0.5) * pres)
        best, bv = None, D[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < nx and 0 <= b < ny and D[a, b] < bv:
                    best, bv = (a, b), D[a, b]
        if best is None:
            break
        i, j = best
    return np.array(path)


# ---------------------------------------------------------------- MPPI
@dataclass
class MPPICfg:
    K: int = 384
    T: int = 25
    dt: float = 0.1
    lam: float = 0.08
    sigma_v: float = 0.30
    sigma_w: float = 0.55
    v_max: float = 0.8
    v_min: float = -0.15
    w_max: float = 1.0
    w_obs: float = 1.0            # proximity is already priced in the navigation function (tie-breaker only)
    w_dyn: float = 200.0
    w_goal: float = 5.0
    w_ctrl: float = 0.05
    w_turn: float = 0.15
    w_head: float = 1.5           # terminal heading critic: align with the navigation function's descent
    tau_v: float = 0.25           # first-order lag of the walking policy's velocity tracking


class MPPI:
    def __init__(self, cfg: MPPICfg | None = None, seed=0):
        self.c = cfg or MPPICfg()
        self.U = np.zeros((self.c.T, 2))
        self.rng = np.random.default_rng(seed)
        self.last_rollouts = None
        self.last_best = None

    def _guide(self, x0, v0, desc, d_origin, d_res):
        """Deterministic rollout that follows the navigation function's descent direction.
        Random samples rarely thread a one-cell corridor for a whole horizon; this one does, so
        MPPI always has at least one feasible trajectory in narrow passages."""
        c = self.c
        x = np.array(x0, float)
        U = np.zeros((c.T, 2))
        for t in range(c.T):
            gx = self._lookup(desc[0], d_origin, d_res, np.array([x[0]]), np.array([x[1]]), 0.0)[0]
            gy = self._lookup(desc[1], d_origin, d_res, np.array([x[0]]), np.array([x[1]]), 0.0)[0]
            if gx == 0 and gy == 0:
                break
            e = (np.arctan2(gy, gx) - x[2] + np.pi) % (2 * np.pi) - np.pi
            w = float(np.clip(2.0 * e, -c.w_max, c.w_max))
            v = float(np.clip(0.5 * c.v_max * max(np.cos(e), 0.0), 0.0, c.v_max))
            U[t] = [v, w]
            x[2] += w * c.dt
            x[0] += v * np.cos(x[2]) * c.dt
            x[1] += v * np.sin(x[2]) * c.dt
        return U

    def _lookup(self, grid, origin, res, X, Y, fill):
        i = ((X - origin[0]) / res).astype(int)
        j = ((Y - origin[1]) / res).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < grid.shape[0]) & (j < grid.shape[1])
        out = np.full(X.shape, fill, float)
        out[ok] = grid[i[ok], j[ok]]
        return out

    def command(self, x0, v0, cost, c_origin, c_res, D, d_origin, d_res, tracks_C, tracks_r, desc=None):
        c = self.c
        self.U = np.roll(self.U, -1, 0)
        self.U[-1] = self.U[-2]
        eps = self.rng.normal(0, 1, (c.K, c.T, 2)) * [c.sigma_v, c.sigma_w]
        eps[0] = 0                                     # keep the warm-started mean as a sample
        eps[1] = -self.U                               # and a "stop" sample
        if desc is not None:                           # and a guide sample down the navigation function
            eps[2] = self._guide(x0, v0, desc, d_origin, d_res) - self.U
        V = np.clip(self.U[None] + eps, [c.v_min, -c.w_max], [c.v_max, c.w_max])
        x = np.repeat(np.asarray(x0, float)[None], c.K, 0)
        v = np.full(c.K, float(v0))
        S = np.zeros(c.K)
        Xs = np.zeros((c.K, c.T, 2))
        a = c.dt / (c.tau_v + c.dt)
        c0 = float(self._lookup(cost, c_origin, c_res, np.array([x0[0]]), np.array([x0[1]]), 0.0)[0])
        lethal = max(0.99, c0 + 0.01)          # already inside inflation: only forbid going deeper
        Dmax = np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 50.0
        for t in range(c.T):
            v = v + a * (V[:, t, 0] - v)
            x[:, 2] += V[:, t, 1] * c.dt
            x[:, 0] += v * np.cos(x[:, 2]) * c.dt
            x[:, 1] += v * np.sin(x[:, 2]) * c.dt
            Xs[:, t] = x[:, :2]
            cm = self._lookup(cost, c_origin, c_res, x[:, 0], x[:, 1], 0.15)
            # running cost in the navigation function's own metric (edge cost = length * 6 * C), so a
            # shortcut through expensive ground costs what it costs in the global plan; + lethal
            S += c.w_goal * 6.0 * cm * np.abs(v) * c.dt + c.w_obs * cm ** 2 + 1e4 * (cm >= lethal)
            if len(tracks_r):
                dd = np.linalg.norm(x[:, None, :2] - tracks_C[None, :, t], axis=2) - tracks_r[None] - 0.30
                S += c.w_dyn * np.sum(np.clip(1.0 - dd / 0.8, 0, None) ** 2, 1) + 1e4 * np.any(dd < 0, 1)
            S += c.w_ctrl * (V[:, t, 0] ** 2) + c.w_turn * V[:, t, 1] ** 2
        Dt = self._lookup(np.where(np.isfinite(D), D, Dmax + 5), d_origin, d_res, x[:, 0], x[:, 1], Dmax + 5)
        S += c.w_goal * Dt
        if desc is not None:       # (gx, gy) unit descent direction grids of the navigation function
            gx = self._lookup(desc[0], d_origin, d_res, x[:, 0], x[:, 1], 0.0)
            gy = self._lookup(desc[1], d_origin, d_res, x[:, 0], x[:, 1], 0.0)
            S += c.w_head * (1.0 - (np.cos(x[:, 2]) * gx + np.sin(x[:, 2]) * gy))
        S -= S.min()
        # temperature from the FEASIBLE samples only: in a narrow passage most samples are lethal
        # (+1e4); a median over all samples then flattens the weights and the output becomes the
        # average of random noise (the robot dithers in place)
        feas = S < 5e3
        scale = np.median(S[feas]) if feas.sum() >= 5 else np.median(S)
        w = np.exp(-S / max(c.lam * scale, 0.05))
        w /= w.sum()
        self.U = np.clip(self.U + np.einsum("k,ktc->tc", w, V - self.U[None]),
                         [c.v_min, -c.w_max], [c.v_max, c.w_max])
        self.last_rollouts = Xs[np.argsort(S)[:24]]
        self.last_best = Xs[np.argmin(S)]
        return self.U[0].copy()


# ---------------------------------------------------------------- planner facade
class MapNavigator:
    """What the recorder calls: integrate scans, then ask for a (vx, wz) command."""

    def __init__(self, lo, hi, v_max=0.8, dyn="doppler", raycast=False, window=12.0, seed=0, persist=0.0):
        self.mapper = OnlineMapper(lo, hi, dyn=dyn, raycast=raycast, seed=seed, persist=persist)
        self.tracker = DynTracker()
        self.cfg = CostCfg()
        self.mppi = MPPI(MPPICfg(v_max=v_max), seed=seed)
        self.half = window
        self.D = None
        self.t_plan = -1e9
        self.goal = None
        self.v_est = 0.0
        self.last = {}
        self.timing = []
        self.shim_dir = 0

    def on_scan(self, t, pts, pose_fn):
        self.mapper.integrate(pts, None, pose_fn)
        if self.mapper.dyn == "doppler":
            self.tracker.update(t, self.mapper.last_dyn_pts)

    def command(self, t, xy, yaw, goal, z_ground=None):
        import time
        t0 = time.perf_counter()
        win = self.mapper.window(xy[0], xy[1], self.half, z_ref=z_ground)
        cost = build_costmap(win, self.cfg)
        replan = self.D is None or t - self.t_plan > 1.0 or self.goal is None or \
            np.linalg.norm(goal - self.goal) > 1e-6
        if replan:
            self.D, self.d_origin, self.d_res = nav_function(cost, win["res"], goal, win["origin"],
                                                             self.cfg.plan_res, start_xy=xy)
            self.t_plan, self.goal = t, goal.copy()
        if self.D is None:
            return 0.0, 0.0
        if replan:
            self.desc = descent_field(self.D)
        C, r = self.tracker.predicted(self.mppi.c.T, self.mppi.c.dt)
        # rotation shim: if the way to go is far off the current heading, turn in place first
        # (a 180 deg turn takes longer than the MPPI horizon, so MPPI alone would stall)
        cp = carrot(self.D, self.d_origin, self.d_res, xy)
        if cp is not None:
            err = (np.arctan2(cp[1] - xy[1], cp[0] - xy[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
            # hysteresis + locked turn direction: with the path behind the robot the error sits
            # near +-180 deg and its sign flips every tick, which makes the robot dither in place
            if self.shim_dir == 0 and abs(err) > 1.05:
                self.shim_dir = 1 if err > 0 else -1
            elif self.shim_dir != 0 and abs(err) < 0.45:
                self.shim_dir = 0
            if self.shim_dir != 0:
                if abs(err) > 2.6:                      # near 180: keep the committed direction
                    err = self.shim_dir * abs(err)
                self.mppi.U[:] = 0.0
                self.v_est = 0.0
                self.last = dict(win=win, cost=cost, D=self.D, d_origin=self.d_origin, d_res=self.d_res,
                                 rollouts=self.mppi.last_rollouts, best=self.mppi.last_best, tracks=(C, r))
                self.timing.append(time.perf_counter() - t0)
                return 0.0, float(np.clip(1.5 * err, -self.mppi.c.w_max, self.mppi.c.w_max))
        u = self.mppi.command([xy[0], xy[1], yaw], self.v_est, cost, win["origin"], win["res"],
                              self.D, self.d_origin, self.d_res, C, r, desc=self.desc)
        self.v_est = float(u[0])
        self.last = dict(win=win, cost=cost, D=self.D, d_origin=self.d_origin, d_res=self.d_res,
                         rollouts=self.mppi.last_rollouts, best=self.mppi.last_best, tracks=(C, r))
        self.timing.append(time.perf_counter() - t0)
        return float(u[0]), float(u[1])
