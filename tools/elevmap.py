"""Online LiDAR mapping for navigation: voxel occupancy -> 2.5D elevation + obstacle layer.

The map is built only from what the robot has: the head FMCW LiDAR scans (with per-point
Doppler and firing time) and an odometry pose stream (GT or any estimator's poses_tum.txt).

Pipeline per scan
  1. per-point pose: body pose interpolated at the firing time t0 + t_off (deskew) and
     LiDAR extrinsics -> world points
  2. dynamic-point handling (one of)
       none      every return is integrated (classic accumulate-only map)
       raycast   free-space carving: voxels a beam passes through get a miss
                 (OctoMap / elevation_mapping visibility cleanup)
       doppler   FMCW: ego-velocity fitted from the scan's own Doppler (RANSAC),
                 points whose Doppler disagrees with a static world are dropped,
                 then grown to the whole object (catches the tangential parts)
       doppler+raycast, oracle (GT actor boxes; reference only)
  3. hit / miss counts in a dense voxel grid (res 0.15 m)
Products (per xy cell): ground height = lowest occupied voxel, obstacle = any occupied
voxel in [ground + 0.30, ground + 1.90] (the G1's body envelope), observed mask.

Evaluation compares the map against the ORACLE map (GT poses + GT dynamic labels), so the
two error sources are separated: pose error (map blur / doubled walls) and dynamic objects
(ghost trails).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.seqio import iter_scans, read_gt, read_meta, read_tum  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BAND_LO, BAND_HI = 0.30, 1.90     # obstacle band above ground [m]


# ---------------------------------------------------------------- poses
class PoseTrack:
    """Body pose in WORLD, interpolated (linear p, slerp R)."""

    def __init__(self, t, p, R):
        o = np.argsort(t)
        t, p, R = t[o], p[o], R[o]
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        self.t, self.p = t[keep], p[keep]
        self.rot = Rotation.from_matrix(R[keep])
        self._slerp = Slerp(self.t, self.rot)

    def at(self, tq):
        tq = np.clip(np.asarray(tq, np.float64), self.t[0], self.t[-1])
        p = np.stack([np.interp(tq, self.t, self.p[:, k]) for k in range(3)], 1)
        return p, self._slerp(tq).as_matrix()

    def velocity(self, tq, dt=0.05):
        p1, _ = self.at(tq - dt)
        p2, _ = self.at(tq + dt)
        return (p2 - p1) / (2 * dt)


def load_poses(seq: Path, source: str) -> PoseTrack:
    gt = read_gt(seq)
    q = gt["q"]                                   # wxyz
    Rg = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    if source == "gt":
        return PoseTrack(gt["t"], gt["p"], Rg)
    scene = seq.name
    path = ROOT / "results" / scene / source / "poses_tum.txt"
    if not path.exists():
        path = Path(source)
    te, pe, qe = read_tum(path)
    Re = Rotation.from_quat(qe[:, [1, 2, 3, 0]]).as_matrix()
    # first-pose alignment (what the robot can actually do: known start pose)
    i0 = int(np.argmin(np.abs(gt["t"] - te[0])))
    R0 = Rg[i0] @ Re[0].T
    p_w = pe @ R0.T + (gt["p"][i0] - R0 @ pe[0])
    R_w = np.einsum("ij,njk->nik", R0, Re)
    return PoseTrack(te, p_w, R_w)


# ---------------------------------------------------------------- dynamic labels (GT)
@dataclass
class ActorBox:
    name: str
    actor: object
    half: np.ndarray        # body-frame half extents (x, y, z) incl. margin
    center: np.ndarray      # body-frame box centre


def gt_actor_boxes(scene_name: str, seed: int = 0, margin: float = 0.25):
    """Actor-frame AABB of every geom in each actor's body SUBTREE (vehicles have child bodies:
    boom, forks, bucket), computed with forward kinematics at a known actor pose."""
    import mujoco
    from sim import scenes as S
    from sim.record import G1_XML
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    sc = S.build(spec, scene_name, seed=seed)
    m = spec.compile()
    d = mujoco.MjData(m)
    poses = {}
    for a in sc.actors:
        a.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.name)
        a.mocap_id = int(m.body_mocapid[a.body_id])
        pos, _, yaw = a.state(0.0)
        d.mocap_pos[a.mocap_id] = pos
        d.mocap_quat[a.mocap_id] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        poses[a.name] = (pos, yaw)
    mujoco.mj_forward(m, d)
    boxes = []
    for a in sc.actors:
        pos, yaw = poses[a.name]
        Ra = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] != a.body_id:
                continue
            c = Ra.T @ (d.geom_xpos[g] - pos)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                Rg = Ra.T @ d.geom_xmat[g].reshape(3, 3)
                ext = np.abs(Rg) @ m.geom_size[g]
            else:
                ext = np.full(3, float(np.linalg.norm(m.geom_size[g])))
            lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
        if lo[0] > hi[0]:
            continue
        # yaw-symmetric in xy: older recordings drove some vehicles with the opposite body
        # heading, so the box must cover the actor whichever way it faces
        hxy = np.maximum(np.abs(lo[:2]), np.abs(hi[:2]))
        half = np.array([hxy[0], hxy[1], (hi[2] - lo[2]) / 2]) + margin
        boxes.append(ActorBox(a.name, a, half, np.array([0.0, 0.0, (hi[2] + lo[2]) / 2])))
    return boxes


def gt_dynamic_mask(pw, t_pts, boxes, n_slices=5):
    """Points inside any actor's box at their firing time (time sliced for speed)."""
    dyn = np.zeros(len(pw), bool)
    if not boxes:
        return dyn
    edges = np.linspace(t_pts.min(), t_pts.max() + 1e-9, n_slices + 1)
    for k in range(n_slices):
        sel = (t_pts >= edges[k]) & (t_pts < edges[k + 1])
        if not sel.any():
            continue
        tm = 0.5 * (edges[k] + edges[k + 1])
        P = pw[sel]
        hit = np.zeros(len(P), bool)
        for b in boxes:
            pos, _, yaw = b.actor.state(tm)
            c, s = np.cos(yaw), np.sin(yaw)
            d = P - pos
            lx = c * d[:, 0] + s * d[:, 1] - b.center[0]
            ly = -s * d[:, 0] + c * d[:, 1] - b.center[1]
            lz = d[:, 2] - b.center[2]
            hit |= (np.abs(lx) < b.half[0]) & (np.abs(ly) < b.half[1]) & (np.abs(lz) < b.half[2])
        dyn[np.where(sel)[0]] = hit
    return dyn


class GtVelocity:
    """True LiDAR-origin velocity (world) at any time, from gt.npz (IMU v world, w body)."""

    def __init__(self, seq, p_bl):
        g = read_gt(seq)
        self.t, self.v, self.w = g["t"], g["v"], g["w"]
        self.R = Rotation.from_quat(g["q"][:, [1, 2, 3, 0]]).as_matrix()
        self.p_bl = p_bl

    def lidar_vel(self, tq):
        k = np.clip(np.searchsorted(self.t, tq), 0, len(self.t) - 1)
        R = self.R[k]
        return self.v[k] + np.einsum("nij,nj->ni", R, np.cross(self.w[k], self.p_bl[None]))


def gt_doppler_moving(pw, p_lw, dop, v_l, thr=0.15):
    """Points whose Doppler disagrees with a static world under the TRUE sensor velocity
    (simulated Doppler noise is 0.03 m/s): exact 'on a moving body' label, except for
    purely tangential motion (covered by the actor boxes)."""
    d = pw - p_lw
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return np.abs(dop + np.einsum("ij,ij->i", d, v_l)) > thr


# ---------------------------------------------------------------- Doppler dynamic detection
def doppler_ego_velocity(d_hat, r, rng, t=None, iters=80, thr=0.12):
    """Sensor velocity (LiDAR frame) from a static-world Doppler model, RANSAC + LS.

    r = -d . v(t),  v(t) = v0 + a t   (t = firing time within the scan, if given)
    A humanoid head's velocity changes noticeably within one 0.1 s scan (heel strike,
    turning); a constant-velocity model then flags static points at the end of the scan
    as moving. Returns (params (3,) or (6,), inlier mask).
    """
    n = len(r)
    A = -d_hat if t is None else np.hstack([-d_hat, -d_hat * (t - t.mean())[:, None]])
    k = A.shape[1]
    if n < 10 * k:
        return None, np.zeros(n, bool)
    best, best_in = None, None
    for _ in range(iters):
        idx = rng.choice(n, k, replace=False)
        try:
            v = np.linalg.solve(A[idx], r[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(A @ v - r) < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = v, inl
    if best_in is None or best_in.sum() < 20:
        return None, np.zeros(n, bool)
    v, *_ = np.linalg.lstsq(A[best_in], r[best_in], rcond=None)
    inl = np.abs(A @ v - r) < thr
    v, *_ = np.linalg.lstsq(A[inl], r[inl], rcond=None)       # one refinement pass
    return v, inl


def doppler_dynamic_mask(pl, dop, pw, rng, t=None, thr=0.30, grow_voxel=0.35, min_seeds=3):
    """Dynamic points of one scan: Doppler residual seeds, grown to the object.

    Doppler only sees the RADIAL component of an object's motion; a worker walking across
    the beam has ~0 residual on its tangential side. Seeds are therefore grown through
    0.35 m world voxels (26-neighbourhood) that hold >= min_seeds seeds, so the whole body
    is removed. Floor points next to a worker's feet are lost too (harmless: re-observed).
    """
    rng_ = np.linalg.norm(pl, axis=1)
    ok = rng_ > 0.5
    d_hat = np.zeros_like(pl)
    d_hat[ok] = pl[ok] / rng_[ok, None]
    v, _ = doppler_ego_velocity(d_hat[ok], dop[ok], rng, None if t is None else t[ok])
    if v is None:
        return np.zeros(len(pl), bool), None
    if t is None:
        pred = -d_hat @ v
    else:
        tc = t - t[ok].mean()
        pred = -(d_hat @ v[:3]) - (d_hat @ v[3:]) * tc
    res = dop - pred
    seed = ok & (np.abs(res) > thr)
    if seed.sum() < min_seeds:
        return np.zeros(len(pl), bool), v
    key = np.floor(pw / grow_voxel).astype(np.int64)
    kmin = key.min(0)
    key -= kmin
    dims = key.max(0) + 3
    flat = (key[:, 0] + 1) * dims[1] * dims[2] + (key[:, 1] + 1) * dims[2] + (key[:, 2] + 1)
    cnt = np.bincount(flat[seed], minlength=int(np.prod(dims)))
    hot = np.where(cnt >= min_seeds)[0]
    if len(hot) == 0:
        return seed, v
    grid = np.zeros(int(np.prod(dims)), bool)
    grid[hot] = True
    grid = binary_dilation(grid.reshape(dims), np.ones((3, 3, 3), bool)).ravel()
    return seed | grid[flat], v


# ---------------------------------------------------------------- voxel map
class VoxelMap:
    def __init__(self, lo, hi, res=0.15):
        self.res = res
        self.lo = np.asarray(lo, float)
        self.dims = np.ceil((np.asarray(hi) - self.lo) / res).astype(int) + 1
        n = int(np.prod(self.dims))
        self.hits = np.zeros(n, np.uint32)
        self.miss = np.zeros(n, np.uint32)
        # temporal persistence: first / last scan index a voxel was hit
        self.first = np.full(n, 65535, np.uint16)
        self.last = np.zeros(n, np.uint16)
        self._hbuf, self._mbuf = [], []

    def flat(self, P):
        k = np.floor((P - self.lo) / self.res).astype(np.int64)
        ok = np.all((k >= 0) & (k < self.dims), 1)
        k = k[ok]
        return k[:, 0] * self.dims[1] * self.dims[2] + k[:, 1] * self.dims[2] + k[:, 2]

    def add_hits(self, P, k=0):
        idx = np.unique(self.flat(P))
        self._hbuf.append(idx)
        self.first[idx] = np.minimum(self.first[idx], k)
        self.last[idx] = k

    def add_rays(self, origin, P, stop_short=0.3, every=3, protect=None):
        """Misses along origin->P (excluding the last stop_short m) for a subset of beams.
        protect: voxel indices hit in this same scan; they never receive a miss (grazing beams
        otherwise erode thin real structure such as columns and railings)."""
        P = P[::every]
        d = P - origin
        L = np.linalg.norm(d, axis=1)
        keep = L > stop_short + self.res
        d, L = d[keep] / L[keep, None], L[keep] - stop_short
        n = np.ceil(L / self.res).astype(int)
        tot = int(n.sum())
        if tot == 0:
            return
        ray = np.repeat(np.arange(len(L)), n)
        step = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        S = origin + d[ray] * (step[:, None] * self.res)
        m = np.unique(self.flat(S))
        if protect is not None and len(protect):
            m = m[~np.isin(m, protect, assume_unique=True)]
        self._mbuf.append(m)

    def flush(self):
        n = len(self.hits)
        if self._hbuf:
            self.hits += np.bincount(np.concatenate(self._hbuf), minlength=n).astype(np.uint32)
        if self._mbuf:
            self.miss += np.bincount(np.concatenate(self._mbuf), minlength=n).astype(np.uint32)
        self._hbuf, self._mbuf = [], []

    def occupied(self, min_hits=2, p_occ=0.5, persist_scans=0):
        """persist_scans > 0: a voxel must have been hit over a span of at least that many
        scans. A person crossing the beam sideways (no radial Doppler) occupies a voxel for
        ~0.5 s and is rejected; walls and parked machines are seen for many seconds."""
        self.flush()
        h = self.hits.astype(np.float32)
        occ = (h >= min_hits) & (h / np.maximum(h + self.miss, 1) > p_occ)
        if persist_scans:
            occ &= (self.last.astype(np.int32) - self.first.astype(np.int32)) >= persist_scans
        return occ.reshape(self.dims)

    def products(self, persist_scans=0):
        """2.5D layers: ground height, obstacle mask, observed mask (xy cells)."""
        occ = self.occupied(persist_scans=persist_scans)
        nz = self.dims[2]
        any_occ = occ.any(2)
        first = np.argmax(occ, axis=2)                       # lowest occupied voxel
        ground = self.lo[2] + (first + 0.5) * self.res
        ground[~any_occ] = np.nan
        zi = np.arange(nz)[None, None, :]
        lo_i = first[..., None] + int(round(BAND_LO / self.res))
        hi_i = first[..., None] + int(round(BAND_HI / self.res))
        obst = (occ & (zi >= lo_i) & (zi <= hi_i)).any(2) & any_occ
        return dict(ground=ground, obstacle=obst, observed=any_occ)


# ---------------------------------------------------------------- driver
@dataclass
class MapConfig:
    res: float = 0.15
    max_range: float = 20.0
    dyn: str = "none"           # none | raycast | doppler | persist | combos with '+' | oracle
    persist: float = 2.0        # [s] temporal persistence for the 'persist' variants
    decim: int = 1              # use every k-th scan
    seed: int = 0


def build_map(seq: Path, pose_src: str, cfg: MapConfig, boxes=None, gt_track=None, verbose=True):
    meta = read_meta(seq)
    R_bl = np.array(meta["R_bl"]).reshape(3, 3)
    p_bl = np.array(meta["p_bl_b"])
    track = load_poses(seq, pose_src)
    gt_track = gt_track or load_poses(seq, "gt")
    gt_vel = GtVelocity(seq, p_bl) if boxes else None
    lo = gt_track.p.min(0) - [cfg.max_range, cfg.max_range, 4.0]
    hi = gt_track.p.max(0) + [cfg.max_range, cfg.max_range, 4.5]
    vm = VoxelMap(lo, hi, cfg.res)
    rng = np.random.default_rng(cfg.seed)
    stats = dict(scans=0, pts=0, dyn_removed=0, gt_dyn=0, tp=0, fp=0, fn=0)
    t_start = track.t[0]
    for k, (t0, pts) in enumerate(iter_scans(seq)):
        if k % cfg.decim or t0 < t_start:
            continue
        pl = np.stack([pts["x"], pts["y"], pts["z"]], 1).astype(np.float64)
        r = np.linalg.norm(pl, axis=1)
        keep = (r > 0.5) & (r < cfg.max_range)
        pl, dop, tp = pl[keep], pts["doppler"][keep].astype(np.float64), t0 + pts["t_off"][keep].astype(np.float64)
        if len(pl) < 50:
            continue
        pb, Rb = track.at(tp)
        p_lw = pb + np.einsum("nij,j->ni", Rb, p_bl)
        pw = p_lw + np.einsum("nij,nj->ni", Rb @ R_bl, pl)

        need_gt = cfg.dyn == "oracle" or boxes is not None
        gt_dyn = None
        if need_gt and boxes:
            pb_g, Rb_g = gt_track.at(tp)
            p_lw_g = pb_g + np.einsum("nij,j->ni", Rb_g, p_bl)
            pw_g = p_lw_g + np.einsum("nij,nj->ni", Rb_g @ R_bl, pl)
            gt_dyn = gt_dynamic_mask(pw_g, tp, boxes)
            if gt_vel is not None:
                gt_dyn |= gt_doppler_moving(pw_g, p_lw_g, dop, gt_vel.lidar_vel(tp))
            stats["gt_dyn"] += int(gt_dyn.sum())

        if cfg.dyn == "oracle":
            drop = gt_dyn if gt_dyn is not None else np.zeros(len(pl), bool)
        elif cfg.dyn.startswith("doppler"):
            drop, _ = doppler_dynamic_mask(pl, dop, pw, rng, t=tp - t0)
        else:
            drop = np.zeros(len(pl), bool)
        if gt_dyn is not None and cfg.dyn.startswith("doppler"):
            stats["tp"] += int((drop & gt_dyn).sum())
            stats["fp"] += int((drop & ~gt_dyn).sum())
            stats["fn"] += int((~drop & gt_dyn).sum())
        stats["dyn_removed"] += int(drop.sum())
        stats["pts"] += len(pl)
        stats["scans"] += 1

        P = pw[~drop]
        vm.add_hits(P, k)
        if "raycast" in cfg.dyn:
            vm.add_rays(p_lw[len(p_lw) // 2], P, protect=(vm._hbuf[-1] if "guard" in cfg.dyn else None))
        if stats["scans"] % 40 == 0:
            vm.flush()
            if verbose:
                print(f"  {seq.name}/{pose_src}/{cfg.dyn}: scan {k}", flush=True)
    vm.flush()
    return vm, stats


def compare(prod, ref, res):
    """Map vs reference map (both 2.5D products on the same grid)."""
    both = prod["observed"] & ref["observed"]
    A, B = prod["obstacle"] & both, ref["obstacle"] & both
    tp, fp, fn = int((A & B).sum()), int((A & ~B).sum()), int((~A & B).sum())
    g = prod["ground"][both & ~A & ~B]
    gr = ref["ground"][both & ~A & ~B]
    dz = g - gr
    cell = res * res
    return dict(obst_iou=tp / max(tp + fp + fn, 1), ghost_m2=fp * cell, missed_m2=fn * cell,
                obst_ref_m2=int(B.sum()) * cell,
                ground_rmse=float(np.sqrt(np.nanmean(dz ** 2))) if len(dz) else float("nan"),
                ground_p95=float(np.nanpercentile(np.abs(dz), 95)) if len(dz) else float("nan"),
                coverage_m2=int(both.sum()) * cell)


def compare_tol(prod, ref, res, tol=0.3):
    """Tolerance-based obstacle agreement (mapping-standard): a mapped obstacle cell is correct if
    a reference obstacle lies within tol, and vice versa. Robust to the one-cell offsets that make
    IoU of thin walls collapse under small pose errors."""
    from scipy.ndimage import distance_transform_edt
    both = prod["observed"] & ref["observed"]
    A, B = prod["obstacle"] & both, ref["obstacle"] & both
    dB = distance_transform_edt(~B) * res
    dA = distance_transform_edt(~A) * res
    tp_a = int((A & (dB <= tol)).sum())
    tp_b = int((B & (dA <= tol)).sum())
    prec = tp_a / max(int(A.sum()), 1)
    rec = tp_b / max(int(B.sum()), 1)
    cell = res * res
    return dict(prec_tol=prec, rec_tol=rec, f1_tol=2 * prec * rec / max(prec + rec, 1e-9),
                ghost_tol_m2=int((A & (dB > tol)).sum()) * cell, missed_tol_m2=int((B & (dA > tol)).sum()) * cell)


def ghost_on_actor_paths(prod, ref, boxes, vm: VoxelMap, width=1.0):
    """Ghost obstacle area that lies on the swept paths of moving actors."""
    if not boxes:
        return 0.0
    nx, ny = prod["obstacle"].shape
    sweep = np.zeros((nx, ny), bool)
    for b in boxes:
        P = b.actor.path
        for i in range(len(P) - 1):
            for s in np.linspace(0, 1, 60):
                x, y = P[i] + s * (P[i + 1] - P[i])
                ix, iy = int((x - vm.lo[0]) / vm.res), int((y - vm.lo[1]) / vm.res)
                r = int(np.ceil((max(b.half[:2]) + width * 0.5) / vm.res))
                sweep[max(ix - r, 0):ix + r + 1, max(iy - r, 0):iy + r + 1] = True
    ghost = prod["obstacle"] & ~ref["obstacle"] & sweep & ref["observed"]
    return float(ghost.sum()) * vm.res ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dynamic")
    ap.add_argument("--poses", nargs="+", default=["gt", "fmcw3d", "fmcw4d_leg"])
    ap.add_argument("--dyn", nargs="+", default=["none", "raycast", "doppler"])
    ap.add_argument("--combos", nargs="+", default=None,
                    help="explicit pose__dyn pairs, overrides --poses x --dyn")
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--decim", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seq = ROOT / "data" / a.scene
    out = Path(a.out or ROOT / "results" / a.scene / "elevmap")
    out.mkdir(parents=True, exist_ok=True)
    boxes = gt_actor_boxes(a.scene)
    gt_track = load_poses(seq, "gt")

    print(f"[{a.scene}] oracle map (GT pose + GT dynamic labels)")
    ref_vm, _ = build_map(seq, "gt", MapConfig(res=a.res, dyn="oracle", decim=a.decim), boxes, gt_track)
    ref = ref_vm.products()
    np.savez_compressed(out / "map_oracle.npz", lo=ref_vm.lo, res=ref_vm.res, **ref)

    results = {}
    combos = a.combos or [f"{ps}__{dyn}" for ps in a.poses for dyn in a.dyn]
    for name in combos:
        ps, dyn = name.split("__")
        if True:
            print(f"[{a.scene}] {name}")
            vm, st = build_map(seq, ps, MapConfig(res=a.res, dyn=dyn, decim=a.decim), boxes, gt_track)
            ps_scans = int(round(2.0 / 0.1)) if "persist" in dyn else 0
            prod = vm.products(persist_scans=ps_scans)
            r = compare(prod, ref, vm.res)
            r.update(compare_tol(prod, ref, vm.res))
            r["ghost_on_actor_paths_m2"] = ghost_on_actor_paths(prod, ref, boxes, vm)
            if dyn.startswith("doppler") and st["gt_dyn"]:
                r["dyn_precision"] = st["tp"] / max(st["tp"] + st["fp"], 1)
                r["dyn_recall"] = st["tp"] / max(st["tp"] + st["fn"], 1)
            r["dyn_removed_frac"] = st["dyn_removed"] / max(st["pts"], 1)
            r["gt_dyn_frac"] = st["gt_dyn"] / max(st["pts"], 1)
            results[name] = r
            np.savez_compressed(out / f"map_{name}.npz", lo=vm.lo, res=vm.res, **prod)
            print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    mf = out / "metrics.json"
    allres = json.loads(mf.read_text()) if mf.exists() else {}
    allres.update(results)
    mf.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
