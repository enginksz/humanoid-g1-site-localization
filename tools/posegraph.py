"""Loop closure + pose-graph optimisation (GTSAM) on top of any odometry.

  keyframes   every 1 m / 15 deg of odometry; each keeps a local submap (scans
              within +-N keyframes, deskewed and stitched with odometry)
  candidates  (a) odometry proximity with a drift-aware radius (r0 + 5% of the
              distance travelled since the candidate), (b) Scan Context
              place recognition (ring x sector max-height, yaw-shift search)
              which also works when the odometry is far off
  verify      robust point-to-plane ICP between the two submaps, accept on
              inlier ratio and RMSE (structure points only, like maploc)
  optimise    Pose3 graph: odometry BetweenFactors + loop BetweenFactors with a
              Cauchy kernel (a wrong loop cannot drag the graph), Levenberg-Marquardt

Also merges two sessions/robots that started in unknown relative poses:
inter-session loops are found by Scan Context only (no shared frame), then one
graph with an anchor on session A solves both.

  python3 tools/posegraph.py single data/outdoor results/outdoor/fmcw3d/poses_tum.txt results/outdoor/pgo_fmcw3d
  python3 tools/posegraph.py merge  A_seq A_odom  B_seq B_odom  out_dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gtsam
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.maploc import deskew, pose_mat, voxel, yaw_of  # noqa: E402
from tools.seqio import iter_scans, read_meta, read_tum  # noqa: E402


# ------------------------------------------------------------------ keyframes
def keyframes(seq, odom_tum, step_m=1.0, step_deg=15.0, window=6):
    seq = Path(seq)
    meta = read_meta(seq)
    R_bl, p_bl = np.array(meta["R_bl"]).reshape(3, 3), np.array(meta["p_bl_b"])
    to, po, qo = read_tum(odom_tum)
    scans, Ts, ts = [], [], []
    prev = None
    for t0, s in iter_scans(seq):
        t = t0 + 0.098
        k = int(np.argmin(np.abs(to - t)))
        if abs(to[k] - t) > 0.06:
            continue
        T = pose_mat(po[k], qo[k])
        dT = np.eye(4) if prev is None else np.linalg.inv(prev) @ T
        prev = T
        scans.append(voxel(deskew(s, R_bl, p_bl, dT), 0.2))
        Ts.append(T)
        ts.append(t)
    kf = [0]
    for i in range(1, len(Ts)):
        d = np.linalg.inv(Ts[kf[-1]]) @ Ts[i]
        if np.linalg.norm(d[:3, 3]) > step_m or abs(np.degrees(yaw_of(d[:3, :3]))) > step_deg:
            kf.append(i)
    # local submap per keyframe: scans of the neighbouring keyframes' spans, in the keyframe frame
    K = []
    for j, i in enumerate(kf):
        lo = kf[max(j - window, 0)]
        hi = kf[min(j + window, len(kf) - 1)]
        Ti_inv = np.linalg.inv(Ts[i])
        P = np.concatenate([scans[m] @ (Ti_inv @ Ts[m])[:3, :3].T + (Ti_inv @ Ts[m])[:3, 3]
                            for m in range(lo, hi + 1, 2)])
        K.append(dict(t=ts[i], T_odom=Ts[i], cloud=voxel(P, 0.25)))
    return K


# ------------------------------------------------------------------ Scan Context
def scan_context(P, n_ring=20, n_sector=60, r_max=40.0, z_off=0.8):
    r = np.hypot(P[:, 0], P[:, 1])
    a = np.arctan2(P[:, 1], P[:, 0])
    ok = r < r_max
    ri = np.minimum((r[ok] / r_max * n_ring).astype(int), n_ring - 1)
    si = np.minimum(((a[ok] + np.pi) / (2 * np.pi) * n_sector).astype(int), n_sector - 1)
    D = np.zeros((n_ring, n_sector))
    np.maximum.at(D, (ri, si), P[ok, 2] + z_off)
    return D


def sc_distance(A, B):
    """Min over sector shifts of the mean column cosine distance; returns (dist, shift)."""
    best, shift = 2.0, 0
    na = np.linalg.norm(A, axis=0)
    for s_ in range(A.shape[1]):
        Bs = np.roll(B, s_, axis=1)
        nb = np.linalg.norm(Bs, axis=0)
        m = (na > 0) & (nb > 0)
        if m.sum() < 5:
            continue
        cos = (A[:, m] * Bs[:, m]).sum(0) / (na[m] * nb[m])
        d = 1 - cos.mean()
        if d < best:
            best, shift = d, s_
    return best, shift


# ------------------------------------------------------------------ verification (robust ICP, submap to submap)
def _normals(P, k=12):
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)
    N = np.empty_like(P)
    for i, nb in enumerate(idx):
        Q = P[nb] - P[nb].mean(0)
        N[i] = np.linalg.svd(Q, full_matrices=False)[2][2]
    return tree, N


def icp(src, tgt, tgt_tree, tgt_N, T0, iters=25, c=0.3):
    T = T0.copy()
    for it in range(iters):
        dmax = 2.0 if it < 8 else 0.6
        Pw = src @ T[:3, :3].T + T[:3, 3]
        d, idx = tgt_tree.query(Pw, distance_upper_bound=dmax)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            return T, 0.0, np.inf
        p, q, n = Pw[ok], tgt[idx[ok]], tgt_N[idx[ok]]
        r = np.einsum("ij,ij->i", p - q, n)
        w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
        J = np.hstack([np.cross(p, n), n])
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(dx[:3]).as_matrix()
        dT[:3, 3] = dx[3:]
        T = dT @ T
        if np.linalg.norm(dx) < 1e-5:
            break
    Pw = src @ T[:3, :3].T + T[:3, 3]
    st = (Pw[:, 2] > -0.55) & (Pw[:, 2] < 2.0)           # structure band (body frame, robot ~0.8 m high)
    d, _ = tgt_tree.query(Pw[st], distance_upper_bound=0.2)
    inl = np.isfinite(d)
    fit = float(inl.mean()) if len(inl) else 0.0
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf
    return T, fit, rmse


# ------------------------------------------------------------------ loop search
def find_loops(K, other=None, min_dt=30.0, r0=4.0, drift=0.05, sc_thresh=0.25, use_odom=True, fit_min=0.55,
               log=print):
    """Loops within K (other=None) or between K (query) and other (database)."""
    db = K if other is None else other
    sc_db = [scan_context(k["cloud"]) for k in db]
    trees = {}
    dist_db = np.concatenate([[0], np.cumsum([np.linalg.norm(db[i]["T_odom"][:3, 3] - db[i - 1]["T_odom"][:3, 3])
                                              for i in range(1, len(db))])])
    loops = []
    last_i = -10
    for i, k in enumerate(K):
        if other is None and i - last_i < 5:          # at most one loop per 5 keyframes
            continue
        sc_q = scan_context(k["cloud"])
        cands = []
        for j, kd in enumerate(db):
            if other is None and k["t"] - kd["t"] < min_dt:
                continue
            via = None
            if use_odom and other is None:
                rad = r0 + drift * (dist_db[i] - dist_db[j] if other is None else 0)
                if np.linalg.norm(k["T_odom"][:2, 3] - kd["T_odom"][:2, 3]) < rad:
                    via = "odom"
            d_sc, shift = sc_distance(sc_q, sc_db[j])
            if d_sc < sc_thresh:
                via = via or "sc"
            if via:
                cands.append((d_sc, j, shift, via))
        cands.sort()
        for d_sc, j, shift, via in cands[:3]:
            kd = db[j]
            if j not in trees:
                trees[j] = _normals(kd["cloud"])
            if via == "odom":
                T0 = np.linalg.inv(kd["T_odom"]) @ k["T_odom"]          # relative pose from odometry
            else:
                T0 = np.eye(4)
                T0[:3, :3] = Rotation.from_euler("z", -shift / 60 * 2 * np.pi).as_matrix()
            T, fit, rmse = icp(k["cloud"], kd["cloud"], *trees[j], T0)
            if fit > fit_min and rmse < 0.12:
                loops.append(dict(i=i, j=j, T_ji=T, fit=fit, rmse=rmse, via=via, sc=d_sc))
                last_i = i
                log(f"  loop {i:4d} -> {j:4d}  via {via:4s}  fit {fit:.2f}  rmse {rmse:.3f}  sc {d_sc:.2f}")
                break
    return loops


# ------------------------------------------------------------------ pose graph
def _p3(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def optimise(sessions, loops, inter=()):
    """sessions: list of keyframe lists; loops: list of (session, loop dict);
    inter: loops between session 1 (query) and session 0 (db). Session 0 is anchored."""
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    key = lambda s, i: gtsam.symbol(chr(ord("a") + s), i)
    odo_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.03, 0.03, 0.03]))
    # Loops are plain Gaussian here; outlier rejection is done by GNC below. (A fixed
    # Cauchy kernel also rejects *correct* loops whose residual is large because the
    # odometry drifted a lot - exactly the loops that matter.)
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02] * 3 + [0.05] * 3))
    known = []
    for s, K in enumerate(sessions):
        for i, k in enumerate(K):
            v.insert(key(s, i), _p3(k["T_init"]))
            if i:
                known.append(g.size())
                g.add(gtsam.BetweenFactorPose3(key(s, i - 1), key(s, i),
                                               _p3(np.linalg.inv(K[i - 1]["T_odom"]) @ k["T_odom"]), odo_noise))
    known.append(g.size())
    g.add(gtsam.PriorFactorPose3(key(0, 0), _p3(sessions[0][0]["T_init"]),
                                 gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))))
    for s, L in loops:
        g.add(gtsam.BetweenFactorPose3(key(s, L["j"]), key(s, L["i"]), _p3(L["T_ji"]), loop_noise))
    for L in inter:
        g.add(gtsam.BetweenFactorPose3(key(0, L["j"]), key(1, L["i"]), _p3(L["T_ji"]), loop_noise))
    # Graduated non-convexity (TLS): starts convex, gradually turns wrong loops off;
    # odometry and the prior are declared known inliers.
    gp = gtsam.GncLMParams()
    gp.setKnownInliers(known)
    gp.setLossType(gtsam.GncLossType.TLS)
    opt = gtsam.GncLMOptimizer(g, v, gp)
    res = opt.optimize()
    w = np.array(opt.getWeights())
    loop_idx = [i for i in range(g.size()) if i not in set(known)]
    optimise.last_loop_weights = w[loop_idx] if len(w) else np.array([])
    return [[res.atPose3(key(s, i)).matrix() for i in range(len(K))] for s, K in enumerate(sessions)]


def kf_errors(K, Tlist, gt_tum, align="first"):
    tg, pg, qg = read_tum(gt_tum)
    P = np.array([T[:3, 3] for T in Tlist])
    G = np.array([pg[np.argmin(np.abs(tg - k["t"]))] for k in K])
    Gq = qg[np.argmin(np.abs(tg - K[0]["t"]))]
    T0g = pose_mat(G[0], Gq)
    A = T0g @ np.linalg.inv(Tlist[0]) if align == "first" else np.eye(4)
    Pa = P @ A[:3, :3].T + A[:3, 3]
    e = np.linalg.norm(Pa - G, axis=1)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max=float(e.max()), end=float(e[-1])), Pa, G


def single(seq, odom, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    K = keyframes(seq, odom)
    for k in K:
        k["T_init"] = k["T_odom"]
    print(f"{len(K)} keyframes ({time.time() - t0:.0f} s)")
    loops = find_loops(K)
    # ground-truth check of every accepted loop (relative-pose translation error)
    tg, pg, qg = read_tum(Path(seq) / "gt_tum.txt")
    Tg = lambda t: pose_mat(pg[np.argmin(np.abs(tg - t))], qg[np.argmin(np.abs(tg - t))])
    for L in loops:
        T_true = np.linalg.inv(Tg(K[L["j"]]["t"])) @ Tg(K[L["i"]]["t"])
        L["gt_err"] = float(np.linalg.norm(T_true[:3, 3] - L["T_ji"][:3, 3]))
    Topt = optimise([K], [(0, L) for L in loops])[0]
    wts = getattr(optimise, "last_loop_weights", np.array([]))
    gt = Path(seq) / "gt_tum.txt"
    before, Pb, G = kf_errors(K, [k["T_odom"] for k in K], gt)
    after, Pa, _ = kf_errors(K, Topt, gt)
    res = dict(n_kf=len(K), n_loops=len(loops),
               loops=[dict(i=L["i"], j=L["j"], via=L["via"], fit=round(L["fit"], 3), gt_err=round(L["gt_err"], 3),
                           gnc_weight=float(w_)) for L, w_ in zip(loops, wts)],
               n_true_loops=int(sum(L["gt_err"] < 0.5 for L in loops)), loops_via={v: sum(L["via"] == v for L in loops) for v in ("odom", "sc")},
               before=before, after=after, wall_s=time.time() - t0)
    np.savez_compressed(out / "pgo.npz", t=[k["t"] for k in K], before=Pb, after=Pa, gt=G,
                        loops=np.array([[L["i"], L["j"]] for L in loops]).reshape(-1, 2))
    (out / "pgo.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def merge(seq_a, odom_a, seq_b, odom_b, out):
    """Two sessions with unknown relative start pose -> one consistent map."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A, B = keyframes(seq_a, odom_a), keyframes(seq_b, odom_b)
    for k in A:
        k["T_init"] = k["T_odom"]
    loops_a = find_loops(A, log=lambda *a: None)
    loops_b = find_loops(B, log=lambda *a: None)
    print(f"A {len(A)} kf / {len(loops_a)} loops, B {len(B)} kf / {len(loops_b)} loops; searching inter-session loops")
    inter = find_loops(B, other=A, use_odom=False, sc_thresh=0.22, fit_min=0.6)
    res = dict(n_inter=len(inter))
    if not inter:
        print("no inter-session loop found")
        (out / "merge.json").write_text(json.dumps(res, indent=2))
        return res
    # initialise B in A's frame from the first inter-session loop
    L = inter[0]
    T_A_B = A[L["j"]]["T_init"] @ L["T_ji"] @ np.linalg.inv(B[L["i"]]["T_odom"])
    for k in B:
        k["T_init"] = T_A_B @ k["T_odom"]
    TA, TB = optimise([A, B], [(0, l) for l in loops_a] + [(1, l) for l in loops_b], inter)
    # errors in A's GT-aligned frame (first pose of A)
    ea, Pa, Ga = kf_errors(A, TA, Path(seq_a) / "gt_tum.txt")
    tg, pg, qg = read_tum(Path(seq_a) / "gt_tum.txt")
    T0g = pose_mat(pg[np.argmin(np.abs(tg - A[0]["t"]))], qg[np.argmin(np.abs(tg - A[0]["t"]))])
    Al = T0g @ np.linalg.inv(TA[0])
    tgb, pgb, _ = read_tum(Path(seq_b) / "gt_tum.txt")
    PB = np.array([(Al @ T)[:3, 3] for T in TB])
    GB = np.array([pgb[np.argmin(np.abs(tgb - k["t"]))] for k in B])
    eb = np.linalg.norm(PB - GB, axis=1)
    res.update(A=ea, B=dict(rmse=float(np.sqrt(np.mean(eb ** 2))), max=float(eb.max())),
               inter_loops=[dict(i=l["i"], j=l["j"], fit=l["fit"], sc=l["sc"]) for l in inter], wall_s=time.time() - t0)
    np.savez_compressed(out / "merge.npz", A=Pa, B=PB, GA=Ga, GB=GB)
    (out / "merge.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "inter_loops"}, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("single")
    s1.add_argument("seq"); s1.add_argument("odom"); s1.add_argument("out")
    s2 = sub.add_parser("merge")
    s2.add_argument("seq_a"); s2.add_argument("odom_a"); s2.add_argument("seq_b"); s2.add_argument("odom_b")
    s2.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "single":
        single(a.seq, a.odom, a.out)
    else:
        merge(a.seq_a, a.odom_a, a.seq_b, a.odom_b, a.out)
