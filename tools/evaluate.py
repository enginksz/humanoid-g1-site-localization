"""Trajectory metrics against simulator ground truth (no external deps).

ATE    : RMSE of position after SE(3) Umeyama alignment (no scale)
RPE    : KITTI-style translational drift [%] and rotational drift [deg/m]
         over GT path segments of 5/10/20 m
final  : end-point error after aligning only the first pose (what a robot
         would actually experience without loop closure)
z_rmse : vertical error after first-pose alignment (humanoid bounce / gravity)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.seqio import read_tum  # noqa: E402


def quat_to_R(q):
    w, x, y, z = q.T
    R = np.empty((len(w), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def associate(t_est, t_gt, tol=0.006):
    k = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    k = np.where(np.abs(t_gt[k - 1] - t_est) < np.abs(t_gt[k] - t_est), k - 1, k)
    ok = np.abs(t_gt[k] - t_est) < tol
    return np.nonzero(ok)[0], k[ok]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    C = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def evaluate(est_path, gt_path, seg_lengths=(5.0, 10.0, 20.0)):
    te, pe, qe = read_tum(est_path)
    tg, pg, qg = read_tum(gt_path)
    ie, ig = associate(te, tg)
    if len(ie) < 10:
        return dict(failed=True, reason="too few associated poses")
    te, pe, qe = te[ie], pe[ie], qe[ie]
    pg, qg = pg[ig], qg[ig]
    Re, Rg = quat_to_R(qe), quat_to_R(qg)
    if not np.all(np.isfinite(pe)):
        return dict(failed=True, reason="non-finite estimate")

    # full SE3 alignment
    R, t = umeyama(pe, pg)
    pa = pe @ R.T + t
    ate = float(np.sqrt(np.mean(np.sum((pa - pg) ** 2, 1))))

    # first-pose alignment (pure dead-reckoning view)
    R0 = Rg[0] @ Re[0].T
    p0 = pe @ R0.T + (pg[0] - R0 @ pe[0])
    err0 = p0 - pg
    final = float(np.linalg.norm(err0[-1]))
    z_rmse = float(np.sqrt(np.mean(err0[:, 2] ** 2)))
    yaw_err_final = float(np.degrees(np.arctan2(*(lambda M: (M[1, 0], M[0, 0]))((R0 @ Re[-1]) @ Rg[-1].T))))

    # RPE over path-length segments
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pg, axis=0), axis=1))])
    rpe = {}
    for L in seg_lengths:
        t_err, r_err = [], []
        for i in range(0, len(dist), 5):
            j = np.searchsorted(dist, dist[i] + L)
            if j >= len(dist):
                break
            dG = Rg[i].T @ (pg[j] - pg[i])
            dE = Re[i].T @ (pe[j] - pe[i])
            dRG = Rg[i].T @ Rg[j]
            dRE = Re[i].T @ Re[j]
            t_err.append(np.linalg.norm(dE - dG) / L * 100)
            r_err.append(rot_angle(dRE.T @ dRG) / L)
        if t_err:
            rpe[f"{int(L)}m"] = dict(trans_pct=float(np.mean(t_err)), rot_deg_per_m=float(np.mean(r_err)))

    path_len = float(dist[-1])
    return dict(failed=False, n=len(te), duration=float(te[-1] - te[0]), path_length=path_len,
                ate_rmse=ate, final_err=final, final_err_pct=100 * final / max(path_len, 1e-6),
                z_rmse=z_rmse, yaw_err_final_deg=yaw_err_final, rpe=rpe,
                max_err=float(np.max(np.linalg.norm(pa - pg, axis=1))))


if __name__ == "__main__":
    r = evaluate(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=2))
