"""LiDAR-initialised 3D Gaussian Splatting on the head-camera images.

Question it answers: how much does localization quality matter for a
photometric map? The same images are fitted with camera poses coming from
ground truth or from one of the estimators; Gaussians are seeded from a LiDAR
map accumulated with the *same* poses (so each run is self-consistent, as it
would be on the robot). Quality is PSNR/SSIM on held-out views (every 8th).

  python3 tools/gs_train.py --scene slab --poses gt
  python3 tools/gs_train.py --scene slab --poses fmcw4d_leg
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.record import quat_wxyz_to_R  # noqa: E402
from tools.seqio import iter_scans, read_meta, read_tum  # noqa: E402


def lidar_map(seq: Path, pose_tum: Path, stride=4, voxel=0.08, max_pts=200_000):
    """Accumulate raw scans with the given body poses (pose at scan mid-time)."""
    meta = read_meta(seq)
    R_bl = np.array(meta["R_bl"]).reshape(3, 3)
    p_bl = np.array(meta["p_bl_b"])
    t, p, q = read_tum(pose_tum)
    pts = []
    for i, (t0, s) in enumerate(iter_scans(seq)):
        if i % stride:
            continue
        tm = t0 + 0.05
        k = np.searchsorted(t, tm)
        if k == 0 or k >= len(t) or abs(t[k] - tm) > 0.08:
            continue
        R = quat_wxyz_to_R(q[k])
        xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > 0.8) & (r < 40)]
        pts.append((xyz @ R_bl.T + p_bl) @ R.T + p[k])
    P = np.concatenate(pts)
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    P = P[idx]
    if len(P) > max_pts:
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    return P.astype(np.float32)


def ssim(a, b):
    from torchmetrics.functional import structural_similarity_index_measure as S
    return S(a, b, data_range=1.0)


def train(scene, poses, iters=7000, data_dir=ROOT / "data", res_dir=ROOT / "results", seed=0):
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(seed)
    dev = "cuda"
    cam = res_dir / scene / "cam"
    meta = json.loads((cam / "frames.json").read_text())
    key = "c2w_gt" if poses == "gt" else f"c2w_{poses}"
    frames = [f for f in meta["frames"] if key in f]
    imgs = torch.stack([torch.from_numpy(imageio.imread(cam / f["file"])[..., :3]).float() / 255 for f in frames]).to(dev)
    c2w = torch.tensor(np.array([f[key] for f in frames]), dtype=torch.float32, device=dev)
    viewmats = torch.linalg.inv(c2w)
    H, W = meta["height"], meta["width"]
    K = torch.tensor([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], dtype=torch.float32, device=dev)
    idx = np.arange(len(frames))
    test = idx[idx % 8 == 4]
    train_ids = idx[idx % 8 != 4]

    pose_tum = (data_dir / scene / "gt_tum.txt") if poses == "gt" else (res_dir / scene / poses / "poses_tum.txt")
    from scipy.spatial import cKDTree
    pts_np = lidar_map(data_dir / scene, pose_tum)
    # scale init: mean distance to the 3 nearest neighbours
    dist, _ = cKDTree(pts_np).query(pts_np, k=4)
    d3 = torch.from_numpy(dist[:, 1:].mean(1)).float().to(dev).clamp(1e-3, 0.5)
    pts = torch.from_numpy(pts_np).to(dev)
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts.clone()),
        "scales": torch.nn.Parameter(torch.log(d3)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.3, device=dev))),
        "colors": torch.nn.Parameter(torch.zeros(N, 3, device=dev)),
    })
    bg = torch.nn.Parameter(torch.tensor([0.8, 0.85, 0.92], device=dev))
    lrs = dict(means=1.6e-4 * 20, scales=5e-3, quats=1e-3, opacities=5e-2, colors=2.5e-2)
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    opt_bg = torch.optim.Adam([bg], lr=1e-2)
    strategy = DefaultStrategy(refine_stop_iter=int(iters * 0.6), verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()
    mean_sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / iters))

    def render(vm):
        out, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1),
                                         torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                                         torch.sigmoid(params["colors"]), vm[None], K[None], W, H,
                                         packed=False, absgrad=False)
        img = out[0] + (1 - alpha[0]) * bg.clamp(0, 1)
        return img, info

    t0 = time.time()
    rng = np.random.default_rng(seed)
    for step in range(iters):
        i = int(rng.choice(train_ids))
        img, info = render(viewmats[i])
        gtim = imgs[i]
        l1 = (img - gtim).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gtim.permute(2, 0, 1)[None]))
        strategy.step_pre_backward(params, opts, state, step, info)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opt_bg.step()
        opt_bg.zero_grad(set_to_none=True)
        mean_sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 1000 == 0:
            print(f"  [{poses}] step {step} loss {loss.item():.4f} N={len(params['means'])} {time.time() - t0:.0f}s", flush=True)

    out_dir = res_dir / scene / "gs" / poses
    out_dir.mkdir(parents=True, exist_ok=True)
    psnrs, ssims = [], []
    with torch.no_grad():
        for j, i in enumerate(test):
            img, _ = render(viewmats[i])
            img = img.clamp(0, 1)
            mse = ((img - imgs[i]) ** 2).mean().item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
            ssims.append(ssim(img.permute(2, 0, 1)[None], imgs[i].permute(2, 0, 1)[None]).item())
            if j in (2, len(test) // 2, len(test) - 3):
                both = torch.cat([imgs[i], img], 1).cpu().numpy()
                imageio.imwrite(out_dir / f"test_{i:05d}.png", (both * 255).astype(np.uint8))
    r = dict(scene=scene, poses=poses, n_train=len(train_ids), n_test=len(test), n_gauss=len(params["means"]),
             n_init=N, psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), train_s=time.time() - t0)
    (out_dir / "metrics.json").write_text(json.dumps(r, indent=2))
    print(f"  [{poses}] PSNR {r['psnr']:.2f} dB  SSIM {r['ssim']:.3f}  ({r['n_gauss']} gaussians, {r['train_s']:.0f}s)")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slab")
    ap.add_argument("--poses", nargs="+", default=["gt"])
    ap.add_argument("--iters", type=int, default=7000)
    a = ap.parse_args()
    for p in a.poses:
        train(a.scene, p, a.iters)
