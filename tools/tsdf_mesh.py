"""TSDF fusion + mesh extraction from the head RGB-D camera, per pose source.

Same images, different camera poses (ground truth / each estimator). Estimator
poses live in their own start frame, so each trajectory is aligned to the GT
world by its first pose only (what a robot mapping from its start would get).
Map-localized poses (maploc_*) are already in the map = GT frame.

Metrics against the mesh fused with ground-truth poses (sampled surface points):
  accuracy      mean distance  method -> GT mesh       (wrong geometry, doubled walls)
  completeness  % of GT surface within 5 cm of method  (missing / smeared geometry)

  python3 tools/tsdf_mesh.py --scene corridor --poses gt fmcw4d_leg fmcw3d legekf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fuse(cam_dir: Path, key: str, voxel=0.04, trunc=0.15, max_depth=8.0):
    meta = json.loads((cam_dir / "frames.json").read_text())
    frames = [f for f in meta["frames"] if f"c2w_{key}" in f and "depth" in f]
    intr = o3d.camera.PinholeCameraIntrinsic(meta["width"], meta["height"], meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    A = np.eye(4)
    if key not in ("gt",) and not key.startswith("maploc"):
        f0 = frames[0]
        A = np.array(f0["c2w_gt"]) @ np.linalg.inv(np.array(f0[f"c2w_{key}"]))
    for f in frames:
        color = o3d.geometry.Image(np.ascontiguousarray(imageio.imread(cam_dir / f["file"])[..., :3]))
        depth = o3d.geometry.Image(imageio.imread(cam_dir / f["depth"]).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        c2w = A @ np.array(f[f"c2w_{key}"])
        vol.integrate(rgbd, intr, np.linalg.inv(c2w))
    return vol.extract_triangle_mesh(), len(frames)


def sample(mesh, n=300_000):
    return np.asarray(mesh.sample_points_uniformly(n).points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slab")
    ap.add_argument("--poses", nargs="+", default=["gt", "fmcw4d_leg", "fmcw3d", "legekf"])
    a = ap.parse_args()
    cam = ROOT / "results" / a.scene / "rgbd"
    out = ROOT / "results" / a.scene / "tsdf"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    ref = None
    for k in ["gt"] + [p for p in a.poses if p != "gt"]:
        mesh, n = fuse(cam, k)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out / f"mesh_{k}.ply"), mesh.simplify_vertex_clustering(0.08))
        P = sample(mesh)
        if k == "gt":
            ref, ref_tree = P, cKDTree(P)
            res[k] = dict(frames=n, triangles=len(mesh.triangles))
        else:
            d_acc, _ = ref_tree.query(P)
            d_comp, _ = cKDTree(P).query(ref)
            res[k] = dict(frames=n, triangles=len(mesh.triangles), accuracy_mean=float(d_acc.mean()),
                          accuracy_p90=float(np.percentile(d_acc, 90)),
                          completeness_5cm=float((d_comp < 0.05).mean()),
                          chamfer=float(d_acc.mean() + d_comp.mean()))
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res[k].items()}, flush=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
