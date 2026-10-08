"""Render head-camera images along a recorded trajectory (for Gaussian Splatting).

The camera is co-located with the LiDAR frame (head, pitched 8 deg down),
optical axis = LiDAR +x. Images are rendered from the *ground-truth* pose of
the recorded run; a scene-only model with procedural textures is used (the
robot body is not in the camera's view cone).

Writes <out>/images/*.png and <out>/frames.json with, per image, the time and
the camera-to-world matrix (OpenCV convention: x right, y down, z forward)
for ground truth and for every estimator result found under results/<scene>.

  MUJOCO_GL=glfw python3 -m sim.render_cam --scene slab --out results/slab/cam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from . import scenes as S
from .record import LIDAR_PITCH, LIDAR_POS, IMU_POS, quat_wxyz_to_R, rot_y

ROOT = Path(__file__).resolve().parents[1]

# body(IMU) <- camera(OpenCV) rotation: camera = LiDAR frame re-axed to x right, y down, z forward
R_BL = rot_y(LIDAR_PITCH)
R_LCV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # columns: cv x,y,z in LiDAR frame
R_BCV = R_BL @ R_LCV
P_BC = LIDAR_POS - IMU_POS


def add_materials(spec: mujoco.MjSpec):
    """Procedural textures so the photometric map has something to fit."""
    def tex(name, builtin, rgb1, rgb2, mark="random", random=0.04, w=256):
        t = spec.add_texture()
        t.name = name
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.builtin = builtin
        t.rgb1 = rgb1
        t.rgb2 = rgb2
        t.width = w
        t.height = w
        t.mark = getattr(mujoco.mjtMark, "mjMARK_RANDOM" if mark == "random" else "mjMARK_EDGE")
        t.markrgb = [min(1, c * 1.25) for c in rgb1]
        t.random = random
        return t

    def mat(name, texname, repeat):
        m = spec.add_material()
        m.name = name
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texname
        m.texrepeat = repeat
        m.texuniform = True
        return m

    tex("t_concrete", mujoco.mjtBuiltin.mjBUILTIN_FLAT, [0.62, 0.62, 0.60], [0.5, 0.5, 0.5], random=0.08)
    tex("t_cmu", mujoco.mjtBuiltin.mjBUILTIN_CHECKER, [0.58, 0.56, 0.52], [0.50, 0.48, 0.45], mark="edge")
    tex("t_wood", mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, [0.75, 0.58, 0.36], [0.55, 0.40, 0.22], random=0.05)
    tex("t_floor", mujoco.mjtBuiltin.mjBUILTIN_CHECKER, [0.52, 0.52, 0.50], [0.46, 0.46, 0.45], mark="edge", random=0.02)
    sky = spec.add_texture()
    sky.name = "t_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.62, 0.74, 0.88]
    sky.rgb2 = [0.92, 0.94, 0.96]
    sky.width = 256
    sky.height = 1536
    mat("m_concrete", "t_concrete", [2, 2])
    mat("m_cmu", "t_cmu", [2.5, 5])
    mat("m_wood", "t_wood", [1, 1])
    mat("m_floor", "t_floor", [1, 1])


def build_render_model(scene, seed):
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    add_materials(spec)
    S.build(spec, scene, seed=seed)
    for g in spec.geoms:
        rgba = tuple(np.round(g.rgba[:3], 2))
        if g.name == "floor":
            g.material = "m_floor"
        elif rgba == tuple(np.round(S.CONCRETE[:3], 2)):
            g.material = "m_concrete"
        elif rgba == tuple(np.round(S.CMU[:3], 2)):
            g.material = "m_cmu"
        elif rgba == tuple(np.round(S.WOOD[:3], 2)):
            g.material = "m_wood"
        if g.material:
            g.rgba = [1, 1, 1, 1]
    sun = spec.worldbody.add_light()
    sun.pos = [10, 0, 30]
    sun.dir = [0.2, 0.3, -1]
    sun.diffuse = [0.6, 0.6, 0.6]
    cam_body = spec.worldbody.add_body()
    cam_body.name = "cam_rig"
    cam_body.mocap = True
    cam = cam_body.add_camera()
    cam.name = "head"
    cam.fovy = 70.0
    # MuJoCo camera looks along -z with +y up; OpenCV looks along +z with +y down
    cam.quat = [0, 1, 0, 0]
    return spec.compile()


def interp_pose(t_query, tt, p, q):
    """Nearest-neighbour pose lookup (10 Hz estimates vs 2 Hz images)."""
    k = np.clip(np.searchsorted(tt, t_query), 1, len(tt) - 1)
    k = np.where(np.abs(tt[k - 1] - t_query) < np.abs(tt[k] - t_query), k - 1, k)
    return p[k], q[k], np.abs(tt[k] - t_query)


def cam_c2w(p_wb, q_wb):
    R_wb = quat_wxyz_to_R(q_wb)
    T = np.eye(4)
    T[:3, :3] = R_wb @ R_BCV
    T[:3, 3] = p_wb + R_wb @ P_BC
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--depth", action="store_true", help="also save depth (uint16 mm PNG)")
    a = ap.parse_args()

    seq, res, out = Path(a.data) / a.scene, Path(a.results) / a.scene, Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(seq / "gt_tum.txt")
    tg, pg, qg = gt[:, 0], gt[:, 1:4], gt[:, [7, 4, 5, 6]]

    ests = {}
    for d in sorted(res.iterdir()):
        f = d / "poses_tum.txt"
        if f.exists():
            e = np.loadtxt(f)
            ests[d.name] = (e[:, 0], e[:, 1:4], e[:, [7, 4, 5, 6]])
    t0 = max(e[0][0] for e in ests.values()) if ests else tg[0]
    times = np.arange(t0 + 0.5, tg[-1] - 0.2, 1.0 / a.hz)

    m = build_render_model(a.scene, a.seed)
    d = mujoco.MjData(m)
    mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cam_rig")]
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head")
    r = mujoco.Renderer(m, a.height, a.width)
    rd = None
    if a.depth:
        (out / "depth").mkdir(exist_ok=True)
        rd = mujoco.Renderer(m, a.height, a.width)
        rd.enable_depth_rendering()
    fy = 0.5 * a.height / np.tan(np.deg2rad(m.cam_fovy[cam_id]) / 2)
    frames = []
    for i, t in enumerate(times):
        p, q, _ = interp_pose(t, tg, pg, qg)
        T = cam_c2w(p, q)
        d.mocap_pos[mocap] = T[:3, 3]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].reshape(-1).copy())
        d.mocap_quat[mocap] = quat
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam_id)
        img = r.render()
        name = f"images/{i:05d}.png"
        imageio.imwrite(out / name, img)
        fr = dict(file=name, t=float(t), c2w_gt=T.tolist())
        if rd is not None:
            rd.update_scene(d, camera=cam_id)
            dep = rd.render()                                  # metres, z-depth
            dep = np.where((dep > 0.2) & (dep < 15.0), dep, 0.0)
            imageio.imwrite(out / f"depth/{i:05d}.png", (dep * 1000).astype(np.uint16))
            fr["depth"] = f"depth/{i:05d}.png"
        for k, (te, pe, qe) in ests.items():
            pp, qq, dt = interp_pose(t, te, pe, qe)
            if dt < 0.06:
                fr[f"c2w_{k}"] = cam_c2w(pp, qq).tolist()
        frames.append(fr)
    meta = dict(scene=a.scene, width=a.width, height=a.height, fx=fy, fy=fy, cx=a.width / 2, cy=a.height / 2,
                methods=sorted(ests), frames=frames)
    (out / "frames.json").write_text(json.dumps(meta))
    print(f"rendered {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
