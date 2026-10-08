"""Procedural outdoor site terrain (MuJoCo heightfield).

Layers (all in metres):
  undulation   long-wavelength ground swell (graded earth is never flat)
  gravel       short-wavelength roughness
  ruts         twin wheel ruts along haul roads (polylines)
  features     soil stockpiles (Gaussian mounds), an excavation pit
  pads         flattened areas (start pad, container yard, slab) -> exactly flat

`Terrain.height(x, y)` is the bilinear height the simulator uses, so objects can
be placed on the ground and routes can be checked for slope.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class TerrainSpec:
    x0: float = -20.0
    y0: float = -25.0
    size_x: float = 80.0
    size_y: float = 50.0
    res: float = 0.1
    undulation: float = 0.25          # m, std of the long-wavelength swell
    undulation_wl: float = 12.0       # m
    gravel: float = 0.01              # m, std of short roughness
    rut_depth: float = 0.05
    roads: list = field(default_factory=list)       # list of (K,2) polylines
    mounds: list = field(default_factory=list)      # (x, y, height, radius)
    pits: list = field(default_factory=list)        # (x, y, depth, half_x, half_y)
    pads: list = field(default_factory=list)        # (x, y, half_x, half_y, height)
    seed: int = 0


class Terrain:
    def __init__(self, ts: TerrainSpec):
        self.ts = ts
        rng = np.random.default_rng(ts.seed)
        nx, ny = int(ts.size_x / ts.res) + 1, int(ts.size_y / ts.res) + 1
        self.xs = ts.x0 + np.arange(nx) * ts.res
        self.ys = ts.y0 + np.arange(ny) * ts.res
        X, Y = np.meshgrid(self.xs, self.ys)            # (ny, nx): MuJoCo rows = y
        H = gaussian_filter(rng.normal(size=(ny, nx)), ts.undulation_wl / ts.res / 2.5)
        H *= ts.undulation / max(H.std(), 1e-9)
        if ts.gravel:
            g = gaussian_filter(rng.normal(size=(ny, nx)), 0.6)
            H += g * ts.gravel / max(g.std(), 1e-9)
        for (x, y, h, r) in ts.mounds:
            H += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * r * r))
        for (x, y, d, hx, hy) in ts.pits:
            inside = np.clip(1.0 - np.maximum(np.abs(X - x) / hx, np.abs(Y - y) / hy), 0, 0.15) / 0.15
            H -= d * inside
        for road in ts.roads:
            road = np.asarray(road, float)
            dist = self._dist_to_polyline(X, Y, road)
            # graded road surface: undulation halved, then two ruts 1.8 m apart
            w = np.clip(1 - dist / 3.0, 0, 1)
            H = H * (1 - 0.5 * w)
            for off in (-0.9, 0.9):
                H -= ts.rut_depth * np.exp(-((dist - abs(off)) ** 2) / (2 * 0.18 ** 2)) * (dist < 2.0)
        for (x, y, hx, hy, h) in ts.pads:
            m = (np.abs(X - x) < hx) & (np.abs(Y - y) < hy)
            blend = gaussian_filter(m.astype(float), 8)
            H = H * (1 - blend) + h * blend
            H[m] = h
        self.H = H

    @staticmethod
    def _dist_to_polyline(X, Y, P):
        D = np.full(X.shape, np.inf)
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a
            L2 = max(ab @ ab, 1e-9)
            u = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
            D = np.minimum(D, np.hypot(X - (a[0] + u * ab[0]), Y - (a[1] + u * ab[1])))
        return D

    def height(self, x, y):
        ts = self.ts
        fx = (np.asarray(x) - ts.x0) / ts.res
        fy = (np.asarray(y) - ts.y0) / ts.res
        i0 = np.clip(np.floor(fx).astype(int), 0, len(self.xs) - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, len(self.ys) - 2)
        tx, ty = np.clip(fx - i0, 0, 1), np.clip(fy - j0, 0, 1)
        H = self.H
        return ((1 - tx) * (1 - ty) * H[j0, i0] + tx * (1 - ty) * H[j0, i0 + 1]
                + (1 - tx) * ty * H[j0 + 1, i0] + tx * ty * H[j0 + 1, i0 + 1])

    def slope_deg(self, x, y, d=0.5):
        gx = (self.height(x + d, y) - self.height(x - d, y)) / (2 * d)
        gy = (self.height(x, y + d) - self.height(x, y - d)) / (2 * d)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def add_to_spec(self, spec: mujoco.MjSpec, group=2, name="floor"):
        """Two copies of the same surface:
        - heightfield `name` (group 4: not ray-cast, not drawn) carries the foot contacts;
        - triangle mesh `name`_mesh (group `group`, no contacts) is what the LiDAR and the
          cameras see. MuJoCo ray-casts meshes through a BVH (~0.6 ms per 384 rays here)
          but heightfields by brute force (~70 ms), so this keeps LiDAR simulation fast."""
        ts = self.ts
        lo, hi = float(self.H.min()), float(self.H.max())
        span = max(hi - lo, 1e-3)
        hf = spec.add_hfield()
        hf.name = "site_terrain"
        hf.nrow, hf.ncol = self.H.shape
        hf.size = [ts.size_x / 2, ts.size_y / 2, span, 1.0]
        hf.userdata = ((self.H - lo) / span).astype(np.float32).reshape(-1).tolist()
        g = spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_HFIELD
        g.hfieldname = "site_terrain"
        g.pos = [ts.x0 + ts.size_x / 2, ts.y0 + ts.size_y / 2, lo]
        g.rgba = [0.55, 0.47, 0.38, 1]
        g.group = 4
        ny, nx = self.H.shape
        X, Y = np.meshgrid(self.xs, self.ys)
        V = np.c_[X.ravel(), Y.ravel(), self.H.ravel()]
        idx = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        F = np.r_[np.c_[a, b, d], np.c_[a, d, c]]
        ms = spec.add_mesh()
        ms.name = "site_terrain_mesh"
        ms.uservert = V.astype(np.float32).ravel().tolist()
        ms.userface = F.astype(np.int32).ravel().tolist()
        # dirt texture hides triangle facets (otherwise chase cam looks like water ripples)
        tex = spec.add_texture()
        tex.name = "t_dirt"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        tex.rgb1, tex.rgb2 = [0.55, 0.47, 0.38], [0.48, 0.40, 0.32]
        tex.width, tex.height = 512, 512
        tex.mark = mujoco.mjtMark.mjMARK_RANDOM
        tex.markrgb = [0.42, 0.36, 0.28]
        tex.random = 0.02
        mat = spec.add_material()
        mat.name = "m_dirt"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "t_dirt"
        mat.texrepeat = [8, 8]
        mat.texuniform = True
        gm = spec.worldbody.add_geom()
        gm.name = name + "_mesh"
        gm.type = mujoco.mjtGeom.mjGEOM_MESH
        gm.meshname = "site_terrain_mesh"
        gm.material = "m_dirt"
        gm.rgba = [1, 1, 1, 1]
        gm.group = group
        gm.contype = 0
        gm.conaffinity = 0
        return g
