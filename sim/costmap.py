"""GT free-space / obstacle costmap for cost-aware waypoint following.

Layers (Hybrid plan — ground-truth perception, not learned):
  1. elevation / slope from terrain heightfield
  2. static OGM from MuJoCo geom footprints (inflated)
  3. dynamic OGM from moving actors (workers, trucks, loaders)

Local planner samples free headings so the robot zigzags around piles
instead of walking through inflated obstacle disks.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CostmapConfig:
    res: float = 0.12
    half: float = 5.0          # ego half-extent [m]
    # max climbable grade (percent). 15% ≈ 8.5°. Below → free; above → rising cost.
    slope_pct: float = 15.0
    look_ahead: float = 2.8
    robot_radius: float = 0.32
    obstacle_cost: float = 1.0
    dynamic_cost: float = 1.25
    slope_cost: float = 0.55
    clear_cost: float = 0.02
    lethal: float = 0.80       # above this → treated as blocked

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope_pct / 100.0)))


class GtCostmap:
    """Ego-centric cost grid in world frame, refreshed each nav tick."""

    def __init__(self, model: mujoco.MjModel, terrain=None, cfg: CostmapConfig | None = None):
        self.m = model
        self.terrain = terrain
        self.cfg = cfg or CostmapConfig()
        n = int(2 * self.cfg.half / self.cfg.res) + 1
        self.n = n
        self.cost = np.zeros((n, n), np.float32)
        self.origin = np.zeros(2)
        self.yaw = 0.0
        self._floor = set()
        for name in ("floor", "floor_mesh"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._floor.add(gid)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._robot = {b for b in range(model.nbody) if model.body_rootid[b] == root}
        self._collect_static()
        # body_id → list of geom ids (for multi-part actors)
        self._body_geoms: dict[int, list[int]] = {}
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            self._body_geoms.setdefault(bid, []).append(gid)

    def _collect_static(self):
        m = self.m
        out = []
        for gid in range(m.ngeom):
            if gid in self._floor:
                continue
            if m.geom_group[gid] != 2:
                continue
            bid = int(m.geom_bodyid[gid])
            if bid in self._robot:
                continue
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_HFIELD:
                continue
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                continue   # terrain visual mesh — use heightfield slope instead
            out.append(gid)
        self._static_gids = out

    def _stamp_disk(self, cx, cy, radius, value):
        cfg = self.cfg
        o = self.origin
        i0 = int((cx - radius - o[0]) / cfg.res)
        i1 = int((cx + radius - o[0]) / cfg.res) + 1
        j0 = int((cy - radius - o[1]) / cfg.res)
        j1 = int((cy + radius - o[1]) / cfg.res) + 1
        i0, i1 = max(i0, 0), min(i1, self.n)
        j0, j1 = max(j0, 0), min(j1, self.n)
        if i0 >= i1 or j0 >= j1:
            return
        xs = o[0] + (np.arange(i0, i1) + 0.5) * cfg.res
        ys = o[1] + (np.arange(j0, j1) + 0.5) * cfg.res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        self.cost[i0:i1, j0:j1] = np.maximum(self.cost[i0:i1, j0:j1], mask * value)

    def _stamp_obox(self, cx, cy, hx, hy, yaw, value, inflate=0.0):
        cfg = self.cfg
        o = self.origin
        pad = np.hypot(hx + inflate, hy + inflate)
        i0 = int((cx - pad - o[0]) / cfg.res)
        i1 = int((cx + pad - o[0]) / cfg.res) + 1
        j0 = int((cy - pad - o[1]) / cfg.res)
        j1 = int((cy + pad - o[1]) / cfg.res) + 1
        i0, i1 = max(i0, 0), min(i1, self.n)
        j0, j1 = max(j0, 0), min(j1, self.n)
        if i0 >= i1 or j0 >= j1:
            return
        xs = o[0] + (np.arange(i0, i1) + 0.5) * cfg.res
        ys = o[1] + (np.arange(j0, j1) + 0.5) * cfg.res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = X - cx, Y - cy
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        mask = (np.abs(lx) <= hx + inflate) & (np.abs(ly) <= hy + inflate)
        self.cost[i0:i1, j0:j1] = np.maximum(self.cost[i0:i1, j0:j1], mask * value)

    def _stamp_geom(self, gid, data, value, inflate):
        m, d = self.m, data
        pos = d.geom_xpos[gid]
        sz = m.geom_size[gid]
        if pos[2] - (sz[2] if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else 0) > 1.6:
            return
        typ = m.geom_type[gid]
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            if sz[2] < 0.12 and max(sz[0], sz[1]) > 1.5:
                return
            R = d.geom_xmat[gid].reshape(3, 3)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            self._stamp_obox(float(pos[0]), float(pos[1]), float(sz[0]), float(sz[1]),
                             yaw, value, inflate=inflate)
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = float(sz[0]) + inflate
            if r >= 0.10:
                self._stamp_disk(float(pos[0]), float(pos[1]), r, value)
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(sz[0]) + inflate
            if r >= 0.10:
                self._stamp_disk(float(pos[0]), float(pos[1]), r, value)
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(sz[0]) + inflate
            self._stamp_disk(float(pos[0]), float(pos[1]), r + float(sz[1]) * 0.3, value)
        else:
            r = float(np.linalg.norm(sz[:2])) + inflate
            if r >= 0.12:
                self._stamp_disk(float(pos[0]), float(pos[1]), r, value)

    def update(self, data: mujoco.MjData, actors=(), xy=None, yaw=0.0):
        """Rebuild ego cost: elevation + static OGM + dynamic OGM."""
        cfg = self.cfg
        if xy is None:
            xy = data.qpos[:2]
        self.yaw = float(yaw)
        self.origin[:] = xy - cfg.half
        self.cost.fill(0.0)
        m, d = self.m, data

        # --- elevation / slope layer (≤ slope_pct is free to climb) ---
        T = self.terrain
        if T is not None:
            xs = self.origin[0] + (np.arange(self.n) + 0.5) * cfg.res
            ys = self.origin[1] + (np.arange(self.n) + 0.5) * cfg.res
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            slope = T.slope_deg(X, Y)
            lim = cfg.slope_deg
            # only punish grades ABOVE the climb limit
            excess = np.clip((slope - lim) / max(lim, 1e-3), 0, 2.5)
            self.cost += excess * cfg.slope_cost
            # pits / cliffs: large drop relative to robot — not gentle hills
            z0 = float(T.height(float(xy[0]), float(xy[1])))
            H = T.height(X, Y)
            drop = z0 - H
            self.cost = np.maximum(self.cost, np.clip((drop - 0.55) / 0.45, 0, 1) * cfg.obstacle_cost)

        # --- static OGM ---
        inflate = cfg.robot_radius
        for gid in self._static_gids:
            self._stamp_geom(gid, d, cfg.obstacle_cost, inflate)

        # --- dynamic OGM (all geoms on actor body) ---
        for a in actors:
            if a.body_id < 0:
                continue
            gids = self._body_geoms.get(a.body_id, [])
            if not gids:
                # fallback disk at body origin
                pos = d.xpos[a.body_id]
                rad = 1.2 if any(k in a.name for k in ("tele", "truck", "loader", "dump")) else 0.55
                self._stamp_disk(float(pos[0]), float(pos[1]), rad + inflate, cfg.dynamic_cost)
                continue
            for gid in gids:
                self._stamp_geom(gid, d, cfg.dynamic_cost, inflate + 0.35)
            # short-horizon CV prediction so we don't walk into moving workers
            vel = np.asarray(getattr(a, "cur_vel", [0, 0, 0]), float)
            pos = d.xpos[a.body_id]
            rad = 0.70 if "worker" in a.name else 1.25
            for tau in (0.0, 0.45, 0.9):
                px = float(pos[0] + vel[0] * tau)
                py = float(pos[1] + vel[1] * tau)
                self._stamp_disk(px, py, rad + inflate * 0.6, cfg.dynamic_cost * (1.0 if tau == 0 else 0.9))

    def sample(self, x, y) -> float:
        cfg = self.cfg
        i = int((x - self.origin[0]) / cfg.res)
        j = int((y - self.origin[1]) / cfg.res)
        if i < 0 or j < 0 or i >= self.n or j >= self.n:
            return cfg.clear_cost
        return float(self.cost[i, j])

    def path_cost(self, xy, yaw, length=None, n=11) -> float:
        L = self.cfg.look_ahead if length is None else length
        c, s = np.cos(yaw), np.sin(yaw)
        acc = 0.0
        hit = 0.0
        for k in range(1, n + 1):
            t = L * k / n
            v = self.sample(xy[0] + c * t, xy[1] + s * t)
            acc += v
            if v >= self.cfg.lethal:
                # remaining distance heavily penalized (don't plan through)
                hit += (n + 1 - k) * self.cfg.obstacle_cost
                break
        return acc / n + hit / n

    def steer(self, xy, yaw, goal_xy) -> tuple[float, float, float]:
        """Short-horizon free-space picker.

        Samples headings around the goal bearing; refuses corridors that hit
        lethal cells. Returns (vx_scale, yaw_cmd_offset, best_cost).
        """
        cfg = self.cfg
        to_goal = goal_xy - xy
        goal_yaw = float(np.arctan2(to_goal[1], to_goal[0]))
        # denser fan + reverse headings for escape when already overlapping
        offsets = np.deg2rad(
            [-90, -75, -60, -45, -30, -18, -8, 0, 8, 18, 30, 45, 60, 75, 90]
        )
        best_c, best_off = 1e9, 0.0
        for off in offsets:
            y = goal_yaw + off
            c = self.path_cost(xy, y)
            # prefer goal; punish huge detours lightly
            c += 0.025 * abs(off)
            if c < best_c:
                best_c, best_off = c, off

        # also score current heading (hold if clearly better)
        c_cur = self.path_cost(xy, yaw) + 0.02
        if c_cur < best_c:
            best_c = c_cur
            best_off = ((yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi)

        chosen = goal_yaw + best_off
        near = self.path_cost(xy, chosen, length=1.0, n=7)
        mid = self.path_cost(xy, chosen, length=2.0, n=9)
        here = self.sample(float(xy[0]), float(xy[1]))

        # if overlapping an inflated obstacle, escape along clearest short heading
        if here >= cfg.lethal * 0.55:
            esc_c, esc_off = 1e9, 0.0
            for off in offsets:
                c = self.path_cost(xy, yaw + off, length=1.8, n=9)
                if c < esc_c:
                    esc_c, esc_off = c, off
            best_off = ((yaw + esc_off - goal_yaw + np.pi) % (2 * np.pi) - np.pi)
            best_c = esc_c
            # crawl out if escape corridor is free; only hard-stop if boxed in
            vx_scale = 0.0 if esc_c >= cfg.lethal * 0.5 else 0.45
        elif near >= cfg.lethal * 0.8:
            vx_scale = 0.0
        elif near > 0.45 or best_c > 0.7:
            vx_scale = 0.25
        elif mid > 0.45 or best_c > 0.35:
            vx_scale = 0.55
        elif best_c > 0.18:
            vx_scale = 0.80
        else:
            vx_scale = 1.0

        return float(vx_scale), float(best_off), float(best_c)

    def world_cells(self, max_cost=0.2):
        """Yield (x, y, cost) for cells above max_cost — for viewer overlay."""
        cfg = self.cfg
        for i in range(self.n):
            for j in range(self.n):
                c = self.cost[i, j]
                if c > max_cost:
                    x = self.origin[0] + (i + 0.5) * cfg.res
                    y = self.origin[1] + (j + 0.5) * cfg.res
                    yield x, y, float(c)
