"""Construction-site scenes for the G1 localization study.

Target domain: construction / mining / energy sites, so every scene is
a building under construction: concrete slabs, column grids, partial upper
slabs, material stacks, scaffolding, long service corridors, moving workers
and machines.

All static scene geoms go in geom group 2 and moving actors in group 3, so
the LiDAR ray caster can ignore the robot's own geometry (groups 0/1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

STATIC_GROUP = 2
DYNAMIC_GROUP = 3

CONCRETE = [0.62, 0.62, 0.60, 1]
CMU = [0.55, 0.53, 0.50, 1]
WOOD = [0.70, 0.55, 0.35, 1]
STEEL = [0.35, 0.37, 0.40, 1]
SAFETY = [0.95, 0.75, 0.10, 1]
PLASTIC = [0.20, 0.35, 0.70, 1]


@dataclass
class Actor:
    """A kinematic (mocap) body moving along a closed polyline at constant speed."""
    name: str
    path: np.ndarray            # (K, 2) xy waypoints, traversed back and forth
    speed: float
    z: float                    # body centre height
    phase: float = 0.0          # start offset along the path [m]
    body_id: int = -1
    mocap_id: int = -1
    z_fn: object = None         # optional ground height h(x, y): z is then height above ground

    def __post_init__(self):
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.total = float(self.seg_len.sum())

    def state(self, t: float):
        """Position, velocity and yaw at time t (ping-pong along the polyline)."""
        s = (self.phase + self.speed * t) % (2 * self.total)
        direction = 1.0
        if s > self.total:
            s = 2 * self.total - s
            direction = -1.0
        acc = np.concatenate([[0], np.cumsum(self.seg_len)])
        i = int(np.clip(np.searchsorted(acc, s, side="right") - 1, 0, len(self.seg_len) - 1))
        u = self.path[i + 1] - self.path[i]
        u = u / max(np.linalg.norm(u), 1e-9)
        heading = u * direction
        xy = self.path[i] + u * (s - acc[i])
        z = self.z + (float(self.z_fn(xy[0], xy[1])) if self.z_fn is not None else 0.0)
        pos = np.array([xy[0], xy[1], z])
        vel = np.array([heading[0], heading[1], 0.0]) * self.speed
        yaw = float(np.arctan2(heading[1], heading[0]))
        return pos, vel, yaw


@dataclass
class Slider:
    """A loose sheet on a slide joint, position-servoed along x: x(t) = A sin(2 pi t / T + phase).

    Unlike a mocap body it has a real velocity, so a foot standing on it is
    carried along by friction.
    """
    name: str
    amplitude: float
    period: float
    phase: float = 0.0
    qpos_adr: int = -1
    qvel_adr: int = -1
    act_id: int = -1
    body_id: int = -1

    def target(self, t):
        return self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)


@dataclass
class Scene:
    name: str
    description: str
    waypoints: np.ndarray                     # (K, 2) robot route, xy
    actors: list = field(default_factory=list)
    stops: dict = field(default_factory=dict)  # waypoint index -> stand-still seconds
    slip_patches: list = field(default_factory=list)  # (cx, cy, hx, hy, mu)
    sliders: list = field(default_factory=list)


class _Builder:
    def __init__(self, spec: mujoco.MjSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.n = 0

    def box(self, pos, half, rgba=CONCRETE, yaw=0.0, group=STATIC_GROUP, body=None,
            collide=True, friction=None, priority=0):
        parent = body if body is not None else self.spec.worldbody
        g = parent.add_geom()
        g.name = f"s{self.n}"
        self.n += 1
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(half)
        g.pos = list(pos)
        if yaw:
            g.quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        g.rgba = rgba
        g.group = group
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        if friction is not None:
            g.friction = [friction, 0.005, 0.0001]
            g.priority = priority
        return g

    def cyl(self, pos, radius, half_h, rgba=STEEL, euler=None, group=STATIC_GROUP, collide=True, body=None):
        parent = body if body is not None else self.spec.worldbody
        g = parent.add_geom()
        g.name = f"s{self.n}"
        self.n += 1
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [radius, half_h, 0]
        g.pos = list(pos)
        if euler is not None:
            r = _euler_to_quat(*euler)
            g.quat = list(r)
        g.rgba = rgba
        g.group = group
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    def sphere(self, pos, radius, rgba=STEEL, group=STATIC_GROUP, collide=True, body=None):
        parent = body if body is not None else self.spec.worldbody
        g = parent.add_geom()
        g.name = f"s{self.n}"
        self.n += 1
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [radius, 0, 0]
        g.pos = list(pos)
        g.rgba = rgba
        g.group = group
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    def capsule(self, pos, radius, half_h, rgba=STEEL, euler=None, group=STATIC_GROUP, collide=True, body=None):
        parent = body if body is not None else self.spec.worldbody
        g = parent.add_geom()
        g.name = f"s{self.n}"
        self.n += 1
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [radius, half_h, 0]
        g.pos = list(pos)
        if euler is not None:
            g.quat = list(_euler_to_quat(*euler))
        g.rgba = rgba
        g.group = group
        if not collide:
            g.contype = 0
            g.conaffinity = 0
        return g

    # ----- composite construction elements -----
    def floor(self):
        g = self.spec.worldbody.add_geom()
        g.name = "floor"
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [0, 0, 0.05]
        g.rgba = [0.5, 0.5, 0.48, 1]
        g.group = STATIC_GROUP

    def column(self, x, y, h=3.3, w=0.25):
        self.box([x, y, h / 2], [w, w, h / 2], CONCRETE)

    def slab_above(self, x0, x1, y0, y1, z=3.3, t=0.25):
        self.box([(x0 + x1) / 2, (y0 + y1) / 2, z + t / 2], [(x1 - x0) / 2, (y1 - y0) / 2, t / 2], CONCRETE)

    def wall(self, p0, p1, h=3.0, t=0.1, rgba=CMU, openings=()):
        """Wall from p0 to p1 with optional (s0, s1, z0, z1) openings along its length."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / L
        yaw = np.arctan2(u[1], u[0])
        cuts = sorted(openings)
        s = 0.0
        pieces = []
        for (a, b, z0, z1) in cuts:
            if a > s:
                pieces.append((s, a, 0, h))
            if z0 > 0:
                pieces.append((a, b, 0, z0))       # sill
            if z1 < h:
                pieces.append((a, b, z1, h))       # lintel
            s = b
        if s < L:
            pieces.append((s, L, 0, h))
        for (a, b, z0, z1) in pieces:
            c = p0 + u * (a + b) / 2
            self.box([c[0], c[1], (z0 + z1) / 2], [(b - a) / 2, t / 2, (z1 - z0) / 2], rgba, yaw=yaw)

    def guardrail(self, p0, p1, post_every=2.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / L
        yaw = np.arctan2(u[1], u[0])
        for s in np.arange(0, L + 1e-6, post_every):
            q = p0 + u * s
            self.box([q[0], q[1], 0.55], [0.03, 0.03, 0.55], SAFETY)
        c = (p0 + p1) / 2
        for z in (0.5, 1.05):
            self.box([c[0], c[1], z], [L / 2, 0.02, 0.02], SAFETY, yaw=yaw)

    def pallet_stack(self, x, y, yaw=0.0):
        h = self.rng.uniform(0.4, 1.5)
        self.box([x, y, 0.07], [0.6, 0.5, 0.07], WOOD, yaw=yaw)
        self.box([x, y, 0.14 + h / 2], [0.55, 0.45, h / 2], CMU, yaw=yaw)

    def rebar_bundle(self, x, y, yaw=0.0, L=6.0):
        self.box([x, y, 0.12], [L / 2, 0.2, 0.12], [0.45, 0.25, 0.15, 1], yaw=yaw)

    def scaffold(self, x, y, w=2.5, d=1.2, h=4.0, yaw=0.0):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = [R @ np.array(p) + [x, y] for p in [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]]
        for p in corners:
            self.cyl([p[0], p[1], h / 2], 0.025, h / 2)
        for z in np.arange(1.0, h + 0.01, 1.0):
            for i in range(4):
                a, b = corners[i], corners[(i + 1) % 4]
                m = (a + b) / 2
                L = np.linalg.norm(b - a)
                yy = np.arctan2(b[1] - a[1], b[0] - a[0])
                self.box([m[0], m[1], z], [L / 2, 0.02, 0.02], STEEL, yaw=yy)
            # plank deck
            self.box([x, y, z + 0.03], [w / 2 - 0.05, d / 2 - 0.05, 0.02], WOOD, yaw=yaw)

    def formwork_panel(self, x, y, yaw=0.0):
        self.box([x, y, 0.9], [1.2, 0.03, 0.9], WOOD, yaw=yaw)

    def misc_box(self, x, y, half, rgba, yaw=0.0):
        self.box([x, y, half[2]], half, rgba, yaw=yaw)

    def slider_board(self, name, cx, cy, half=(1.3, 0.7, 0.006), mass=40.0):
        b = self.spec.worldbody.add_body()
        b.name = name
        b.pos = [cx, cy, half[2]]
        j = b.add_joint()
        j.name = name + "_x"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = [1, 0, 0]
        j.damping = [50.0, 0.0, 0.0]   # mujoco 3.4+: damping is a 3-vector
        g = self.box([0, 0, 0], list(half), [0.75, 0.6, 0.4, 1], group=DYNAMIC_GROUP, body=b)
        g.mass = mass
        a = self.spec.add_actuator()
        a.name = name + "_servo"
        a.target = j.name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.set_to_position(kp=40000.0, kv=4000.0)
        return b

    def _mocap(self, name):
        b = self.spec.worldbody.add_body()
        b.name = name
        b.mocap = True
        b.pos = [0, 0, -50]
        return b

    def actor_body(self, name, half, rgba, collide=False):
        """Legacy single-box actor (kept for ad-hoc use). Prefer shaped helpers below."""
        b = self._mocap(name)
        self.box([0, 0, half[2]], half, rgba, group=DYNAMIC_GROUP, body=b, collide=collide)
        return b

    def actor_worker(self, name, collide=False):
        """Simple construction worker: hi-vis vest, hard hat, legs/arms."""
        b = self._mocap(name)
        skin = [0.82, 0.65, 0.52, 1]
        pants = [0.22, 0.28, 0.38, 1]
        vest = [0.95, 0.55, 0.08, 1]
        hat = SAFETY
        boot = [0.15, 0.12, 0.10, 1]
        # body origin = feet on ground; forward = +X
        self.box([0.02, 0.11, 0.06], [0.12, 0.07, 0.06], boot, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([0.02, -0.11, 0.06], [0.12, 0.07, 0.06], boot, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.capsule([0.0, 0.11, 0.48], 0.07, 0.28, pants, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.capsule([0.0, -0.11, 0.48], 0.07, 0.28, pants, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([0.0, 0.0, 1.05], [0.16, 0.22, 0.28], vest, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.capsule([0.0, 0.28, 1.05], 0.05, 0.22, vest, euler=(0, np.pi / 2, 0.35),
                     group=DYNAMIC_GROUP, body=b, collide=collide)
        self.capsule([0.0, -0.28, 1.05], 0.05, 0.22, vest, euler=(0, np.pi / 2, -0.35),
                     group=DYNAMIC_GROUP, body=b, collide=collide)
        self.sphere([0.0, 0.0, 1.48], 0.12, skin, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([0.0, 0.0, 1.62], [0.14, 0.14, 0.07], hat, group=DYNAMIC_GROUP, body=b, collide=collide)
        return b

    def actor_dump_truck(self, name, collide=False):
        """Yellow dump truck: cab + bed + wheels (body origin at ground, +X forward)."""
        b = self._mocap(name)
        yellow = MACHINE
        cab = [0.85, 0.55, 0.12, 1]
        glass = [0.45, 0.65, 0.75, 1]
        tire = [0.12, 0.12, 0.12, 1]
        chassis = [0.25, 0.25, 0.22, 1]
        # chassis / frame
        self.box([0.2, 0.0, 0.55], [2.6, 1.15, 0.22], chassis, group=DYNAMIC_GROUP, body=b, collide=collide)
        # dump bed
        self.box([-0.55, 0.0, 1.35], [1.9, 1.2, 0.75], yellow, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([-0.55, 0.0, 2.05], [1.85, 1.15, 0.08], [0.75, 0.55, 0.12, 1],
                 group=DYNAMIC_GROUP, body=b, collide=collide)
        # cab
        self.box([2.15, 0.0, 1.35], [0.75, 1.05, 0.85], cab, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([2.55, 0.0, 1.55], [0.35, 0.95, 0.45], glass, group=DYNAMIC_GROUP, body=b, collide=collide)
        # wheels (cyl along Y)
        for x, y in [(-1.6, 1.15), (-1.6, -1.15), (0.3, 1.15), (0.3, -1.15), (2.0, 1.15), (2.0, -1.15)]:
            self.cyl([x, y, 0.45], 0.42, 0.18, tire, euler=(np.pi / 2, 0, 0),
                     group=DYNAMIC_GROUP, body=b, collide=collide)
        return b

    def actor_loader(self, name, collide=False):
        """Wheel loader / telehandler: body, cab, boom, bucket."""
        b = self._mocap(name)
        yellow = MACHINE
        cab = [0.25, 0.28, 0.32, 1]
        glass = [0.45, 0.65, 0.75, 1]
        tire = [0.12, 0.12, 0.12, 1]
        steel = STEEL
        self.box([0.0, 0.0, 0.75], [1.7, 1.05, 0.55], yellow, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([-0.55, 0.0, 1.55], [0.7, 0.85, 0.55], cab, group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([-0.35, 0.0, 1.7], [0.45, 0.75, 0.35], glass, group=DYNAMIC_GROUP, body=b, collide=collide)
        # boom + bucket reaching forward
        self.box([1.6, 0.0, 1.55], [1.4, 0.12, 0.12], steel, yaw=0.25,
                 group=DYNAMIC_GROUP, body=b, collide=collide)
        self.box([2.9, 0.0, 0.85], [0.35, 0.85, 0.45], steel, group=DYNAMIC_GROUP, body=b, collide=collide)
        for x, y in [(-1.0, 1.05), (-1.0, -1.05), (1.1, 1.05), (1.1, -1.05)]:
            self.cyl([x, y, 0.55], 0.5, 0.22, tire, euler=(np.pi / 2, 0, 0),
                     group=DYNAMIC_GROUP, body=b, collide=collide)
        return b


def _euler_to_quat(rx, ry, rz):
    cr, sr = np.cos(rx / 2), np.sin(rx / 2)
    cp, sp = np.cos(ry / 2), np.sin(ry / 2)
    cy, sy = np.cos(rz / 2), np.sin(rz / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


# ---------------------------------------------------------------- scenes
def _slab_site(b: _Builder, ceiling=True, change=0):
    """Ground floor of a concrete frame building under construction (~40 x 24 m).

    change = 0: the surveyed site ("day 0", the prior map).
    change = 1: "day 1" - materials moved/delivered, one scaffold struck.
    change = 2: "day 7" - additionally partition walls built, shoring props for the
                next pour, upper slab extended east, plastic sheeting hung.
    The column grid, perimeter walls and core never change (the structural frame).
    """
    rng2 = np.random.default_rng(1000 + change)   # day-N changes; day 0 layout is untouched
    b.floor()
    xs = np.arange(0, 31, 6.0)
    ys = [-9.0, -3.0, 3.0, 9.0]
    for x in xs:
        for y in ys:
            b.column(x, y)
    if ceiling:
        # upper slab poured over the west half only; east half is open to sky
        b.slab_above(-5, 19.5 if change < 2 else 25.5, -12, 12)
        # drop beams under the slab along the column lines
        for y in ys:
            b.box([7.25, y, 3.3 - 0.25], [12.25, 0.2, 0.25], CONCRETE)
    # finished CMU wall on the west, partial walls with window openings N/S
    b.wall([-5, -12], [-5, 12], openings=[(10.5, 13.5, 0, 2.2)])
    b.wall([-5, 12], [14, 12], openings=[(3, 5, 0.9, 2.2), (8, 10, 0.9, 2.2), (13, 15, 0.9, 2.2)])
    b.wall([-5, -12], [10, -12], openings=[(4, 6, 0.9, 2.2), (9, 11, 0.9, 2.2)])
    # open slab edges further east get guardrails
    b.guardrail([14, 12], [35, 12])
    b.guardrail([10, -12], [35, -12])
    b.guardrail([35, -12], [35, 12])
    # stair / elevator core
    b.wall([30.5, 4.5], [34.5, 4.5], h=3.3, t=0.2, rgba=CONCRETE, openings=[(1.5, 2.6, 0, 2.1)])
    b.wall([30.5, 10.5], [34.5, 10.5], h=3.3, t=0.2, rgba=CONCRETE)
    b.wall([30.5, 4.5], [30.5, 10.5], h=3.3, t=0.2, rgba=CONCRETE)
    b.wall([34.5, 4.5], [34.5, 10.5], h=3.3, t=0.2, rgba=CONCRETE)
    # scaffolding along walls
    b.scaffold(-3.6, -6.0, yaw=np.pi / 2)
    if change == 0:
        b.scaffold(4.0, 10.6)          # struck on day 1
    b.scaffold(22.0, -10.6, h=5.0)
    # materials, kept off the robot aisles (y in [-1.5, 1.5] and [4.5, 7.5])
    rng = b.rng
    for _ in range(14):
        x = rng.uniform(-3, 33)
        y = rng.choice([rng.uniform(-11, -4.2), rng.uniform(-2.2, -1.9), rng.uniform(8.3, 11)])
        if abs(x - np.round(x / 6) * 6) < 1.2 and min(abs(y - yy) for yy in ys) < 1.2:
            continue
        yaw = rng.uniform(0, np.pi)
        h = b.rng.uniform(0.4, 1.5)    # keep day-0 rng stream identical
        if change and rng2.random() < 0.45:
            continue                   # stack used up / moved away
        b.rng, saved = _FixedH(h), b.rng
        b.pallet_stack(x, y, yaw=yaw)
        b.rng = saved
    if change:
        for _ in range(6 * change):     # new deliveries
            x = rng2.uniform(-3, 33)
            y = rng2.choice([rng2.uniform(-11, -4.2), rng2.uniform(8.3, 11)])
            b.pallet_stack(x, y, yaw=rng2.uniform(0, np.pi))
    if change == 0:
        b.rebar_bundle(12.0, -6.0, yaw=0.05)
        b.rebar_bundle(26.0, -6.5, yaw=-0.1)
    else:
        b.rebar_bundle(20.0, -7.0, yaw=0.4)
    b.rebar_bundle(16.0, 10.6, yaw=0.0, L=4.0)
    for k in range(5):
        if change == 0:
            b.formwork_panel(8.5 + 1.0 * k, -4.6, yaw=0.0)
        else:
            b.formwork_panel(22.5 + 1.0 * k, 10.0, yaw=0.0)
    if change >= 2:
        # CMU partition walls going up on the column lines (off the robot aisles)
        b.wall([6.25, -3.0], [11.75, -3.0], h=2.4, t=0.15)
        b.wall([12.25, 9.0], [17.75, 9.0], h=2.4, t=0.15)
        b.wall([0.25, -9.0], [5.75, -9.0], h=1.2, t=0.15)      # half-built
        # shoring props for the next pour (dense post forest)
        for x in np.arange(20.5, 30.0, 1.2):
            for y in list(np.arange(-10.5, -4.0, 1.2)) + list(np.arange(8.5, 11.5, 1.2)):
                b.cyl([x, y, 1.65], 0.03, 1.65)
        # plastic sheeting hung from the slab edge
        b.box([19.5, -7.5, 1.7], [0.01, 3.5, 1.6], PLASTIC)
    b.misc_box(-2.5, 9.5, [0.6, 0.6, 1.2], PLASTIC)                     # site toilet
    b.misc_box(27.0, 9.8, [1.0, 0.6, 0.7], [0.15, 0.45, 0.20, 1])       # generator
    if change == 0:
        b.misc_box(17.0, -9.5, [1.5, 1.0, 1.1], STEEL, yaw=0.3)         # gang box
    else:
        b.misc_box(9.0, 10.3, [1.5, 1.0, 1.1], STEEL, yaw=-0.2)         # gang box moved


# ---------------------------------------------------------------- outdoor site
EARTH = [0.55, 0.47, 0.38, 1]
CONTAINER = [0.20, 0.42, 0.62, 1]
MACHINE = [0.93, 0.68, 0.12, 1]
PIPE = [0.25, 0.25, 0.27, 1]

# haul-road loop the robot walks (45-degree corners: gentler turns on uneven ground)
OUTDOOR_ROUTE = np.array([[0.0, 0.0], [32.0, 0.0], [40.0, 8.0], [40.0, 14.0], [37.0, 17.0], [14.0, 17.0],
                          [6.0, 9.0], [0.0, 0.0]])   # every corner <= 45 deg

# Weave past barriers/rocks, then over a ~12% climbable berm (under 15% limit).
OUTDOOR_WEAVE = np.array([
    [0.0, 0.0],
    [8.0, -1.8],
    [13.0, 1.6],
    [16.5, 0.6],   # climb crest
    [20.0, -1.8],
    [26.0, 2.0],
    [32.0, -1.0],
    [38.0, 7.0],
    [36.0, 14.0],
    [18.0, 15.0],
    [8.0, 7.0],
    [0.0, 0.0],
], float)


def _outdoor_site(b: _Builder, actors=True, change=0, route=None, extra_rocks=False, climb_ramp=False):
    """Open-air site of an energy/infrastructure project: graded haul roads over rough
    earth, soil stockpiles, an excavation, a steel-frame building going up, a pipe
    yard, site containers, light towers, fencing, machines and workers.

    The robot walks the graded road (pretrained G1 policy limits: ~0.15 m swell,
    5 mm gravel, 2 cm ruts); off-road ground is rougher and only seen by the LiDAR."""
    from .terrain import Terrain, TerrainSpec
    route = OUTDOOR_ROUTE if route is None else np.asarray(route, float)
    side_road = np.array([[-12.0, -6.0], [52.0, -6.0]])
    ts = TerrainSpec(x0=-22, y0=-26, size_x=84, size_y=54, undulation=0.40, undulation_wl=10.0, gravel=0.01,
                     rut_depth=0.012, roads=[route, side_road],
                     mounds=[(22.0, 8.5, 2.6, 2.6), (50.0, 4.0, 3.2, 3.5), (-12.0, -18.0, 2.2, 3.0),
                             (27.0, 26.0, 1.8, 2.5)],
                     pits=[(30.0, -16.0, 2.5, 5.0, 4.0)],
                     pads=[(0.0, 0.0, 3.5, 3.5, 0.0), (-11.0, 9.0, 5.0, 5.0, 0.05), (8.0, -16.0, 8.0, 5.0, 0.1)],
                     seed=3)
    T = Terrain(ts)
    # graded road: the road layer halves the swell; flatten it further to the safe band
    X, Y = np.meshgrid(T.xs, T.ys)
    for road in (route, side_road):
        w = np.clip(1 - T._dist_to_polyline(X, Y, road) / 3.5, 0, 1)
        T.H = T.H * (1 - 0.8 * w)
    from scipy.ndimage import gaussian_filter
    # climbable berm on the weave road: target ~10–12% grade (under 15% nav limit)
    if climb_ramp:
        # cosine bump: max grade ≈ (π/2)*(peak/half) ; peak=0.28, half=7 → ≈6.3%
        x0, x1, peak = 11.0, 25.0, 0.28
        mid = 0.5 * (x0 + x1)
        half = 0.5 * (x1 - x0)
        bump = peak * 0.5 * (1 + np.cos(np.pi * np.clip((X - mid) / half, -1, 1)))
        bump = np.where(np.abs(X - mid) <= half, bump, 0.0)
        near = np.clip(1 - T._dist_to_polyline(X, Y, route) / 4.5, 0, 1)
        base = float(np.median(T.H[near > 0.5])) if np.any(near > 0.5) else 0.0
        T.H = T.H * (1 - near) + (base + bump) * near
        Hs = gaussian_filter(T.H, 3.0)
        T.H = np.where(near > 0.05, Hs, T.H)
    # turning in place on uneven ground is where the flat-ground policy falls:
    # grade a small level pad at every route corner (at the local road height)
    for (x, y) in route[1:-1]:
        # keep crest waypoint elevated — only flatten non-ramp corners
        if climb_ramp and 12.0 < x < 22.0:
            continue
        h = float(T.height(x, y))
        m = (np.abs(X - x) < 2.5) & (np.abs(Y - y) < 2.5)
        blend = gaussian_filter(m.astype(float), 6)
        T.H = T.H * (1 - blend) + h * blend
        T.H[m] = h
    T.add_to_spec(b.spec)
    gz = T.height

    def on_ground(x, y, half, rgba, yaw=0.0):
        c, s_ = np.cos(yaw), np.sin(yaw)
        corners = [(x + c * dx - s_ * dy, y + s_ * dx + c * dy) for dx in (-half[0], half[0]) for dy in (-half[1], half[1])]
        z0 = min(float(gz(*p)) for p in corners) - 0.05         # bed slightly into the soil
        b.box([x, y, z0 + half[2]], half, rgba, yaw=yaw)
        return z0

    # site containers (offices / stores), two stacked
    for i, y in enumerate([6.5, 9.2, 11.9]):
        on_ground(-11.0, y, [3.05, 1.22, 1.3], CONTAINER)
    b.box([-11.0, 7.85, 0.05 + 2.6 + 1.3], [3.05, 1.22, 1.3], CONTAINER)
    on_ground(52.0, 21.0, [3.05, 1.22, 1.3], [0.55, 0.20, 0.15, 1], yaw=0.3)
    # steel-frame building on its slab pad
    bx, by = 8.0, -16.0
    for cx in (bx - 6, bx, bx + 6):
        for cy in (by - 3, by + 3):
            b.box([cx, cy, 0.1 + 4.0], [0.15, 0.15, 4.0], STEEL)
    for z in (4.0, 8.0):
        for cy in (by - 3, by + 3):
            b.box([bx, cy, 0.1 + z], [6.15, 0.12, 0.2], STEEL)
        for cx in (bx - 6, bx, bx + 6):
            b.box([cx, by, 0.1 + z], [0.12, 3.15, 0.2], STEEL)
    b.box([bx - 3, by, 0.1 + 4.25], [3.0, 3.0, 0.05], [0.6, 0.6, 0.62, 1])          # first metal deck
    for k in range(4):                                                              # rebar cages
        b.box([bx + 9.5, by - 4 + 2.2 * k, 0.1 + 0.35], [3.0, 0.35, 0.35], [0.45, 0.25, 0.15, 1])
    # pipe yard (energy line): pyramid stack of 0.9 m pipes
    rows = [(4, 0.45), (3, 1.23), (2, 2.01)]
    for n, z in rows:
        for k in range(n):
            y = 23.0 + (k - (n - 1) / 2) * 0.95
            b.cyl([14.0, y, float(gz(14.0, y)) + z], 0.45, 6.0, PIPE, euler=(0, np.pi / 2, 0))
    # parked excavator by the pit: tracks, house, boom
    ex, ey = 37.5, -12.0
    z0 = on_ground(ex, ey, [2.2, 1.6, 0.45], [0.2, 0.2, 0.2, 1])
    b.box([ex, ey, z0 + 0.9 + 0.7], [1.6, 1.3, 0.7], MACHINE)
    b.box([ex - 2.6, ey, z0 + 2.6], [1.8, 0.25, 0.25], MACHINE, yaw=0.0)
    b.box([ex - 4.4, ey, z0 + 1.6], [0.25, 0.25, 1.1], MACHINE)
    # light towers
    for (x, y) in [(16.0, 4.0), (42.0, 22.0), (-4.0, -8.0), (44.0, -2.0)]:
        z = float(gz(x, y))
        b.cyl([x, y, z + 4.0], 0.12, 4.0, [0.85, 0.85, 0.85, 1])
        b.box([x, y, z + 8.3], [0.6, 0.15, 0.35], [0.9, 0.9, 0.9, 1])
    # temporary fencing on the site boundary (mesh panels behave like thin walls for a LiDAR)
    for x0 in np.arange(-20.0, 60.0, 3.5):
        for y in (25.0, -25.0):
            z = float(gz(x0 + 1.75, y))
            b.box([x0 + 1.75, y, z + 1.0], [1.7, 0.02, 1.0], [0.75, 0.75, 0.72, 1])
    for y0 in np.arange(-25.0, 25.0, 3.5):
        z = float(gz(60.0, y0 + 1.75))
        b.box([60.0, y0 + 1.75, z + 1.0], [0.02, 1.7, 1.0], [0.75, 0.75, 0.72, 1])
    # jersey barriers and cones along the road
    for x in np.arange(12.0, 30.0, 2.5):
        on_ground(x, -2.8, [1.2, 0.3, 0.4], [0.9, 0.9, 0.88, 1])
    for k, (x, y) in enumerate([(33.5, 4.0), (36.0, 6.5), (38.5, 3.0), (11.0, 16.5), (5.5, 5.0)]):   # >= 1.7 m off the route
        z = float(gz(x, y))
        b.cyl([x, y, z + 0.35], 0.15, 0.35, SAFETY)
    on_ground(-4.0, 14.0, [0.6, 0.6, 1.2], PLASTIC)            # toilet
    on_ground(-6.0, -3.0, [1.0, 0.6, 0.7], [0.15, 0.45, 0.20, 1])   # generator
    if extra_rocks:
        # keep rocks off the climb berm (x≈12–22 on-road)
        for (x, y, hx, hy, hz) in [
            (10.5, -1.9, 0.55, 0.45, 0.55),
            (18.5, 2.6, 0.6, 0.5, 0.65),
            (24.5, -2.0, 0.5, 0.45, 0.5),
            (29.5, 2.2, 0.7, 0.55, 0.7),
        ]:
            on_ground(x, y, [hx, hy, hz], [0.45, 0.38, 0.32, 1])

    acts = []
    if actors:
        b.actor_dump_truck("dump_truck")
        acts.append(Actor("dump_truck", side_road, 3.0, 0.02, phase=20.0, z_fn=gz))
        b.actor_loader("loader")
        acts.append(Actor("loader", np.array([[44.0, -6.0], [48.0, 0.5], [44.0, 12.0]]), 1.8, 0.02, z_fn=gz))
        for i, (path, sp, ph) in enumerate([
                ([[-2, -3.5], [30, -3.5]], 1.2, 0),
                # crosses weave later / wider so robot can stop-and-yield
                ([[8, 3.8], [28, -3.2]], 0.95, 8.0),
                ([[20, 20], [6, 20]], 1.3, 2), ([[2, 12], [-6, 3]], 0.9, 1),
                ([[26, -9], [34, -9]], 1.1, 3), ([[5, -10], [12, -10]], 0.8, 0)]):
            nm = f"worker{i}"
            b.actor_worker(nm)
            acts.append(Actor(nm, np.array(path, float), sp, 0.0, phase=ph, z_fn=gz))
    return T, acts


class _FixedH:
    """Stands in for the rng inside pallet_stack so a pre-drawn height is reused."""
    def __init__(self, h):
        self.h = h

    def uniform(self, *a, **k):
        return self.h


def _corridor(b: _Builder):
    b.floor()
    L, w, h = 52.0, 1.25, 2.8
    b.box([L / 2 - 2, 0, h + 0.1], [L / 2 + 0.5, w + 0.2, 0.1], CONCRETE)     # ceiling
    doors_n = [(s, s + 0.9, 0, 2.1) for s in np.arange(6.0, L - 4, 12.0)]
    doors_s = [(s, s + 0.9, 0, 2.1) for s in np.arange(12.0, L - 4, 12.0)]
    b.wall([-2, w], [L - 2, w], h=h, rgba=[0.85, 0.85, 0.82, 1], openings=doors_n)
    b.wall([-2, -w], [L - 2, -w], h=h, rgba=[0.85, 0.85, 0.82, 1], openings=doors_s)
    b.wall([-2, -w], [-2, w], h=h)
    b.wall([L - 2, -w], [L - 2, w], h=h)
    # door recesses (closed doors 0.3 m behind the wall line)
    for (s0, s1, _, _) in doors_n:
        b.box([s0 - 2 + 0.45, w + 0.35, 1.05], [0.45, 0.03, 1.05], WOOD)
    for (s0, s1, _, _) in doors_s:
        b.box([s0 - 2 + 0.45, -w - 0.35, 1.05], [0.45, 0.03, 1.05], WOOD)
    return np.array([[0.0, 0.0], [44.0, 0.0], [0.0, 0.0]])


def _roof(b: _Builder):
    b.floor()
    # open roof slab: parapet far away, a few HVAC curbs, nothing else
    b.guardrail([-15, -15], [45, -15], post_every=3.0)
    b.guardrail([-15, 15], [45, 15], post_every=3.0)
    b.guardrail([-15, -15], [-15, 15], post_every=3.0)
    b.guardrail([45, -15], [45, 15], post_every=3.0)
    b.misc_box(12.0, 9.0, [2.0, 1.5, 1.0], STEEL)
    b.misc_box(30.0, -10.0, [1.2, 1.2, 0.8], STEEL)
    return np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 4.0], [0.0, 4.0], [0.0, 0.0]])


def _add_boards(b: _Builder, sc: Scene, boards, half_y=0.7):
    for i, (cx, cy, ph) in enumerate(boards):
        nm = f"board{i}"
        b.slider_board(nm, cx, cy, half=(1.3, half_y, 0.006))
        sc.sliders.append(Slider(nm, amplitude=0.5, period=9.0, phase=ph))
        sc.slip_patches.append((cx, cy, 1.3, half_y, -1))


SLAB_LOOP = np.array([[1.5, 0.0], [27.0, 0.0], [27.0, 6.0], [3.0, 6.0], [3.0, 0.0], [1.5, 0.0]])


def build(spec: mujoco.MjSpec, name: str, seed: int = 0) -> Scene:
    rng = np.random.default_rng(seed)
    b = _Builder(spec, rng)

    if name in ("slab_day1", "slab_day7"):
        level = 1 if name == "slab_day1" else 2
        _slab_site(b, change=level)
        return Scene(name, f"Same building {'1 day' if level == 1 else '1 week'} later: materials moved"
                           + (", partitions, shoring, extended slab, sheeting." if level == 2 else "."),
                     SLAB_LOOP, stops={2: 3.0})

    if name in ("outdoor", "outdoor_static", "outdoor_weave"):
        weave = name == "outdoor_weave"
        T, acts = _outdoor_site(b, actors=(name != "outdoor_static"),
                                route=(OUTDOOR_WEAVE if weave else None),
                                extra_rocks=weave, climb_ramp=weave)
        if weave:
            wps, stops = OUTDOOR_WEAVE, {6: 2.0}
            desc = ("Off-road energy site: weave past barriers/rocks, climb a ~12% berm "
                    "(nav limit 15% grade), GT elevation+OGM costmap.")
        else:
            wps, stops = OUTDOOR_ROUTE, {3: 3.0}
            desc = ("Open-air energy/infrastructure site: graded haul-road loop (~100 m) over rough earth, "
                    "stockpiles, excavation, steel frame, pipe yard, containers, light towers, fencing"
                    + (", dump truck, loader and 6 workers." if name == "outdoor" else " (no moving objects)."))
        sc = Scene(name, desc, wps, actors=acts, stops=stops)
        sc.terrain = T
        return sc

    if name == "slab":
        _slab_site(b)
        return Scene(name, "Ground floor of a concrete-frame building under construction; "
                           "column grid, partial upper slab, scaffolding, material stacks. One ~63 m loop.",
                     SLAB_LOOP, stops={2: 3.0})

    if name == "dynamic":
        _slab_site(b)
        actors = []
        # telehandler shuttling in the y=-6 lane and a second in the y=+9.5 lane
        for i, (lane, sp, ph) in enumerate([(-6.0, 2.2, 0.0), (11.0, 1.6, 10.0)]):
            nm = f"telehandler{i}"
            b.actor_loader(nm)
            actors.append(Actor(nm, np.array([[-2.0, lane], [32.0, lane]]), sp, 0.0, phase=ph))
        # workers walking parallel to the robot aisles and across them
        worker_paths = [
            ([[-2, -1.8], [30, -1.8]], 1.3, 0),
            ([[30, 1.9], [-2, 1.9]], 1.1, 7),
            ([[0, 4.3], [28, 4.3]], 1.4, 3),
            ([[28, 7.8], [0, 7.8]], 1.2, 11),
            ([[9, -8], [9, -2.2]], 1.0, 0),
            ([[15, 8.5], [15, 11.0]], 0.9, 1),
            ([[21, -10], [21, -2.4]], 1.2, 2),
            ([[25, 8.2], [25, 11]], 1.0, 4),
        ]
        for i, (p, sp, ph) in enumerate(worker_paths):
            nm = f"worker{i}"
            b.actor_worker(nm)
            actors.append(Actor(nm, np.array(p, float), sp, 0.0, phase=ph))
        return Scene(name, "Same slab with 8 walking workers and 2 telehandlers (dynamic objects).",
                     SLAB_LOOP, actors=actors, stops={2: 3.0})

    if name == "nav_clutter":
        # Navigation benchmark: the surveyed slab with obstacles ON the route, so a
        # waypoint follower must plan. Layout jittered by the seed.
        _slab_site(b)
        j = lambda a: float(rng.uniform(-a, a))
        # 1) U-trap (formwork panels, 1.8 m tall) opening toward the robot in the y=0 aisle
        ux, uy = 11.0 + j(0.6), j(0.3)
        b.box([ux + 1.2, uy, 0.9], [0.05, 1.3, 0.9], WOOD)               # back
        b.box([ux, uy + 1.3, 0.9], [1.2, 0.05, 0.9], WOOD)               # sides
        b.box([ux, uy - 1.3, 0.9], [1.2, 0.05, 0.9], WOOD)
        # 2) barrier line across the aisle with one 1.0 m gap (Jersey-style blocks)
        gx, gy = 19.0 + j(0.5), float(rng.choice([-1.6, 1.6])) + j(0.3)
        for y0, y1 in [(-2.75, gy - 0.5), (gy + 0.5, 2.75)]:
            if y1 - y0 > 0.05:
                b.box([gx, (y0 + y1) / 2, 0.45], [0.3, (y1 - y0) / 2, 0.45], SAFETY)
        # 3) pallet stacks and rebar on the y=6 return aisle
        for k in range(4):
            x = 6.0 + 5.0 * k + j(1.0)
            y = 6.0 + float(rng.choice([-1.0, 0.0, 1.0])) + j(0.3)
            b.box([x, y, 0.55], [0.6, 0.5, 0.55], WOOD, yaw=j(0.4))
        b.box([22.5 + j(0.5), 6.4, 0.25], [1.5, 0.35, 0.25], STEEL)
        # 4) people: two crossing the y=0 aisle, one walking head-on in it, one in the y=6 aisle
        actors = []
        worker_paths = [
            ([[6.0, -2.6], [6.0, 2.6]], 1.0, j(2) + 2),
            ([[23.5, 2.6], [23.5, -2.6]], 1.1, j(2) + 2),
            ([[18.2, -0.6], [13.2, -0.6]], 0.8, j(2)),
            ([[8.0, 6.5], [23.5, 6.5]], 1.0, 8 + j(2)),
        ]
        for i, (pth, sp, ph) in enumerate(worker_paths):
            nm = f"worker{i}"
            b.actor_worker(nm)
            actors.append(Actor(nm, np.array(pth, float), sp, 0.0, phase=ph))
        return Scene(name, "Navigation benchmark: slab with a U-trap, a 1 m gap, pallets on the route "
                           "and workers crossing/head-on.", SLAB_LOOP, actors=actors, stops={2: 3.0})

    if name == "slippery":
        _slab_site(b)
        patches = [(9.0, 0.0, 1.5, 1.2, 0.12), (19.0, 0.0, 1.5, 1.2, 0.12),
                   (15.0, 6.0, 1.5, 1.2, 0.12)]
        for (cx, cy, hx, hy, mu) in patches:
            # wet plastic sheeting / form-release oil: priority makes its friction win
            b.box([cx, cy, 0.001], [hx, hy, 0.001], [0.3, 0.5, 0.9, 1], friction=mu, priority=1)
        return Scene(name, "Slab with low-friction patches (wet plastic sheeting, mu=0.12) on the route.",
                     SLAB_LOOP, slip_patches=patches, stops={2: 3.0})

    if name == "sliding":
        # Loose plywood / formwork sheets lying on the slab that slide under the
        # robot's feet. The stance foot does not slip on the sheet, but the sheet
        # moves in the world, which breaks the leg-odometry "static foot"
        # assumption exactly like a slip. (Low-friction patches were tried first:
        # the pretrained G1 policy falls for mu <= 0.7, see report.)
        _slab_site(b)
        boards = [(9.0, 0.0, 0.0), (19.0, 0.0, 2.0), (15.0, 6.0, 4.0)]
        sliders = []
        for i, (cx, cy, ph) in enumerate(boards):
            nm = f"board{i}"
            b.slider_board(nm, cx, cy)
            sliders.append(Slider(nm, amplitude=0.5, period=9.0, phase=ph))
        return Scene(name, "Slab with three loose plywood sheets on the route that slide back and forth "
                           "(+-0.5 m, peak 0.35 m/s) under the feet: leg-odometry slip without a fall.",
                     SLAB_LOOP, sliders=sliders, slip_patches=[(c[0], c[1], 1.3, 0.7, -1) for c in boards],
                     stops={2: 3.0})

    if name in ("corridor", "corridor_boards"):
        wps = _corridor(b)
        sc = Scene(name, "52 m x 2.5 m service corridor, smooth walls, doors every 12 m: "
                         "geometric degeneracy along the corridor axis (tunnel/mine analogue).",
                   wps, stops={1: 2.0})
        if name == "corridor_boards":
            _add_boards(b, sc, [(10.0, 0.0, 0.0), (22.0, 0.0, 2.0), (34.0, 0.0, 4.0)], half_y=0.6)
            sc.description += " Plus three loose sliding sheets on the floor."
        return sc

    if name in ("roof", "roof_boards"):
        wps = _roof(b)
        sc = Scene(name, "Open roof slab, guardrails 15 m away and two HVAC units: few features.", wps)
        if name == "roof_boards":
            _add_boards(b, sc, [(8.0, 0.0, 0.0), (20.0, 0.0, 2.0), (15.0, 4.0, 4.0), (5.0, 4.0, 1.0)])
            sc.description += " Plus four loose sliding sheets on the route."
        return sc

    raise ValueError(f"unknown scene {name}")


SCENES = ["slab", "corridor", "dynamic", "slippery", "sliding", "roof", "corridor_boards", "roof_boards",
          "slab_day1", "slab_day7", "outdoor", "outdoor_static", "outdoor_weave", "nav_clutter"]
