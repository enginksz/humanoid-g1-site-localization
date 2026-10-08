"""Leg odometry for the G1: stance-foot kinematic velocity + a Leg-IMU ESKF.

Kinematic velocity (per IMU sample, body = IMU frame):
    a stance foot's sole point is assumed static in the world, so
        v_b = -( w_b x r(q) + J(q) dq )
    with r the sole point in the body frame and J its joint Jacobian.
Contact comes from the foot contact force (stands in for a torque-based
contact estimator; the G1 has no foot force sensors). Slip violates the
static-foot assumption, which is exactly what the Doppler gate in the fused
estimator is meant to catch.

Outputs:
    <seq>/legvel.bin : float64 records (t, vx, vy, vz, sigma, n_contact)
    run_leg_ekf()    : TUM trajectory from IMU + leg velocity only
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.record import G1_XML, IMU_POS  # noqa: E402
from tools.seqio import read_imu, read_meta  # noqa: E402

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
SOLE_OFFSET = np.array([0.035, 0.0, -0.03])   # centre of the 4 contact spheres
CONTACT_N = 40.0
# Stance-phase window. Measured on the sim (tools/legodo analysis): the first
# ~30 ms after touchdown (impact) and toe-off (>0.3 s) are off by 0.1-0.4 m/s;
# mid-stance is good to ~1.5 cm/s. Only mid-stance samples are used.
STANCE_MIN = 0.04         # s after touchdown
STANCE_MAX = 0.28         # s after touchdown
STANCE_FORCE = 150.0      # N
SIGMA_V = 0.03            # m/s


class LegKinematics:
    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(G1_XML))
        s = spec.body("pelvis").add_site()
        s.name = "imu"
        s.pos = list(IMU_POS)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES]
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.jacp = np.zeros((3, self.m.nv))

    def foot(self, q, k):
        """Sole point of foot k in the body (IMU) frame and its 3x12 joint Jacobian."""
        d = self.d
        d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[7:] = q
        mujoco.mj_fwdPosition(self.m, d)
        bid = self.feet[k]
        R = d.xmat[bid].reshape(3, 3)
        p = d.xpos[bid] + R @ SOLE_OFFSET
        mujoco.mj_jac(self.m, d, self.jacp, None, p, bid)
        return p - d.site_xpos[self.sid], self.jacp[:, 6:].copy()


class LegVelocityOnline:
    """Per-sample stance-velocity estimator (same logic offline and in closed loop)."""

    def __init__(self):
        self.kin = LegKinematics()
        self.last_td = np.full(2, -1.0)
        self.prev_c = np.zeros(2, bool)

    def update(self, t, w, q, dq, F):
        """Returns (v_b, sigma, n_feet); sigma = 1e3 and n = 0 when no foot is in mid-stance."""
        c = F > CONTACT_N
        for k in range(2):
            if c[k] and not self.prev_c[k]:
                self.last_td[k] = t
        self.prev_c = c
        vs, ws = [], []
        for k in range(2):
            since = t - self.last_td[k]
            if not (c[k] and F[k] > STANCE_FORCE and STANCE_MIN <= since <= STANCE_MAX):
                continue
            r, J = self.kin.foot(q, k)
            vs.append(-(np.cross(w, r) + J @ dq))
            ws.append(F[k])
        if not vs:
            return np.zeros(3), 1e3, 0
        vs, ws = np.array(vs), np.array(ws)
        return (vs * ws[:, None]).sum(0) / ws.sum(), SIGMA_V, len(vs)


def leg_velocity(seq, write=True):
    seq = Path(seq)
    imu = read_imu(seq)
    legs = dict(np.load(seq / "legs.npz"))
    t, q, dq, F = legs["t"], legs["q"], legs["dq"], legs["foot_force"]
    assert len(t) == len(imu) and np.allclose(t, imu[:, 0])
    lvo = LegVelocityOnline()
    out = np.zeros((len(t), 6))
    out[:, 0] = t
    for i in range(len(t)):
        v, sig, n = lvo.update(t[i], imu[i, 1:4], q[i], dq[i], F[i])
        out[i, 1:4], out[i, 4], out[i, 5] = v, sig, n
    if write:
        out.astype(np.float64).tofile(seq / "legvel.bin")
    return out


# ------------------------------------------------------------------ Leg-IMU ESKF
def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-9:
        return np.eye(3) + _skew(phi)
    k = phi / a
    K = _skew(k)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def run_leg_ekf(seq, out_dir, t_init=(1.0, 2.0)):
    """15-state error-state EKF: IMU propagation, stance-velocity updates."""
    seq, out_dir = Path(seq), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imu = read_imu(seq)
    lv = leg_velocity(seq, write=True) if not (seq / "legvel.bin").exists() else \
        np.fromfile(seq / "legvel.bin", dtype=np.float64).reshape(-1, 6)
    g = np.array([0, 0, -9.81])
    # init window skips the spawn drop; the robot steps in place here, so the
    # mean specific force over whole gait cycles still points along gravity
    a0, n0 = np.searchsorted(imu[:, 0], t_init)
    acc0 = imu[a0:n0, 4:7].mean(0)
    # gravity alignment (yaw = 0)
    z = acc0 / np.linalg.norm(acc0)
    v = np.cross(z, [0, 0, 1.0])
    s, c = np.linalg.norm(v), z[2]
    R = np.eye(3) if s < 1e-9 else np.eye(3) + _skew(v) + _skew(v) @ _skew(v) * (1 - c) / s ** 2
    p, vel = np.zeros(3), np.zeros(3)
    bg, ba = imu[a0:n0, 1:4].mean(0), np.zeros(3)
    P = np.diag([1e-6] * 3 + [1e-4] * 3 + [1e-4] * 3 + [1e-5] * 3 + [1e-2] * 3)  # p v th bg ba
    Qg, Qa, Qbg, Qba = 0.004 ** 2, 0.03 ** 2, 2e-5 ** 2, 2e-4 ** 2
    poses = []
    for i in range(n0, len(imu)):
        dt = imu[i, 0] - imu[i - 1, 0]
        w = imu[i - 1, 1:4] - bg
        a = imu[i - 1, 4:7] - ba
        # propagate nominal
        acc_w = R @ a + g
        p = p + vel * dt + 0.5 * acc_w * dt * dt
        vel = vel + acc_w * dt
        R = R @ _exp(w * dt)
        # propagate error covariance (theta in body frame)
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt
        Fx[3:6, 6:9] = -R @ _skew(a) * dt
        Fx[3:6, 12:15] = -R * dt
        Fx[6:9, 6:9] = _exp(-w * dt)
        Fx[6:9, 9:12] = -np.eye(3) * dt
        Qd = np.zeros((15, 15))
        Qd[3:6, 3:6] = np.eye(3) * Qa * dt * 250 * dt   # per-sample sigma -> discrete
        Qd[6:9, 6:9] = np.eye(3) * Qg * dt * 250 * dt
        Qd[9:12, 9:12] = np.eye(3) * Qbg
        Qd[12:15, 12:15] = np.eye(3) * Qba
        P = Fx @ P @ Fx.T + Qd
        # stance velocity update: z = R^T v
        if lv[i, 5] > 0:
            zb = lv[i, 1:4]                      # already includes w x r
            h = R.T @ vel
            H = np.zeros((3, 15))
            H[:, 3:6] = R.T
            H[:, 6:9] = _skew(h)
            Rm = np.eye(3) * lv[i, 4] ** 2
            S = H @ P @ H.T + Rm
            r = zb - h
            if r @ np.linalg.solve(S, r) < 25.0:          # loose outlier gate
                K = P @ H.T @ np.linalg.inv(S)
                dx = K @ r
                p += dx[0:3]
                vel += dx[3:6]
                R = R @ _exp(dx[6:9])
                bg += dx[9:12]
                ba += dx[12:15]
                P = (np.eye(15) - K @ H) @ P
        if i % 25 == 0:   # 10 Hz output
            qw = np.empty(4)
            mujoco.mju_mat2Quat(qw, R.reshape(-1))
            poses.append([imu[i, 0], *p, qw[1], qw[2], qw[3], qw[0]])
    np.savetxt(out_dir / "poses_tum.txt", np.array(poses), fmt="%.6f")
    return out_dir / "poses_tum.txt"


if __name__ == "__main__":
    seq = sys.argv[1]
    lv = leg_velocity(seq)
    print("contact ratio", (lv[:, 5] > 0).mean())
    if len(sys.argv) > 2:
        print(run_leg_ekf(seq, sys.argv[2]))
