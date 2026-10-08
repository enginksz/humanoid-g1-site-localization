"""FMCW-LIO as a live process for closed-loop simulation (stream mode, see main_offline.cpp).

    est = OnlineEstimator(meta, doppler=True, leg=True)
    est.imu(t, gyro, acc); est.leg(t, v_b, sigma, n); est.scan(t_begin, pts)
    for pose in est.poll(): ...   # dicts with t, T (4x4, LIO world), sigma_pos, sigma_yaw_deg

Self-check:  python3 tools/online_est.py data/slab results/slab/fmcw4d_leg/poses_tum.txt
"""
from __future__ import annotations

import json
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BIN = ROOT / "fmcw_lio_standalone/build/fmcw_lio_offline"
CFG = ROOT / "configs/fmcw_g1_aeva.yaml"


def quat_xyzw_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class OnlineEstimator:
    def __init__(self, meta, out_dir, doppler=True, leg=True, extra=()):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        args = [str(BIN), str(CFG), "-", str(out_dir),
                f"common/use_doppler={'true' if doppler else 'false'}",
                f"common/R_bl={json.dumps(meta['R_bl'])}", f"common/p_bl_b={json.dumps(meta['p_bl_b'])}",
                f"leg/use={'true' if leg else 'false'}", *extra]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(Path(out_dir) / "stderr.txt", "w"), bufsize=0)
        self.q: queue.Queue = queue.Queue()
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        for line in self.p.stdout:
            f = line.split()
            if not f or f[0] != b"P":
                continue
            v = [float(x) for x in f[1:]]
            T = np.eye(4)
            T[:3, :3] = quat_xyzw_to_R(*v[4:8])
            T[:3, 3] = v[1:4]
            self.q.put(dict(t=v[0], T=T, sigma_pos=v[8], sigma_yaw_deg=v[9], doppler_ok=bool(v[10])))

    def imu(self, t, gyro, acc):
        self.p.stdin.write(b"I" + struct.pack("<7d", t, *gyro, *acc))

    def leg(self, t, v_b, sigma, n):
        self.p.stdin.write(b"G" + struct.pack("<6d", t, *v_b, sigma, n))

    def scan(self, t_begin, pts_f32_5):
        self.p.stdin.write(b"L" + struct.pack("<dI", t_begin, len(pts_f32_5)) + pts_f32_5.astype(np.float32).tobytes())
        self.p.stdin.flush()

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        try:
            self.p.stdin.write(b"E")
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.wait(timeout=30)
        self.t.join(timeout=5)
        return self.poll()


def _selfcheck(seq, ref):
    from tools.seqio import iter_scans, read_imu, read_meta
    seq = Path(seq)
    meta = read_meta(seq)
    imu = read_imu(seq)
    lv = np.fromfile(seq / "legvel.bin").reshape(-1, 6)
    est = OnlineEstimator(meta, ROOT / "results/_online_check")
    i = 0
    got = []
    for t0, s in iter_scans(seq):
        t_end = t0 + float(s["t_off"][-1])
        while i < len(imu) and imu[i, 0] <= t_end + 0.02:
            est.imu(imu[i, 0], imu[i, 1:4], imu[i, 4:7])
            est.leg(*lv[i, :1], lv[i, 1:4], lv[i, 4], lv[i, 5])
            i += 1
        a = np.stack([s["x"], s["y"], s["z"], s["doppler"], s["t_off"]], 1)
        est.scan(t0, a)
        got += est.poll()
    got += est.close()
    r = np.loadtxt(ref)
    tt = np.array([g["t"] for g in got])
    pp = np.array([g["T"][:3, 3] for g in got])
    k = np.searchsorted(tt, r[:, 0])
    ok = (k < len(tt)) & (np.abs(tt[np.minimum(k, len(tt) - 1)] - r[:, 0]) < 1e-4)
    d = np.linalg.norm(pp[k[ok]] - r[ok, 1:4], axis=1)
    print(f"stream poses {len(got)}, file poses {len(r)}, matched {ok.sum()}, max |dp| {d.max():.2e} m")


if __name__ == "__main__":
    _selfcheck(sys.argv[1], sys.argv[2])
