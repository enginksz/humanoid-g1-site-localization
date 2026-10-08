"""Readers for the simulator sequence format (see sim/record.py)."""
import json
import struct
from pathlib import Path

import numpy as np

PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("doppler", "<f4"), ("t_off", "<f4")])


def read_imu(seq):
    return np.fromfile(Path(seq) / "imu.bin", dtype=np.float64).reshape(-1, 7)


def iter_scans(seq):
    with open(Path(seq) / "lidar.bin", "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            t0, n = struct.unpack("<dI", h)
            yield t0, np.frombuffer(f.read(n * PT.itemsize), dtype=PT)


def read_meta(seq):
    return json.loads((Path(seq) / "meta.json").read_text())


def read_gt(seq):
    return dict(np.load(Path(seq) / "gt.npz"))


def read_tum(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1:4], a[:, [7, 4, 5, 6]]   # t, p, q(wxyz)
