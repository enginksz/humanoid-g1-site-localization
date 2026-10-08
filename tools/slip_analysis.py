"""How well does the Doppler gate detect leg-odometry slip?

Label per fused leg update (one per LiDAR scan): the true error of the
averaged leg velocity in that window, |v_leg - v_gt|, computed from the
same legvel.bin samples the estimator averaged. "Slip" = error > 0.10 m/s.
Detection = the gate rejected the update.

  python3 tools/slip_analysis.py [results_dir] [data_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.record import quat_wxyz_to_R  # noqa: E402

SLIP_THRESH = 0.10
WINDOW = 0.05


def read_leg_log(f):
    rows = []
    for line in f.read_text().splitlines()[1:]:
        r = line.split(",")
        if len(r) == 8 and float(r[1]) > 0:
            rows.append([float(x) for x in r])
    return np.array(rows)


def analyse(seq: Path, run: Path):
    log = read_leg_log(run / "leg_updates.csv")
    lv = np.fromfile(seq / "legvel.bin").reshape(-1, 6)
    gt = np.load(seq / "gt.npz")
    vb = np.einsum("nji,nj->ni", np.array([quat_wxyz_to_R(q) for q in gt["q"]]), gt["v"])
    err_s = np.linalg.norm(lv[:, 1:4] - vb, axis=1)
    ok = lv[:, 5] > 0
    true_err = np.full(len(log), np.nan)
    for k, t in enumerate(log[:, 0]):
        sel = ok & (lv[:, 0] >= t - WINDOW) & (lv[:, 0] <= t + 1e-6)
        if sel.any():
            # same 1/sigma^2 weighting as the estimator (sigma is constant -> mean)
            true_err[k] = np.linalg.norm((lv[sel, 1:4] - vb[sel]).mean(0))
    slip = true_err > SLIP_THRESH
    rej = log[:, 6] == 0
    tp, fp = int((slip & rej).sum()), int((~slip & rej).sum())
    fn, tn = int((slip & ~rej).sum()), int((~slip & ~rej).sum())
    return dict(n=len(log), n_slip=int(slip.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                fpr=fp / max(fp + tn, 1),
                accepted_err_rms=float(np.sqrt(np.nanmean(true_err[~rej] ** 2))),
                all_err_rms=float(np.sqrt(np.nanmean(true_err ** 2))),
                sample_err_median=float(np.median(err_s[ok])))


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data"
    out = {}
    for scene in ["slab", "sliding", "corridor_boards", "roof_boards"]:
        run = res / scene / "fmcw4d_leg"
        if (run / "leg_updates.csv").exists():
            out[scene] = r = analyse(data / scene, run)
            print(f"{scene:16s} slip windows {r['n_slip']:4d}/{r['n']}  precision {r['precision']:.2f}  "
                  f"recall {r['recall']:.2f}  FPR {r['fpr']:.3f}  leg err RMS all {r['all_err_rms']:.3f} "
                  f"-> accepted {r['accepted_err_rms']:.3f} m/s")
    (res / "slip_gate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
