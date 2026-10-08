"""Run every method on every recorded sequence and write a summary table.

  python3 tools/run_all.py [--scenes slab corridor ...] [--methods ...]

Methods
  fastlio2    original hku-mars FAST_LIO (FAST-LIO2), ROS-free build
  fmcw3d      FMCW-LIO with Doppler disabled  (ESIKF point-to-plane LIO, FAST-LIO2 family)
  fmcw4d      FMCW-LIO with Doppler velocity update + Doppler dynamic-point removal
  legekf      Leg-IMU ESKF (proprioceptive only)
  fmcw3d_leg     3D LIO + leg velocity update, no gate
  fmcw3d_leg_pg  3D LIO + leg velocity update gated by the filter prediction (ablation)
  fmcw4d_leg     4D LIO + leg velocity update gated by Doppler ego-velocity (slip check)
  fmcw4d_leg_ng  4D LIO + leg velocity update, no gate (ablation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate import evaluate  # noqa: E402
from tools.legodo import leg_velocity, run_leg_ekf  # noqa: E402
from tools.run_fastlio import run as run_fastlio  # noqa: E402
from tools.run_fmcw import run as run_fmcw  # noqa: E402

METHODS = {
    "fastlio2": "fastlio",
    "fmcw3d": dict(doppler=False, extra=()),
    "fmcw4d": dict(doppler=True, extra=()),
    "fmcw3d_leg": dict(doppler=False, extra=("leg/use=true", "leg/chi2_gate=1e9")),
    "fmcw3d_leg_pg": dict(doppler=False, extra=("leg/use=true",)),
    # geometry-only + legs with realistic point noise (the fix for the open-field slide)
    "fmcw3d_leg_cov": dict(doppler=False, extra=("leg/use=true", "leg/chi2_gate=1e9", "update/point_cov=0.05")),
    "fmcw4d_leg": dict(doppler=True, extra=("leg/use=true",)),
    "fmcw4d_leg_ng": dict(doppler=True, extra=("leg/use=true", "leg/chi2_gate=1e9")),
    "legekf": None,
}


def run_scene(seq: Path, res: Path, methods, tag=""):
    out = {}
    if not (seq / "legvel.bin").exists():
        leg_velocity(seq)
    for m in methods:
        d = res / m
        try:
            if METHODS[m] is None:
                run_leg_ekf(seq, d)
            elif METHODS[m] == "fastlio":
                code, msg = run_fastlio(seq, d)
                if code != 0:
                    out[m] = dict(failed=True, reason=f"exit {code}: {msg[-200:]}")
                    continue
            else:
                code, msg = run_fmcw(seq, d, METHODS[m]["doppler"], extra=METHODS[m]["extra"])
                if code != 0:
                    out[m] = dict(failed=True, reason=f"exit {code}: {msg[-200:]}")
                    continue
            r = evaluate(d / "poses_tum.txt", seq / "gt_tum.txt")
        except Exception as e:  # keep the batch going; record the failure
            r = dict(failed=True, reason=repr(e)[:300])
        out[m] = r
        if r.get("failed"):
            print(f"  {seq.name:12s}{tag} {m:11s} FAILED {r.get('reason')}")
        else:
            print(f"  {seq.name:12s}{tag} {m:11s} ATE {r['ate_rmse']:.3f} m  end {r['final_err']:.3f} m "
                  f"({r['final_err_pct']:.2f}%)  z {r['z_rmse']:.3f}  yaw {r['yaw_err_final_deg']:+.2f} deg", flush=True)
        (d / "metrics.json").write_text(json.dumps(r, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    data, results = Path(a.data), Path(a.results)
    scenes = a.scenes or sorted(p.name for p in data.iterdir() if (p / "meta.json").exists() and not p.name.startswith("_"))
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for s in scenes:
        summary.setdefault(s, {}).update(run_scene(data / s, results / s, a.methods))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
