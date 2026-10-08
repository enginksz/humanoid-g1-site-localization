"""Run the ROS-free FMCW-LIO build on a simulator sequence.

  python3 tools/run_fmcw.py data/slab results/slab/fmcw4d --doppler 1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "fmcw_lio_standalone/build/fmcw_lio_offline"


def fov_overrides(meta):
    """Velocimeter angle gates follow the sensor FoV (defaults are Aeva's 120x30)."""
    L = meta["lidar"]
    if L["kind"] == "aeva":
        return []
    return [f"velocimeter/azimuth_thresh_deg={L['fov_az_deg'] / 2 + 1}",
            f"velocimeter/elevation_thresh_deg={L['fov_el_deg'] / 2 + 1}"]


def run(seq, out, doppler=True, config=ROOT / "configs/fmcw_g1_aeva.yaml", extra=()):
    meta = json.loads((Path(seq) / "meta.json").read_text())
    args = [str(BIN), str(config), str(seq), str(out),
            f"common/use_doppler={'true' if doppler else 'false'}",
            f"common/R_bl={json.dumps(meta['R_bl'])}",
            f"common/p_bl_b={json.dumps(meta['p_bl_b'])}",
            f"map_io/dump=true", *fov_overrides(meta), *extra]
    Path(out).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(args, capture_output=True, text=True)
    (Path(out) / "log.txt").write_text(r.stdout[-20000:] + "\n--- stderr\n" + r.stderr[-20000:])
    last = [l for l in r.stdout.splitlines() if l.startswith("scans=")]
    return r.returncode, (last[-1] if last else r.stdout[-500:] + r.stderr[-500:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("out")
    ap.add_argument("--doppler", type=int, default=1)
    a, extra = ap.parse_known_args()   # remaining "section/key=value" overrides
    code, msg = run(a.seq, a.out, bool(a.doppler), extra=extra)
    print(code, msg)
    sys.exit(code)
