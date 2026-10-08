"""Run the ROS-free FAST-LIO2 build on a simulator sequence.

  python3 tools/run_fastlio.py data/corridor results/corridor/fastlio2
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "fast_lio_standalone/build/fast_lio_offline"


def run(seq, out, config=ROOT / "configs/fastlio_g1_aeva.yaml", extra=()):
    meta = json.loads((Path(seq) / "meta.json").read_text())
    seq, out = Path(seq).resolve(), Path(out).resolve()
    args = [str(BIN), str(config), str(seq), str(out),
            f"mapping/extrinsic_R={json.dumps(meta['R_bl'])}",
            f"mapping/extrinsic_T={json.dumps(meta['p_bl_b'])}",
            f"mapping/fov_degree={min(meta['lidar']['fov_az_deg'], 360.0)}", *extra]
    Path(out).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(args, capture_output=True, text=True)
    (Path(out) / "log.txt").write_text(r.stdout[-20000:] + "\n--- stderr\n" + r.stderr[-20000:])
    last = [l for l in r.stdout.splitlines() if l.startswith("scans=")]
    return r.returncode, (last[-1] if last else r.stdout[-500:] + r.stderr[-500:])


if __name__ == "__main__":
    code, msg = run(sys.argv[1], sys.argv[2], extra=sys.argv[3:])
    print(code, msg)
    sys.exit(code)
