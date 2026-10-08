#!/usr/bin/env bash
# Full outdoor-site batch: record (3 seeds + static map run), all estimators, prior map,
# map localization, closed-loop navigation.
cd "$(dirname "$0")/.." || exit 1
set -x
rm -rf data/outdoor data/outdoor_static results/outdoor results_s1/outdoor results_s2/outdoor results/closed_loop/outdoor_*
python3 -m sim.record --scene outdoor --out data/outdoor
python3 -m sim.record --scene outdoor_static --out data/outdoor_static
python3 tools/run_all.py --scenes outdoor
for seed in 1 2; do
  python3 -m sim.record --scene outdoor --seed $seed --out data_s$seed/outdoor
  python3 tools/run_all.py --data data_s$seed --results results_s$seed --scenes outdoor
  rm -f data_s$seed/outdoor/lidar.bin
done
python3 tools/maploc.py build-map data/outdoor_static results/maploc/map_outdoor.npz
for o in fmcw4d_leg fmcw3d legekf; do
  OMP_NUM_THREADS=4 python3 tools/maploc.py run data/outdoor results/outdoor/$o/poses_tum.txt results/outdoor/maploc_$o --map results/maploc/map_outdoor.npz
done
printf "3d -\n4d -\n4d_leg -\n3d results/maploc/map_outdoor.npz\n" | xargs -P 2 -L 1 sh -c \
  'if [ "$1" = "-" ]; then OMP_NUM_THREADS=2 python3 -m sim.closed_loop --scene outdoor --est $0 --out results/closed_loop/outdoor_$0; else OMP_NUM_THREADS=2 python3 -m sim.closed_loop --scene outdoor --est $0 --map $1 --out results/closed_loop/outdoor_$0_map; fi'
echo BATCH_DONE
