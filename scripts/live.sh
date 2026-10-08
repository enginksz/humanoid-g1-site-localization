#!/usr/bin/env bash
# Live view of a closed-loop run.   ./scripts/live.sh <scene> <estimator> [--map file] [--ros] [--speed 1.0]
#   scenes: slab dynamic sliding corridor roof corridor_boards roof_boards slab_day1 slab_day7
#   estimators: 3d 4d 3d_leg 4d_leg
# With --ros, open RViz2 in another terminal:  source /opt/ros/humble/setup.bash && rviz2 -d configs/live.rviz
cd "$(dirname "$0")/.." || exit 1
scene=${1:-dynamic}; est=${2:-4d_leg}; shift 2 2>/dev/null
if [[ " $* " == *" --ros "* ]]; then source /opt/ros/humble/setup.bash; fi
export MUJOCO_GL=glfw
exec python3 -m sim.live_view --scene "$scene" --est "$est" "$@"
