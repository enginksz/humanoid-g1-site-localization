#!/usr/bin/env bash
# Clone the upstream projects at the commits used in this work and apply our patches.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party && cd third_party
[ -d FMCW-LIO ] || git clone https://github.com/IMRL/FMCW-LIO.git && git -C FMCW-LIO checkout 5c21d1bef29a63946c2ef556dc0882fb7868c57e
[ -d FAST_LIO ] || git clone --recursive https://github.com/hku-mars/FAST_LIO.git && git -C FAST_LIO checkout 7cc4175de6f8ba2edf34bab02a42195b141027e9
[ -d unitree_rl_gym ] || git clone https://github.com/unitreerobotics/unitree_rl_gym.git && git -C unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b
cd ../fmcw_lio_standalone
mkdir -p include/bithub include/system
cp ../third_party/FMCW-LIO/src/system/fmcw_lio.cpp src/fmcw_lio_leg.cpp
patch -p1 src/fmcw_lio_leg.cpp < patches/fmcw_lio_leg.patch
cp ../third_party/FMCW-LIO/include/bithub/bithub.hpp include/bithub/bithub.hpp
patch -p1 include/bithub/bithub.hpp < patches/bithub.hpp.patch
cp ../third_party/FMCW-LIO/include/system/fmcw_lio.hpp include/system/fmcw_lio.hpp
patch -p1 include/system/fmcw_lio.hpp < patches/fmcw_lio.hpp.patch
echo "third_party ready"
