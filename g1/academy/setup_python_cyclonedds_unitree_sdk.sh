#!/usr/bin/env bash
set -euo pipefail

# Example only: review versions and paths before running.
DEPS="$HOME/g1-deps"
CYCLONE_SRC="$DEPS/cyclonedds"
CYCLONE_PREFIX="$DEPS/cyclonedds-0.10"
SDK_SRC="$DEPS/unitree_sdk2_python"
VENV="$DEPS/venv"

mkdir -p "$DEPS"
git clone --branch releases/0.10.x --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONE_SRC"
cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CYCLONE_PREFIX" -DBUILD_EXAMPLES=OFF
cmake --build "$CYCLONE_SRC/build" --parallel
cmake --install "$CYCLONE_SRC/build"

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK_SRC"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -e "$SDK_SRC"
"$VENV/bin/python" -m pip install openai dash dash-bootstrap-components pandas rich pyzmq opencv-python pyrealsense2

echo "Source $VENV/bin/activate and export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "Also export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:$LD_LIBRARY_PATH"
echo "Also export PYTHONPATH=$SDK_SRC:$PYTHONPATH"
