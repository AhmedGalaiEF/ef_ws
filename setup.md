# Unitree SDK2 Python and CycloneDDS Setup

This setup targets Ubuntu 24.04, Python 3.10 managed by `uv`, and the
`unitree_sdk2_python` package.

## Version compatibility

`unitree_sdk2_python` currently requires:

- Python 3.8 or newer
- Python package `cyclonedds==0.10.2`
- A native CycloneDDS installation from the compatible `releases/0.10.x`
  series

Use the native CycloneDDS `0.10.5` tag for a reproducible installation. Do not
build the current CycloneDDS `master` branch for this SDK.

## 1. Install system build dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  curl \
  git \
  libssl-dev \
  ninja-build \
  pkg-config
```

## 2. Install `uv` and Python 3.10

Skip the first command when `uv` is already installed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.10
uv python find 3.10
```

## 3. Create the workspace and clone both repositories

```bash
export UNITREE_WS="$HOME/unitree_sdk2_ws"
mkdir -p "$UNITREE_WS"
cd "$UNITREE_WS"

git clone --branch 0.10.5 --depth 1 \
  https://github.com/eclipse-cyclonedds/cyclonedds.git

git clone \
  https://github.com/unitreerobotics/unitree_sdk2_python.git
```

## 4. Build and install native CycloneDDS

Install into the repository-local `install` directory. This path must be the
same path later assigned to `CYCLONEDDS_HOME`.

```bash
export UNITREE_WS="$HOME/unitree_sdk2_ws"
export CYCLONEDDS_HOME="$UNITREE_WS/cyclonedds/install"

cmake \
  -S "$UNITREE_WS/cyclonedds" \
  -B "$UNITREE_WS/cyclonedds/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
  -DBUILD_TESTING=OFF

cmake --build "$UNITREE_WS/cyclonedds/build" --target install --parallel
```

Verify the native installation:

```bash
test -f "$CYCLONEDDS_HOME/lib/libddsc.so"
test -f "$CYCLONEDDS_HOME/lib/cmake/CycloneDDS/CycloneDDSConfig.cmake"
"$CYCLONEDDS_HOME/bin/idlc" -h >/dev/null
```

## 5. Set the environment correctly

If ROS 2 is needed, source it first. Its existing prefixes must be preserved.

```bash
source /opt/ros/kilted/setup.bash

export UNITREE_WS="$HOME/unitree_sdk2_ws"
export CYCLONEDDS_HOME="$UNITREE_WS/cyclonedds/install"
export CycloneDDS_ROOT="$CYCLONEDDS_HOME"

export PATH="$CYCLONEDDS_HOME/bin${PATH:+:$PATH}"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$CYCLONEDDS_HOME/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
```

Important details:

- `CYCLONEDDS_HOME` is the main variable used while building and running the
  Python `cyclonedds==0.10.2` package.
- `CMAKE_PREFIX_PATH` lets CMake find
  `lib/cmake/CycloneDDS/CycloneDDSConfig.cmake`.
- Always prepend with `${VARIABLE:+:$VARIABLE}`. Do not discard existing ROS
  prefixes.
- Do not add this CycloneDDS prefix to `AMENT_PREFIX_PATH`. CycloneDDS is a
  plain CMake installation, not an ament install prefix. After sourcing ROS
  Kilted, `AMENT_PREFIX_PATH` should continue to contain `/opt/ros/kilted` and
  any real colcon overlays. Source an overlay's `install/setup.bash` instead of
  editing `AMENT_PREFIX_PATH` manually.

Check the resulting values:

```bash
printf 'CYCLONEDDS_HOME=%s\n' "$CYCLONEDDS_HOME"
printf 'CMAKE_PREFIX_PATH=%s\n' "$CMAKE_PREFIX_PATH"
printf 'AMENT_PREFIX_PATH=%s\n' "$AMENT_PREFIX_PATH"
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"

cmake --find-package \
  -DNAME=CycloneDDS \
  -DCOMPILER_ID=GNU \
  -DLANGUAGE=C \
  -DMODE=EXIST
```

## 6. Create and activate the Python 3.10 environment

```bash
cd "$UNITREE_WS"
uv venv --python 3.10 .venv
source "$UNITREE_WS/.venv/bin/activate"

python --version
```

The output must report Python 3.10.x.

## 7. Install Unitree SDK2 Python

Keep the environment exports from step 5 active. Forcing a source build of
`cyclonedds==0.10.2` makes the Python extension link against the selected local
CycloneDDS installation.

```bash
cd "$UNITREE_WS"

uv pip install \
  --no-binary cyclonedds \
  -e "$UNITREE_WS/unitree_sdk2_python"
```

Verify the environment:

```bash
python - <<'PY'
from importlib.metadata import version
import cyclonedds
import unitree_sdk2py

print("Python cyclonedds:", version("cyclonedds"))
print("Unitree SDK:", version("unitree-sdk2py"))
print("cyclonedds module:", cyclonedds.__file__)
print("unitree_sdk2py module:", unitree_sdk2py.__file__)
PY

ldd "$UNITREE_WS/.venv/lib/python3.10/site-packages/cyclonedds/_clayer"*.so \
  | grep ddsc
```

`libddsc.so` should resolve to:

```text
~/unitree_sdk2_ws/cyclonedds/install/lib/libddsc.so
```

## 8. Start a new development shell later

Run these commands in every new terminal:

```bash
source /opt/ros/kilted/setup.bash

export UNITREE_WS="$HOME/unitree_sdk2_ws"
export CYCLONEDDS_HOME="$UNITREE_WS/cyclonedds/install"
export CycloneDDS_ROOT="$CYCLONEDDS_HOME"
export PATH="$CYCLONEDDS_HOME/bin${PATH:+:$PATH}"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$CYCLONEDDS_HOME/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

source "$UNITREE_WS/.venv/bin/activate"
cd "$UNITREE_WS/unitree_sdk2_python"
```

Only source `/opt/ros/kilted/setup.bash` when the current project also needs
ROS 2. Unitree SDK2 Python itself does not require ROS.

## 9. Network interface and smoke test

Find the wired interface connected to the robot:

```bash
ip -brief address
```

Run the local DDS publisher in one terminal:

```bash
cd "$UNITREE_WS/unitree_sdk2_python"
python example/helloworld/publisher.py
```

Run the subscriber in a second terminal with the same environment:

```bash
cd "$UNITREE_WS/unitree_sdk2_python"
python example/helloworld/subscriber.py
```

Robot examples take the interface name as an argument, for example:

```bash
python example/high_level/read_highstate.py enp2s0
```

Replace `enp2s0` with the interface reported by `ip -brief address`.

## Existing machine note

The old `.bashrc` value

```text
/home/ag/academy/academy_content/docs/repos/cyclonedds_0_10/install_0_10
```

does not exist on this machine. Remove or replace that stale block before using
this setup. A currently built alternative exists at:

```text
/home/ag/Desktop/unitree/cyclonedds/install
```

If reusing that installation instead of creating `~/unitree_sdk2_ws`, set
`CYCLONEDDS_HOME` to the latter path and use the same prefix exports above.

## References

- Unitree SDK2 Python: <https://github.com/unitreerobotics/unitree_sdk2_python>
- CycloneDDS 0.10 branch:
  <https://github.com/eclipse-cyclonedds/cyclonedds/tree/releases/0.10.x>
- `uv` installation: <https://docs.astral.sh/uv/getting-started/installation/>
