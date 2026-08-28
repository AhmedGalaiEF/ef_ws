#!/usr/bin/env bash
set -euo pipefail

# Example only: review versions and paths before running.
#
# Builds the reference environment that solved/reset_users.sh later clones
# into every teilnehmerN account, so every package/binary a notebook might
# need (SDK, DDS, Piper TTS, openaiapi, Jupyter) has to be reliable here
# first.
DEPS="$HOME/g1-deps"
CYCLONE_SRC="$DEPS/cyclonedds"
CYCLONE_PREFIX="$DEPS/cyclonedds-0.10"
SDK_SRC="$DEPS/unitree_sdk2_python"
VENV="$DEPS/venv"
VOICE_DIR="$DEPS/piper-voices"
# Matches util.py's PIPER_VOICES map — keep these in sync.
PIPER_VOICES=(en_US-lessac-medium de_DE-thorsten-medium fr_FR-siwis-medium es_ES-davefx-medium)

# espeak-ng ships the phoneme data some piper-phonemize wheels expect at
# runtime. Best-effort: don't fail the whole setup if apt isn't usable here.
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y espeak-ng || true
fi

mkdir -p "$DEPS" "$VOICE_DIR"
git clone --branch releases/0.10.x --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONE_SRC"
cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CYCLONE_PREFIX" -DBUILD_EXAMPLES=OFF
cmake --build "$CYCLONE_SRC/build" --parallel
cmake --install "$CYCLONE_SRC/build"

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK_SRC"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -e "$SDK_SRC"
# openai/pydantic: the openaiapi agent (g1/modules/scripts/openaiapi) talks
# to the REST API directly and only needs pydantic; keep openai too since
# other scripts import the SDK. ipython/jupyterlab/ipykernel: notebook
# tooling used throughout the academy. numpy: explicit rather than relying
# on it as a transitive opencv dep. piper-tts: Piper text-to-speech CLI
# used by util.py/sdk_wrapper.py's play_piper_text().
"$VENV/bin/python" -m pip install \
  openai pydantic dash dash-bootstrap-components pandas numpy rich pyzmq \
  opencv-python pyrealsense2 ipython jupyterlab notebook ipykernel piper-tts

# Register a Jupyter kernel backed by this venv so notebooks reliably pick
# up the right interpreter (SDK + DDS libs) without manual activation.
"$VENV/bin/python" -m ipykernel install --user --name unitree_sdk2 \
  --display-name "Unitree SDK2 (g1 academy)"

# Fetch the Piper voice models util.py expects at $G1_PIPER_VOICE_DIR.
# rhasspy/piper-voices lays files out as <lang>/<locale>/<name>/<quality>/...
for voice in "${PIPER_VOICES[@]}"; do
  locale="${voice%%-*}"; rest="${voice#*-}"
  name="${rest%-*}"; quality="${rest##*-}"
  lang="${locale%%_*}"
  dest="$VOICE_DIR/$voice"
  mkdir -p "$dest"
  base="https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$name/$quality"
  for ext in onnx onnx.json; do
    [ -f "$dest/$voice.$ext" ] || curl -fsSL "$base/$voice.$ext" -o "$dest/$voice.$ext" || \
      echo "warning: failed to download $voice.$ext (check network)" >&2
  done
done

echo "Source $VENV/bin/activate and export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "Also export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:$LD_LIBRARY_PATH"
echo "Also export PYTHONPATH=$SDK_SRC:$PYTHONPATH"
echo "Also export G1_PIPER_BIN=$VENV/bin/piper and G1_PIPER_VOICE_DIR=$VOICE_DIR"
