#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_python_cyclonedds_unitree_sdk.sh [--help]

Build CycloneDDS, Unitree SDK2, the academy virtual environment, and Piper
voice assets. Override the workspace with G1_DEPS_DIR.
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

ensure_clone() {
  local url="$1" destination="$2" branch="${3:-}"
  if [[ -d "$destination/.git" ]]; then
    echo "Reusing existing checkout: $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite non-git path: $destination" >&2
    exit 1
  fi
  if [[ -n "$branch" ]]; then
    git clone --branch "$branch" --depth 1 "$url" "$destination"
  else
    git clone "$url" "$destination"
  fi
}

download_if_missing() {
  local url="$1" destination="$2" temporary
  [[ -s "$destination" ]] && return
  temporary="${destination}.part"
  if curl -fsSL --retry 3 --retry-delay 2 "$url" -o "$temporary"; then
    mv -f "$temporary" "$destination"
  else
    rm -f "$temporary"
    echo "warning: failed to download $(basename "$destination") (check network)" >&2
  fi
}

# Example only: review versions and paths before running.
#
# Builds the reference environment that solved/reset_users.sh later clones
# into every teilnehmerN account, so every package/binary a notebook might
# need (SDK, DDS, Piper TTS, openaiapi, Jupyter) has to be reliable here
# first.
DEPS="${G1_DEPS_DIR:-$HOME/g1-deps}"
CYCLONE_SRC="$DEPS/cyclonedds"
CYCLONE_PREFIX="$CYCLONE_SRC/install"
SDK_SRC="$DEPS/unitree_sdk2_python"
VENV="$DEPS/venv"
VOICE_DIR="$DEPS/piper-voices"
# Matches util.py's PIPER_VOICES map — keep these in sync.
PIPER_VOICES=(en_US-lessac-medium de_DE-thorsten-medium fr_FR-siwis-medium es_ES-davefx-medium)

for command_name in git cmake python3 curl; do
  require_command "$command_name"
done

# espeak-ng ships the phoneme data some piper-phonemize wheels expect at
# runtime. Best-effort: don't fail the whole setup if apt isn't usable here.
if command -v apt-get >/dev/null 2>&1; then
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update && apt-get install -y espeak-ng || true
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y espeak-ng || true
  else
    echo "warning: sudo is unavailable; skipping optional espeak-ng installation" >&2
  fi
fi

mkdir -p "$DEPS" "$VOICE_DIR"
ensure_clone https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONE_SRC" releases/0.10.x
cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CYCLONE_PREFIX" -DBUILD_EXAMPLES=OFF
cmake --build "$CYCLONE_SRC/build" --parallel
cmake --install "$CYCLONE_SRC/build"

ensure_clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK_SRC"
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
    download_if_missing "$base/$voice.$ext" "$dest/$voice.$ext"
  done
done

printf 'Source %s/bin/activate and export CYCLONEDDS_HOME=%s\n' "$VENV" "$CYCLONE_PREFIX"
printf 'Also export LD_LIBRARY_PATH=%s/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$CYCLONE_PREFIX"
printf 'Also export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}\n' "$SDK_SRC"
printf 'Also export G1_PIPER_BIN=%s/bin/piper and G1_PIPER_VOICE_DIR=%s\n' "$VENV" "$VOICE_DIR"
