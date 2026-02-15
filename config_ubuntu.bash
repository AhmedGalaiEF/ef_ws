#!/usr/bin/env bash
set -euo pipefail

# Ubuntu provisioning for EF workspace.
# Run: sudo bash /home/ag/ef_ws/config_ubuntu.bash

show_help() {
  cat <<'USAGE'
Usage: sudo bash config_ubuntu.bash

What it does:
  1) apt update && apt upgrade -y
  2) Installs base build deps + mesa-utils
  3) Installs uv (Astral) via curl install script if not present
  4) Adds env-var block to the invoking user's ~/.bashrc using /home/ag/.bashrc as template
  5) Clones:
     - https://github.com/Livox-SDK/openpylivox.git -> ~/openpylivox
     - https://github.com/eclipse-cyclonedds/cyclonedds.git -> ~/cyclonedds
     - https://github.com/Livox-SDK/Livox-SDK2.git -> ~/Livox-SDK2
     - Unitree repos to ~/unitree_sdk2 and ~/unitree_sdk_python (URLs must be set)
  6) Builds:
     - CycloneDDS (install prefix: ~/cyclonedds/install)
     - Unitree SDK2 (install prefix: ~/unitree_sdk2/install)
     - Livox-SDK2 (build-only, used via LIVOX_SDK2_DIR)
  7) Creates uv venv at ~/.vens/python310 and installs:
     - openpylivox (editable from ~/openpylivox)
     - unitree_sdk2_python (editable from ~/unitree_sdk_python)

Required config for Unitree (set before running, or export in environment):
  UNITREE_SDK2_REPO=https://.../unitree_sdk2.git
  UNITREE_SDK_PY_REPO=https://.../unitree_sdk_python.git

Notes:
  - This script modifies the invoking user's ~/.bashrc (the user who ran sudo).
  - It does not run SLAM; it only provisions deps and builds.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

# User who invoked sudo.
REAL_USER="${SUDO_USER:-}"
if [[ -z "${REAL_USER}" ]]; then
  echo "SUDO_USER is empty. Run via sudo (not as a direct root login)." >&2
  exit 1
fi

REAL_HOME="$(getent passwd "${REAL_USER}" | cut -d: -f6)"
if [[ -z "${REAL_HOME}" || ! -d "${REAL_HOME}" ]]; then
  echo "Could not resolve home for ${REAL_USER}: '${REAL_HOME}'" >&2
  exit 1
fi

as_user() {
  sudo -u "${REAL_USER}" -H bash -lc "$*"
}

log() { printf '\n[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$*"; }

log "APT update/upgrade"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

log "Install base packages"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake ninja-build pkg-config git curl ca-certificates \
  python3-dev \
  mesa-utils libgl1-mesa-dev \
  libssl-dev

log "Install uv (Astral) if missing"
if ! as_user 'command -v uv >/dev/null 2>&1'; then
  # Official installer (installs to ~/.local/bin/uv by default).
  as_user 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
as_user 'uv --version'

log "Apply ~/.bashrc env template from /home/ag/.bashrc"
TEMPLATE_BASHRC="/home/ag/.bashrc"
TARGET_BASHRC="${REAL_HOME}/.bashrc"
MARK_BEGIN="# >>> ef_ws env (from /home/ag/.bashrc) >>>"
MARK_END="# <<< ef_ws env (from /home/ag/.bashrc) <<<"

# Extract exports/unsets, and de-ag-ify paths.
# This keeps it simple and predictable: only exports/unsets are copied.
TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_ENV}"' EXIT

awk '/^(export|unset)[[:space:]]+/{print}' "${TEMPLATE_BASHRC}" \
  | sed -e 's#/home/ag#'$REAL_HOME'#g' \
  > "${TMP_ENV}"

# Replace (or append) a marker block in target ~/.bashrc
if grep -qF "${MARK_BEGIN}" "${TARGET_BASHRC}" 2>/dev/null; then
  # Delete existing block
  perl -0777 -i -pe 's/\Q'"${MARK_BEGIN}"'\E.*?\Q'"${MARK_END}"'\E\n?/''/s' "${TARGET_BASHRC}"
fi

{
  echo
  echo "${MARK_BEGIN}"
  cat "${TMP_ENV}"
  echo "${MARK_END}"
} >> "${TARGET_BASHRC}"
chown "${REAL_USER}:${REAL_USER}" "${TARGET_BASHRC}"

clone_or_update() {
  local url="$1"
  local dst="$2"

  if [[ -d "${dst}/.git" ]]; then
    log "Update ${dst}"
    as_user "cd '${dst}' && git fetch --all --prune && git pull --ff-only"
  elif [[ -d "${dst}" ]]; then
    log "Skip clone (exists, not a git repo): ${dst}"
  else
    log "Clone ${url} -> ${dst}"
    as_user "git clone '${url}' '${dst}'"
  fi
}

log "Clone repos"
clone_or_update "https://github.com/Livox-SDK/openpylivox.git" "${REAL_HOME}/openpylivox"
clone_or_update "https://github.com/eclipse-cyclonedds/cyclonedds.git" "${REAL_HOME}/cyclonedds"
clone_or_update "https://github.com/Livox-SDK/Livox-SDK2.git" "${REAL_HOME}/Livox-SDK2"

if [[ -n "${UNITREE_SDK2_REPO:-}" ]]; then
  clone_or_update "${UNITREE_SDK2_REPO}" "${REAL_HOME}/unitree_sdk2"
else
  log "UNITREE_SDK2_REPO not set; skipping unitree_sdk2 clone"
fi

if [[ -n "${UNITREE_SDK_PY_REPO:-}" ]]; then
  clone_or_update "${UNITREE_SDK_PY_REPO}" "${REAL_HOME}/unitree_sdk_python"
else
  log "UNITREE_SDK_PY_REPO not set; skipping unitree_sdk_python clone"
fi

log "Build CycloneDDS"
if [[ -d "${REAL_HOME}/cyclonedds" ]]; then
  as_user "cd '${REAL_HOME}/cyclonedds' && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX='${REAL_HOME}/cyclonedds/install'"
  as_user "cmake --build '${REAL_HOME}/cyclonedds/build'"
  as_user "cmake --install '${REAL_HOME}/cyclonedds/build'"
fi

log "Build Unitree SDK2"
if [[ -d "${REAL_HOME}/unitree_sdk2'" ]]; then
  :
fi
if [[ -d "${REAL_HOME}/unitree_sdk2" ]]; then
  as_user "cd '${REAL_HOME}/unitree_sdk2' && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX='${REAL_HOME}/unitree_sdk2/install'"
  as_user "cmake --build '${REAL_HOME}/unitree_sdk2/build'"
  as_user "cmake --install '${REAL_HOME}/unitree_sdk2/build'"
fi

log "Build Livox-SDK2 (per SLAM steps flags)"
if [[ -d "${REAL_HOME}/Livox-SDK2" ]]; then
  as_user "cd '${REAL_HOME}/Livox-SDK2' && mkdir -p build"
  as_user "cd '${REAL_HOME}/Livox-SDK2/build' && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_FLAGS='-include cstdint'"
  as_user "cmake --build '${REAL_HOME}/Livox-SDK2/build'"
fi

log "Create uv venv ~/.vens/python310 and install Python deps"
VENV_DIR="${REAL_HOME}/.vens/python310"
as_user "uv venv --python 3.10 '${VENV_DIR}'"

# Install openpylivox (editable)
as_user "uv pip install --python '${VENV_DIR}/bin/python' -e '${REAL_HOME}/openpylivox'"

# Install unitree python wrapper (editable). Repo must exist.
if [[ -d "${REAL_HOME}/unitree_sdk_python" ]]; then
  as_user "uv pip install --python '${VENV_DIR}/bin/python' -e '${REAL_HOME}/unitree_sdk_python'"
else
  log "unitree_sdk_python not present; skipping python install"
fi

log "Done. Open a new shell or: source ~/.bashrc"
