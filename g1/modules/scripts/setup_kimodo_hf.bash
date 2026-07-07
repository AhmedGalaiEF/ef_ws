#!/usr/bin/env bash
# setup_kimodo_hf.bash — install Kimodo + its Hugging Face model/text-encoder
# dependencies on the G1 EDU's onboard Jetson.
#
# Run this ON THE JETSON (not the dev workstation). See setup_kimodo_hf.md
# for why: Jetson uses unified memory (no separate VRAM), so Kimodo's normal
# ~17GB VRAM footprint has to be brought down to fit total device RAM before
# it's worth attempting here at all. This script does NOT silently downgrade
# that decision for you — it measures your actual hardware, tells you what
# will/won't fit, and only proceeds past detection with --yes or an
# interactive confirmation.
#
# What it does, in order:
#   1. Confirm this is actually a Jetson, and detect L4T/JetPack version.
#   2. Check free disk space and total RAM against the mode you selected.
#   3. Install system packages (build tools, plus python + venv only when
#      the script needs to create its own environment).
#   4. Reuse a compatible active Python env, or create/select a venv, then
#      install a Jetson-built PyTorch wheel matching the detected JetPack
#      version. JetPack 5 uses NVIDIA's direct cp38 wheel; JetPack 6 can use
#      the community jetson-ai-lab index. Plain PyPI has no aarch64+CUDA torch
#      wheels for Jetson.
#   5. Install the `kimodo` package (official repo, or the bitsandbytes
#      quantized fork — see --quantize).
#   6. Prompt for Hugging Face auth (needed for the gated
#      meta-llama/Meta-Llama-3-8B-Instruct text-encoder weights).
#   7. Optionally run a small smoke-test generation to confirm the whole
#      pipeline actually produces a motion on this device.
#
# Usage
# -----
#   ./setup_kimodo_hf.bash [options]
#
# Options
# -------
#   --quantize MODE     nf4 (default) | fp4 | int8 | cpu | none
#                        See setup_kimodo_hf.md for what each mode needs and
#                        whether it plausibly fits in your Jetson's RAM.
#   --venv-dir PATH      Where to create the venv (default: $HOME/kimodo_venv).
#                        If a supported virtualenv/guv env is already active,
#                        that active env is reused instead.
#   --force-new-venv     Ignore an active virtualenv/guv env and create/use
#                        --venv-dir instead.
#   --kimodo-repo URL    Git URL to pip-install kimodo from. Default depends
#                        on --quantize: the bitsandbytes fork for nf4/fp4/int8,
#                        the official nv-tlabs repo for cpu/none.
#   --kimodo-extra NAME  Optional dependency extra to install. Default: none on
#                        JetPack 5, all elsewhere. Use "all" only with Python
#                        3.10+ because py-soma-x requires it.
#   --index-url URL      Override the auto-detected Jetson PyTorch wheel index.
#   --torch-wheel URL    Install torch from a direct wheel URL/path instead of
#                        an index. Needed for official JetPack 5 wheels.
#   --generic-torch      Use the active Python 3.10 env and install generic
#                        PyTorch from PyPI/default index instead of a Jetson
#                        CUDA wheel. This may be CPU-only on JetPack 5.
#   --min-free-gb N      Abort if free disk on $HOME's filesystem is below N
#                        GB (default 40).
#   --check-only         Run all detection/capacity checks and print a GO /
#                        NO-GO summary. Installs nothing.
#   --skip-apt           Skip the system apt-get step (already done).
#   --torch-only         Install/verify torch, then stop before bitsandbytes
#                        and kimodo package installation.
#   --force-kimodo       Attempt kimodo installation even when the detected
#                        JetPack/Python stack is known incompatible.
#   --run-smoke-test     After install, generate one short test motion to
#                        confirm the pipeline actually works end to end.
#   -y, --yes            Don't pause for confirmation before apt/pip steps.
#   -h, --help           Show this help and exit.
#
# This script is deliberately conservative: every irreversible or slow step
# (apt-get, multi-GB downloads) is behind a confirmation prompt unless -y is
# given, and every hardware assumption it makes is printed so you can catch
# a wrong guess before it wastes your bandwidth/disk.

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────
QUANTIZE="nf4"
VENV_DIR="${HOME}/kimodo_venv"
KIMODO_REPO=""
KIMODO_EXTRA=""
INDEX_URL=""
TORCH_WHEEL=""
GENERIC_TORCH=0
MIN_FREE_GB=40
CHECK_ONLY=0
SKIP_APT=0
RUN_SMOKE_TEST=0
TORCH_ONLY=0
ASSUME_YES=0
FORCE_NEW_VENV=0
FORCE_KIMODO=0

OFFICIAL_REPO="git+https://github.com/nv-tlabs/kimodo.git"
QUANT_FORK_REPO="git+https://github.com/matbeedotcom/kimodo.git"

# ── Colour (disabled if not a tty) ────────────────────────────────────────
if [[ -t 1 ]]; then
  C_R=$'\033[91m'; C_Y=$'\033[93m'; C_G=$'\033[92m'; C_C=$'\033[96m'; C_B=$'\033[1m'; C_X=$'\033[0m'
else
  C_R=""; C_Y=""; C_G=""; C_C=""; C_B=""; C_X=""
fi

log()  { printf '%s\n' "$*"; }
info() { printf '%s[*]%s %s\n' "$C_C" "$C_X" "$*"; }
ok()   { printf '%s[ok]%s %s\n' "$C_G" "$C_X" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_Y" "$C_X" "$*"; }
die()  { printf '%s[fail]%s %s\n' "$C_R" "$C_X" "$*" >&2; exit 1; }

confirm() {
  # confirm "prompt text"  — aborts the script if the user declines.
  if [[ "$ASSUME_YES" == "1" ]]; then return 0; fi
  local reply
  read -r -p "$1 [y/N] " reply || true
  [[ "$reply" =~ ^[Yy]$ ]] || die "Aborted by user."
}

# ── Arg parsing ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quantize)       QUANTIZE="$2"; shift 2 ;;
    --venv-dir)       VENV_DIR="$2"; shift 2 ;;
    --kimodo-repo)    KIMODO_REPO="$2"; shift 2 ;;
    --kimodo-extra)   KIMODO_EXTRA="$2"; shift 2 ;;
    --index-url)      INDEX_URL="$2"; shift 2 ;;
    --torch-wheel)    TORCH_WHEEL="$2"; shift 2 ;;
    --generic-torch)  GENERIC_TORCH=1; shift ;;
    --min-free-gb)    MIN_FREE_GB="$2"; shift 2 ;;
    --check-only)     CHECK_ONLY=1; shift ;;
    --skip-apt)       SKIP_APT=1; shift ;;
    --torch-only)     TORCH_ONLY=1; shift ;;
    --run-smoke-test) RUN_SMOKE_TEST=1; shift ;;
    --force-new-venv) FORCE_NEW_VENV=1; shift ;;
    --force-kimodo)   FORCE_KIMODO=1; shift ;;
    -y|--yes)         ASSUME_YES=1; shift ;;
    -h|--help)        sed -n '2,62p' "$0"; exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

case "$QUANTIZE" in
  nf4|fp4|int8|cpu|none) ;;
  *) die "--quantize must be one of: nf4, fp4, int8, cpu, none (got '$QUANTIZE')" ;;
esac

if [[ -z "$KIMODO_REPO" ]]; then
  case "$QUANTIZE" in
    nf4|fp4|int8) KIMODO_REPO="$QUANT_FORK_REPO" ;;
    cpu|none)     KIMODO_REPO="$OFFICIAL_REPO" ;;
  esac
fi

log "${C_B}────────────────────────────────────────────────────────────${C_X}"
log "${C_B}  Kimodo + Hugging Face setup for the G1 EDU onboard Jetson${C_X}"
log "${C_B}────────────────────────────────────────────────────────────${C_X}"

python_major_minor() {
  local exe="$1"
  "$exe" - <<'PYEOF'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PYEOF
}

python_is_version() {
  local exe="$1"
  local want="$2"
  [[ "$(python_major_minor "$exe" 2>/dev/null || true)" == "$want" ]]
}

python_abi_tag() {
  local exe="$1"
  "$exe" - <<'PYEOF'
import sys
print(f"cp{sys.version_info[0]}{sys.version_info[1]}")
PYEOF
}

python_has_modules() {
  local exe="$1"
  shift
  "$exe" - "$@" <<'PYEOF' >/dev/null 2>&1
import importlib.util
import sys
missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PYEOF
}

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    if [[ -z "${UV_CACHE_DIR:-}" && ! -w "${HOME}/.cache/uv" ]]; then
      export UV_CACHE_DIR="/tmp/uv-cache"
      mkdir -p "$UV_CACHE_DIR"
    fi
    uv pip install --python "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" -m pip install "$@"
  fi
}

install_kimodo_runtime_deps_without_viz() {
  # Kimodo's declared base deps include visualization/demo packages that do not
  # currently build cleanly on JetPack 5 aarch64 (notably scenepic). The robot
  # pipeline imports kimodo.model.load_model, so install the model/runtime deps
  # directly and then install kimodo itself with --no-deps.
  pip_install \
    "hydra-core>=1.3" \
    "omegaconf>=2.3" \
    "numpy>=1.23" \
    "scipy>=1.10" \
    "transformers==5.1.0" \
    "urllib3>=2.6.3" \
    "boto3" \
    "peft>=0.18" \
    "einops>=0.7" \
    "tqdm>=4.0" \
    "packaging>=21.0" \
    "pydantic>=2.0" \
    "filelock>=3.20.3" \
    "trimesh>=3.21.7" \
    "pillow>=9.0" \
    "av>=16.1.0" \
    "bvhio"
}

# ── 1. Confirm this is a Jetson, detect L4T/JetPack ───────────────────────
info "Checking hardware …"

IS_JETSON=0
MODEL_STR=""
if [[ -r /proc/device-tree/model ]]; then
  MODEL_STR="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
  [[ "$MODEL_STR" == *NVIDIA*Jetson* || "$MODEL_STR" == *Orin* || "$MODEL_STR" == *Tegra* ]] && IS_JETSON=1
fi

L4T_VERSION=""
if [[ -r /etc/nv_tegra_release ]]; then
  IS_JETSON=1
  L4T_VERSION="$(head -1 /etc/nv_tegra_release)"
fi

if [[ "$IS_JETSON" != "1" ]]; then
  die "This does not look like a Jetson (no /etc/nv_tegra_release, no matching /proc/device-tree/model). Run this ON the G1 EDU's onboard Jetson, not your workstation."
fi

ok "Jetson detected.${MODEL_STR:+ Model: $MODEL_STR}"
[[ -n "$L4T_VERSION" ]] && log "  L4T release string: $L4T_VERSION"

# L4T R36.x -> JetPack 6.x ; R35.x -> JetPack 5.x ; older -> unsupported here.
JETPACK_MAJOR=""
if [[ "$L4T_VERSION" =~ R([0-9]+) ]]; then
  L4T_MAJOR="${BASH_REMATCH[1]}"
  case "$L4T_MAJOR" in
    36) JETPACK_MAJOR=6 ;;
    35) JETPACK_MAJOR=5 ;;
    *)  JETPACK_MAJOR="" ;;
  esac
fi

if [[ -n "$JETPACK_MAJOR" ]]; then
  ok "Detected JetPack ${JETPACK_MAJOR}.x (L4T R${L4T_MAJOR})."
else
  warn "Could not map L4T version to a JetPack major version from '$L4T_VERSION'."
  warn "This script only has a wheel-index mapping for JetPack 5 and 6."
fi

CUDA_VERSION=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
  ok "CUDA toolkit: ${CUDA_VERSION:-unknown}"
else
  warn "nvcc not found — CUDA toolkit may not be on PATH. JetPack normally ships one at /usr/local/cuda."
fi

# ── 2. Memory / disk reality check ────────────────────────────────────────
TOTAL_RAM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
FREE_DISK_GB="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')"

log ""
log "${C_B}Hardware summary${C_X}"
log "  Unified RAM (shared CPU+GPU, no separate VRAM on Jetson): ${TOTAL_RAM_GB} GB total"
log "  Free disk on \$HOME filesystem: ${FREE_DISK_GB} GB"

# Rough peak-memory expectations per mode (text encoder + kimodo diffusion
# model + activations; does NOT include OS/ROS/DDS overhead already running
# on the robot's Jetson). See setup_kimodo_hf.md for sourcing/derivation.
case "$QUANTIZE" in
  nf4)    EST_PEAK_GB=6  ;;
  fp4)    EST_PEAK_GB=6  ;;
  int8)   EST_PEAK_GB=9  ;;
  cpu)    EST_PEAK_GB=17 ;;   # 8B text encoder in fp16 on CPU RAM + diffusion model
  none)   EST_PEAK_GB=18 ;;   # official config, everything on GPU
esac

log "  Estimated peak memory for --quantize=$QUANTIZE: ~${EST_PEAK_GB} GB"
log "    (leaves ~$(( TOTAL_RAM_GB - EST_PEAK_GB )) GB for the OS, ROS/DDS stack, and everything else)"

RECOMMEND_GO=1
if (( TOTAL_RAM_GB - EST_PEAK_GB < 3 )); then
  warn "Less than 3GB of RAM would be left for the OS + robot control stack."
  warn "This is very likely to fail under memory pressure or destabilize the robot's control loop."
  RECOMMEND_GO=0
fi
if (( FREE_DISK_GB < MIN_FREE_GB )); then
  warn "Free disk (${FREE_DISK_GB}GB) is below --min-free-gb (${MIN_FREE_GB}GB)."
  RECOMMEND_GO=0
fi

log ""
if [[ "$RECOMMEND_GO" == "1" ]]; then
  ok "GO: hardware numbers look plausible for --quantize=$QUANTIZE. This is an estimate, not a guarantee."
else
  warn "NO-GO by this script's numbers. See setup_kimodo_hf.md for options (lower quantization, or run Kimodo off-board on a workstation instead)."
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  log ""
  info "--check-only: stopping here, nothing was installed."
  exit 0
fi

if [[ "$RECOMMEND_GO" != "1" ]]; then
  confirm "Proceed anyway despite the NO-GO estimate above?"
fi

if [[ "$JETPACK_MAJOR" == "5" && "$QUANTIZE" != "cpu" && "$QUANTIZE" != "none" && -z "$INDEX_URL" ]]; then
  warn "JetPack 5 torch can be installed from NVIDIA's cp38 wheel, but quantized mode also needs bitsandbytes."
  warn "No JetPack 5 bitsandbytes wheel index is configured. After torch installs, this script will stop unless bitsandbytes is already installed or you pass a working --index-url."
fi

SELECTED_ENV=""
SELECTED_ENV_KIND=""
REQUIRED_PYTHON=""
case "$JETPACK_MAJOR" in
  5) REQUIRED_PYTHON="3.8" ;;
  6) REQUIRED_PYTHON="3.10" ;;
esac
if [[ "$GENERIC_TORCH" == "1" ]]; then
  REQUIRED_PYTHON=""
  warn "--generic-torch selected: using generic PyTorch packaging instead of Jetson-specific CUDA torch wheels."
  warn "On JetPack 5 this is expected to be CPU-only unless your index provides a compatible aarch64 CUDA build."
fi

if [[ "$FORCE_NEW_VENV" != "1" && -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  ACTIVE_VERSION="$(python_major_minor "${VIRTUAL_ENV}/bin/python" 2>/dev/null || true)"
  if [[ -z "$REQUIRED_PYTHON" || "$ACTIVE_VERSION" == "$REQUIRED_PYTHON" || "$(python_has_modules "${VIRTUAL_ENV}/bin/python" torch && echo yes || true)" == "yes" ]]; then
    SELECTED_ENV="$VIRTUAL_ENV"
    SELECTED_ENV_KIND="active"
    ok "Will reuse active Python ${ACTIVE_VERSION} environment: $VIRTUAL_ENV"
  elif [[ "$JETPACK_MAJOR" == "5" && "$ACTIVE_VERSION" == "3.10" ]]; then
    warn "Active env is Python 3.10, but JetPack 5 NVIDIA torch wheels are Python 3.8 (cp38)."
  else
    warn "Active environment Python ${ACTIVE_VERSION:-unknown} does not match the expected Python ${REQUIRED_PYTHON:-version} for JetPack ${JETPACK_MAJOR:-unknown}."
  fi
fi

if [[ -z "$SELECTED_ENV" && "$FORCE_NEW_VENV" != "1" && "$GENERIC_TORCH" != "1" && "$JETPACK_MAJOR" == "5" && -x "$HOME/.guv/envs/base/bin/python" ]]; then
  if python_is_version "$HOME/.guv/envs/base/bin/python" "3.8"; then
    SELECTED_ENV="$HOME/.guv/envs/base"
    SELECTED_ENV_KIND="guv-base"
    VENV_DIR="$SELECTED_ENV"
    ok "Will use JetPack 5 compatible Python 3.8 guv environment: $SELECTED_ENV"
  fi
fi

if [[ -n "$SELECTED_ENV" ]]; then
  VENV_DIR="$SELECTED_ENV"
fi

# ── 3. System packages ─────────────────────────────────────────────────────
if [[ "$SKIP_APT" == "1" ]]; then
  info "Skipping apt-get step (--skip-apt)."
else
  log ""
  APT_PACKAGES=(git build-essential cmake ninja-build)
  if [[ "$JETPACK_MAJOR" == "5" ]]; then
    APT_PACKAGES+=(python3-pip libopenblas-base libopenmpi-dev libomp-dev)
  elif [[ -z "$SELECTED_ENV" ]]; then
    APT_PACKAGES+=(python3.10 python3.10-venv)
  fi
  info "About to run apt-get to install: ${APT_PACKAGES[*]}"
  confirm "Run apt-get install now (requires sudo)?"
  SUDO=""
  [[ "$EUID" != "0" ]] && SUDO="sudo"
  $SUDO apt-get update
  if [[ -z "$SELECTED_ENV" && "$JETPACK_MAJOR" != "5" ]] && ! command -v python3.10 >/dev/null 2>&1; then
    if [[ "$JETPACK_MAJOR" == "5" ]]; then
      warn "JetPack 5 ships Python 3.8 by default; kimodo needs 3.10."
      warn "Adding the deadsnakes PPA to get python3.10 — review before trusting a third-party PPA on your robot's Jetson."
      confirm "Add ppa:deadsnakes/ppa and install python3.10 from it?"
      $SUDO apt-get install -y software-properties-common
      $SUDO add-apt-repository -y ppa:deadsnakes/ppa
      $SUDO apt-get update
    fi
  fi
  $SUDO apt-get install -y "${APT_PACKAGES[@]}"
  ok "System packages installed."
fi

# ── 4. Python env + Jetson PyTorch wheel ──────────────────────────────────
log ""
if [[ -n "$SELECTED_ENV" ]]; then
  info "Using ${SELECTED_ENV_KIND:-selected} environment at $VENV_DIR …"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  command -v python3.10 >/dev/null 2>&1 || die "python3.10 still not available. Activate a compatible guv/env first, or install python3.10 + python3.10-venv and re-run."
  info "Creating venv at $VENV_DIR …"
  python3.10 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
PYTHON_VERSION="$(python_major_minor "$PYTHON_BIN")"
PYTHON_ABI="$(python_abi_tag "$PYTHON_BIN")"
HAS_TORCH=0
python_has_modules "$PYTHON_BIN" torch && HAS_TORCH=1
if [[ -n "$REQUIRED_PYTHON" && "$PYTHON_VERSION" != "$REQUIRED_PYTHON" && "$HAS_TORCH" != "1" ]]; then
  die "Selected Python is $PYTHON_VERSION ($PYTHON_ABI), but JetPack $JETPACK_MAJOR torch wheels require Python $REQUIRED_PYTHON. For this robot, use: source ~/.guv/envs/base/bin/activate"
fi
ok "Selected Python: $($PYTHON_BIN --version 2>&1) ($PYTHON_ABI)"

if command -v uv >/dev/null 2>&1; then
  ok "Using uv pip for package installs."
elif ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  warn "pip is missing from $VENV_DIR; bootstrapping it with ensurepip."
  "$PYTHON_BIN" -m ensurepip --upgrade || die "Could not bootstrap pip in $VENV_DIR. Check that the environment is writable, or install uv."
  "$PYTHON_BIN" -m pip install --upgrade pip
else
  "$PYTHON_BIN" -m pip install --upgrade pip
fi

TORCH_READY=0
if python_has_modules "$PYTHON_BIN" torch; then
  TORCH_READY=1
  ok "torch already installed in $VENV_DIR."
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
fi

if [[ "$TORCH_READY" != "1" ]]; then
  if [[ "$GENERIC_TORCH" == "1" ]]; then
    info "Installing generic PyTorch into $VENV_DIR …"
    warn "This is not NVIDIA's JetPack CUDA torch wheel. Verify cuda availability after install."
    confirm "Install generic torch from the configured/default Python package index?"
    if command -v uv >/dev/null 2>&1; then
      pip_install --torch-backend cpu torch
    else
      pip_install torch
    fi
    "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  elif [[ -z "$TORCH_WHEEL" && "$JETPACK_MAJOR" == "5" ]]; then
    case "$L4T_VERSION" in
      *R35*) TORCH_WHEEL="https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl" ;;
    esac
  fi

  if [[ "$GENERIC_TORCH" == "1" ]]; then
    :
  elif [[ -n "$TORCH_WHEEL" ]]; then
    case "$TORCH_WHEEL" in
      *cp38*cp38*) [[ "$PYTHON_ABI" == "cp38" ]] || die "$TORCH_WHEEL is a cp38 wheel, but selected Python is $PYTHON_ABI. Activate ~/.guv/envs/base or pass a wheel matching $PYTHON_ABI." ;;
      *cp310*cp310*) [[ "$PYTHON_ABI" == "cp310" ]] || die "$TORCH_WHEEL is a cp310 wheel, but selected Python is $PYTHON_ABI." ;;
    esac
    info "Installing numpy and torch from NVIDIA JetPack wheel:"
    log "  $TORCH_WHEEL"
    confirm "Install torch from this direct wheel?"
    if [[ "$PYTHON_VERSION" == "3.8" ]]; then
      pip_install "numpy<2"
    else
      pip_install "numpy==1.26.1"
    fi
    pip_install --no-cache-dir "$TORCH_WHEEL"
    "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  else
    if [[ -z "$INDEX_URL" ]]; then
    case "$JETPACK_MAJOR" in
      6) INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126" ;;
      *)
        die "No torch wheel configured for this JetPack. For JetPack 5.1.1, use Python 3.8 and the NVIDIA cp38 wheel, or pass --torch-wheel with a compatible direct wheel URL/path."
        ;;
    esac
    fi

    info "Installing torch from Jetson wheel index: $INDEX_URL"
    warn "This index is community-maintained (jetson-ai-lab), not PyPI or NVIDIA-official. Verify it's reachable and trusted in your network before proceeding."
    confirm "Install torch from $INDEX_URL ?"
    pip_install torch --index-url "$INDEX_URL"
    "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  fi
fi

if [[ "$TORCH_ONLY" == "1" ]]; then
  log ""
  ok "--torch-only: torch setup finished. Stopping before bitsandbytes/kimodo installation."
  log "    source $VENV_DIR/bin/activate"
  log "    python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
  exit 0
fi

if [[ "$JETPACK_MAJOR" == "5" && "$GENERIC_TORCH" != "1" && "$PYTHON_VERSION" == "3.8" && "$FORCE_KIMODO" != "1" ]]; then
  log ""
  warn "Torch CUDA is installed, but current Kimodo cannot be installed on this JetPack 5 Python 3.8 environment."
  warn "Kimodo's current dependency tree pulls Python 3.10+ packages (for example py-soma-x via kimodo[all], and current Gradio/Transformers in the base package)."
  warn "Using Python 3.10 on this JetPack 5 image loses the NVIDIA CUDA torch wheel, because the available JetPack 5 torch wheel is cp38."
  die "No compatible on-robot Kimodo install path is available for JetPack 5/L4T R35. Use --torch-only, upgrade the robot image to JetPack 6, or run Kimodo off-board on a workstation."
fi

if [[ "$JETPACK_MAJOR" == "5" && "$QUANTIZE" != "cpu" && "$QUANTIZE" != "none" && -z "$INDEX_URL" ]] && ! python_has_modules "$PYTHON_BIN" bitsandbytes; then
  log ""
  warn "Cannot automate $QUANTIZE quantization on this JetPack 5 stack."
  warn "Current upstream bitsandbytes requires newer Python/PyTorch/CUDA than JetPack 5 provides, and the JetPack 6 wheel indexes do not apply to L4T R35/CUDA 11.4."
  warn "The automated fallback is --quantize cpu, which skips bitsandbytes but may run out of memory on a 15GB Jetson."
  confirm "Switch to --quantize cpu and continue without bitsandbytes?"
  QUANTIZE="cpu"
  KIMODO_REPO="$OFFICIAL_REPO"
fi

if [[ "$QUANTIZE" != "cpu" && "$QUANTIZE" != "none" ]]; then
  if python_has_modules "$PYTHON_BIN" bitsandbytes; then
    ok "bitsandbytes already installed in $VENV_DIR."
  else
    [[ -n "$INDEX_URL" ]] || die "No wheel index is configured for bitsandbytes. Preinstall bitsandbytes, pass --index-url, or re-run with --quantize cpu."
    info "Installing bitsandbytes for quantized ($QUANTIZE) text encoding …"
    warn "bitsandbytes on Jetson/aarch64 is not officially supported by either bitsandbytes or Kimodo — this may fail to build or may miscompute silently on untested GPU architectures."
    if ! pip_install bitsandbytes --index-url "$INDEX_URL"; then
      warn "Prebuilt bitsandbytes wheel not found on $INDEX_URL for this platform."
      warn "You would need to build bitsandbytes from source for this Jetson's compute capability — not attempted automatically by this script."
      die "bitsandbytes install failed. Re-run with --quantize cpu to skip quantization, or build bitsandbytes manually first."
    fi
    ok "bitsandbytes installed."
  fi
fi

# ── 5. Kimodo package ──────────────────────────────────────────────────────
log ""
info "Installing kimodo from: $KIMODO_REPO"
KIMODO_NO_DEPS=0
if [[ -z "$KIMODO_EXTRA" ]]; then
  if [[ "$GENERIC_TORCH" == "1" && "$JETPACK_MAJOR" == "5" ]]; then
    KIMODO_SPEC="kimodo @ ${KIMODO_REPO}"
    warn "Installing base kimodo package only. kimodo[all] pulls py-soma-x/usd-core, which has no aarch64 Python 3.10 wheel."
    export SKIP_MOTION_CORRECTION_IN_SETUP=1
    warn "Skipping Kimodo's bundled motion_correction C++ extension; it uses x86-only build flags on aarch64."
    warn "Installing Kimodo model/runtime dependencies without scenepic/gradio visualization dependencies."
    install_kimodo_runtime_deps_without_viz
    KIMODO_NO_DEPS=1
  else
    KIMODO_SPEC="kimodo[all] @ ${KIMODO_REPO}"
  fi
elif [[ "$KIMODO_EXTRA" == "none" ]]; then
  KIMODO_SPEC="kimodo @ ${KIMODO_REPO}"
  if [[ "$GENERIC_TORCH" == "1" && "$JETPACK_MAJOR" == "5" ]]; then
    export SKIP_MOTION_CORRECTION_IN_SETUP=1
    warn "Skipping Kimodo's bundled motion_correction C++ extension; it uses x86-only build flags on aarch64."
  fi
else
  KIMODO_SPEC="kimodo[${KIMODO_EXTRA}] @ ${KIMODO_REPO}"
fi
confirm "pip install ${KIMODO_SPEC}?"
if [[ "$KIMODO_NO_DEPS" == "1" ]]; then
  pip_install --no-deps "$KIMODO_SPEC"
else
  pip_install "$KIMODO_SPEC"
fi
ok "kimodo package installed."

# ── 6. Hugging Face auth ───────────────────────────────────────────────────
log ""
log "${C_B}Hugging Face auth${C_X}"
HF_AUTH_OK=0
if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
  ok "Hugging Face CLI is logged in: $(hf auth whoami 2>/dev/null | head -1)"
  HF_AUTH_OK=1
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
  ok "Found an existing HF token at ~/.cache/huggingface/token."
  HF_AUTH_OK=1
fi

if [[ "$HF_AUTH_OK" != "1" ]]; then
  warn "No HF token found. Kimodo needs one to download nvidia/Kimodo-G1-SEED-v1"
  warn "and the gated meta-llama/Meta-Llama-3-8B-Instruct text encoder."
  warn "You must accept the Llama 3 license on its Hugging Face page with the"
  warn "same account before this will work: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
  if [[ "$ASSUME_YES" != "1" ]]; then
    read -r -p "Run 'hf auth login' now? [y/N] " reply || true
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      command -v hf >/dev/null 2>&1 || die "'hf' command not found after install; try: $PYTHON_BIN -m pip install huggingface_hub"
      hf auth login
      HF_AUTH_OK=1
    else
      warn "Skipping. Run 'hf auth login' yourself before generating."
    fi
  fi
fi

if [[ "$HF_AUTH_OK" == "1" ]]; then
  info "Checking gated Llama text-encoder access …"
  if "$PYTHON_BIN" - <<'PYEOF'
from huggingface_hub import hf_hub_download

hf_hub_download("meta-llama/Meta-Llama-3-8B-Instruct", "config.json")
print("Llama text-encoder access OK.")
PYEOF
  then
    ok "Hugging Face gated model access verified."
  else
    warn "Could not verify access to meta-llama/Meta-Llama-3-8B-Instruct."
    warn "Log in with 'hf auth login' and accept the Llama 3 license on Hugging Face before running Kimodo."
  fi
fi

if [[ "$QUANTIZE" != "none" && "$QUANTIZE" != "cpu" ]]; then
  export LLM2VEC_QUANTIZE="$QUANTIZE"
  ok "Set LLM2VEC_QUANTIZE=$QUANTIZE for this session. Add it to your shell profile to persist it."
elif [[ "$QUANTIZE" == "cpu" ]]; then
  export TEXT_ENCODER_DEVICE=cpu
  ok "Set TEXT_ENCODER_DEVICE=cpu for this session. Add it to your shell profile to persist it."
fi

# ── 7. Optional smoke test ────────────────────────────────────────────────
if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
  log ""
  info "Running a short smoke-test generation (no robot involved) …"
  "$PYTHON_BIN" - <<'PYEOF'
import time
from kimodo.model import load_model

t0 = time.monotonic()
model = load_model("nvidia/Kimodo-G1-SEED-v1")
print(f"Model loaded in {time.monotonic()-t0:.1f}s.")

t0 = time.monotonic()
result = model(
    text=["a person waves with their right hand"],
    num_frames=30,
    fps=30,
    num_denoising_steps=20,
    as_numpy=True,
)
print(f"Generated in {time.monotonic()-t0:.1f}s.")
lrm = result["local_rot_mats"]
print("local_rot_mats shape:", lrm.shape)
print("Smoke test OK.")
PYEOF
  ok "Smoke test finished."
else
  log ""
  info "Skipped smoke test (pass --run-smoke-test to generate one test motion)."
  info "To test with the actual robot pipeline (safety analysis, no robot connection needed):"
  log "    source $VENV_DIR/bin/activate"
  log "    python3 kimodo_interactive.py --snapshot dev_stand_snapshot.json --no-robot"
fi

log ""
ok "Setup finished. Activate with: source $VENV_DIR/bin/activate"
