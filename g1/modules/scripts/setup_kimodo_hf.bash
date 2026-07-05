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
#   3. Install system packages (python3.10 + venv, build tools).
#   4. Create a venv and install a Jetson-built PyTorch wheel matching the
#      detected JetPack version (from the community jetson-ai-lab index —
#      NOT plain PyPI, which has no aarch64+CUDA torch wheels at all).
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
#   --venv-dir PATH      Where to create the venv (default: $HOME/kimodo_venv)
#   --kimodo-repo URL    Git URL to pip-install kimodo from. Default depends
#                        on --quantize: the bitsandbytes fork for nf4/fp4/int8,
#                        the official nv-tlabs repo for cpu/none.
#   --index-url URL      Override the auto-detected Jetson PyTorch wheel index.
#   --min-free-gb N      Abort if free disk on $HOME's filesystem is below N
#                        GB (default 40).
#   --check-only         Run all detection/capacity checks and print a GO /
#                        NO-GO summary. Installs nothing.
#   --skip-apt           Skip the system apt-get step (already done).
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
INDEX_URL=""
MIN_FREE_GB=40
CHECK_ONLY=0
SKIP_APT=0
RUN_SMOKE_TEST=0
ASSUME_YES=0

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
    --index-url)      INDEX_URL="$2"; shift 2 ;;
    --min-free-gb)    MIN_FREE_GB="$2"; shift 2 ;;
    --check-only)     CHECK_ONLY=1; shift ;;
    --skip-apt)       SKIP_APT=1; shift ;;
    --run-smoke-test) RUN_SMOKE_TEST=1; shift ;;
    -y|--yes)         ASSUME_YES=1; shift ;;
    -h|--help)        sed -n '2,45p' "$0"; exit 0 ;;
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

# ── 3. System packages ─────────────────────────────────────────────────────
if [[ "$SKIP_APT" == "1" ]]; then
  info "Skipping apt-get step (--skip-apt)."
else
  log ""
  info "About to run apt-get to install: git, build-essential, cmake, ninja-build, python3.10, python3.10-venv"
  confirm "Run apt-get install now (requires sudo)?"
  SUDO=""
  [[ "$EUID" != "0" ]] && SUDO="sudo"
  $SUDO apt-get update
  if ! command -v python3.10 >/dev/null 2>&1; then
    if [[ "$JETPACK_MAJOR" == "5" ]]; then
      warn "JetPack 5 ships Python 3.8 by default; kimodo needs 3.10."
      warn "Adding the deadsnakes PPA to get python3.10 — review before trusting a third-party PPA on your robot's Jetson."
      confirm "Add ppa:deadsnakes/ppa and install python3.10 from it?"
      $SUDO apt-get install -y software-properties-common
      $SUDO add-apt-repository -y ppa:deadsnakes/ppa
      $SUDO apt-get update
    fi
  fi
  $SUDO apt-get install -y git build-essential cmake ninja-build python3.10 python3.10-venv
  ok "System packages installed."
fi

command -v python3.10 >/dev/null 2>&1 || die "python3.10 still not available after the apt step. Install it manually (kimodo requires Python 3.10) and re-run with --skip-apt."

# ── 4. Venv + Jetson PyTorch wheel ────────────────────────────────────────
log ""
info "Creating venv at $VENV_DIR …"
python3.10 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

if [[ -z "$INDEX_URL" ]]; then
  case "$JETPACK_MAJOR" in
    6) INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126" ;;
    *)
      die "No known Jetson PyTorch wheel index for this JetPack version. Find the right wheel for your JetPack/CUDA at https://pypi.jetson-ai-lab.io/ or the NVIDIA Jetson PyTorch forum thread, then re-run with --index-url."
      ;;
  esac
fi

info "Installing torch/torchvision from Jetson wheel index: $INDEX_URL"
warn "This index is community-maintained (jetson-ai-lab), not PyPI or NVIDIA-official. Verify it's reachable and trusted in your network before proceeding."
confirm "Install torch/torchvision from $INDEX_URL ?"
pip install torch torchvision --index-url "$INDEX_URL"
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ "$QUANTIZE" != "cpu" && "$QUANTIZE" != "none" ]]; then
  info "Installing bitsandbytes for quantized ($QUANTIZE) text encoding …"
  warn "bitsandbytes on Jetson/aarch64 is not officially supported by either bitsandbytes or Kimodo — this may fail to build or may miscompute silently on untested GPU architectures."
  if ! pip install bitsandbytes --index-url "$INDEX_URL"; then
    warn "Prebuilt bitsandbytes wheel not found on $INDEX_URL for this platform."
    warn "You would need to build bitsandbytes from source for this Jetson's compute capability — not attempted automatically by this script."
    die "bitsandbytes install failed. Re-run with --quantize cpu to skip quantization, or build bitsandbytes manually first."
  fi
  ok "bitsandbytes installed."
fi

# ── 5. Kimodo package ──────────────────────────────────────────────────────
log ""
info "Installing kimodo from: $KIMODO_REPO"
confirm "pip install kimodo[all] from this repo?"
pip install "kimodo[all] @ ${KIMODO_REPO}"
ok "kimodo package installed."

# ── 6. Hugging Face auth ───────────────────────────────────────────────────
log ""
log "${C_B}Hugging Face auth${C_X}"
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  ok "Found an existing HF token at ~/.cache/huggingface/token."
else
  warn "No HF token found. Kimodo needs one to download nvidia/Kimodo-G1-SEED-v1"
  warn "and the gated meta-llama/Meta-Llama-3-8B-Instruct text encoder."
  warn "You must accept the Llama 3 license on its Hugging Face page with the"
  warn "same account before this will work: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
  if [[ "$ASSUME_YES" != "1" ]]; then
    read -r -p "Run 'hf auth login' now? [y/N] " reply || true
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      hf auth login
    else
      warn "Skipping. Run 'hf auth login' yourself before generating."
    fi
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
  python3 - <<'PYEOF'
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
