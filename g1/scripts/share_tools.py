#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: run_real_eval_server.sh [ENV OVERRIDES]

Common overrides:
  CKPT_PATH=/path/to/checkpoints/pytorch_model.pt
  VLM_PRETRAINED_PATH=/path/to/Unifolm-VLM-Base
  PORT=8777
  UNNORM_KEY=g1_stack_block
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Resolve script directory and repository root (two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

# Checkpoint and VLM paths (override via env vars)
ckpt_path=${CKPT_PATH:-/path/to/your/Unifolm-VLA-Base/checkpoints/pytorch_model.pt}
vlm_pretrained_path=${VLM_PRETRAINED_PATH:-/path/to/your/Unifolm-VLM-Base}
# Server port and dataset unnormalization key
port=${PORT:-8777}
unnorm_key=${UNNORM_KEY:-g1_stack_block}

# Basic validations
if [[ ! -d "${repo_root}/deployment/model_server" ]]; then
  echo "ERROR: repo_root looks incorrect: ${repo_root}"
  exit 1
fi
if [[ ! -f "${ckpt_path}" ]]; then
  echo "ERROR: checkpoint not found: ${ckpt_path}"
  exit 1
fi
if [[ ! -d "${vlm_pretrained_path}" ]]; then
  echo "ERROR: VLM_PRETRAINED_PATH not found: ${vlm_pretrained_path}"
  exit 1
fi
if [[ -z "${port}" ]]; then
  echo "ERROR: PORT is empty."
  exit 1
fi

# Launch the real evaluation model server
python "${repo_root}/deployment/model_server/run_real_eval_server.py" \
    --ckpt_path "${ckpt_path}" \
    --port "${port}" \
    --unnorm_key "${unnorm_key}" \
    --vlm_pretrained_path "${vlm_pretrained_path}"
