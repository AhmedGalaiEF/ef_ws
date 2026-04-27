#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: run_libero_train.sh [ENV OVERRIDES]

Common overrides:
  VLM=/path/to/Unifolm-VLM-0
  OXE=/path/to/data
  DATA_MIX=libero_4_task_no_noops
  RUN_ROOT_DIR=/path/to/runs
  RUN_ID=exp_name
  NUM_PROCESSES=8
  DS_CONFIG=/path/to/deepspeed_zero2.yaml

Notes:
  - Requires: accelerate (HF), deepspeed config, valid data and model paths.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v accelerate >/dev/null 2>&1 || { echo "ERROR: accelerate not found in PATH."; exit 1; }

# NCCL settings for distributed training (adjust to your network)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}


# Model configuration
# VLM backbone
Framework_name=unifolm_vla
base_vlm=${VLM:-/path/to/your/Unifolm-VLM-0}
model_type=qwen2_5_vl
freeze_module_list='' 
window_size=2
# Dataset configuration
# VLA dataset
oxe_data_root=${OXE:-/path/to/your/data}
data_mix=${DATA_MIX:-your_data_mix}   # libero_4_task_no_noops libero_90_no_noops

# Run output path
run_root_dir=${RUN_ROOT_DIR:-/path/to/your/run_root_dir}
run_id=${RUN_ID:-your_run_id}

# Resolve script directory and repository root (two levels up)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

# Basic validations
if [[ ! -d "${repo_root}/src/unifolm_vla" ]]; then
  echo "ERROR: repo_root looks incorrect: ${repo_root}"
  exit 1
fi
if [[ ! -d "${base_vlm}" ]]; then
  echo "ERROR: base_vlm not found: ${base_vlm}"
  exit 1
fi
if [[ ! -d "${oxe_data_root}" ]]; then
  echo "ERROR: oxe_data_root not found: ${oxe_data_root}"
  exit 1
fi
if [[ -z "${run_root_dir}" || -z "${run_id}" ]]; then
  echo "ERROR: RUN_ROOT_DIR and RUN_ID must be set."
  exit 1
fi

# Create output folder and save a copy of this script for reproducibility
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# Launch training with Hugging Face Accelerate + DeepSpeed
ds_config=${DS_CONFIG:-${repo_root}/src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml}
num_processes=${NUM_PROCESSES:-8}

if [[ ! -f "${ds_config}" ]]; then
  echo "ERROR: DeepSpeed config not found: ${ds_config}"
  exit 1
fi

accelerate launch \
  --config_file "${ds_config}" \
  --num_processes "${num_processes}" \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.model_type ${model_type} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.window_size ${window_size} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  --trainer.shuffle_buffer_size 10000 \
  --trainer.save_interval 10000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 500 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project vla_jiang \
  --wandb_entity zbdz 
