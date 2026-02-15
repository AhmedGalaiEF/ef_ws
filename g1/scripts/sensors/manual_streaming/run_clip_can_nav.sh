#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAP_PATH="${1:-${ROOT_DIR}/navigation/obstacle_avoidance/maps/live_slam_latest.npz}"

python "${SCRIPT_DIR}/receive_realsense_gst_clip_can.py" \
  --trigger-nav-on-detect \
  --nav-map "${MAP_PATH}" \
  --nav-extra-args "--use-live-map --smooth --allow-diagonal"
