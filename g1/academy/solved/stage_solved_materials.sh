#!/usr/bin/env bash
set -euo pipefail

# Stage the SOLVED (full reference-answer) notebooks for a day into one or more
# participant accounts -- unlike stage_later_day_materials.sh, notebook code
# cells are NOT stripped into TODOs. Intended for an instructor/verification
# account (e.g. teilnehmer13), not the whole cohort.
#
# Usage:
#   sudo bash stage_solved_materials.sh DAY [USER ...]
# Examples:
#   sudo bash stage_solved_materials.sh 2                 # -> teilnehmer13 (default)
#   sudo bash stage_solved_materials.sh 2 teilnehmer13
#   sudo bash stage_solved_materials.sh 2 teilnehmer1 teilnehmer13

day="${1:?Usage: sudo bash $0 DAY [USER ...] (DAY = 2, 3, or 4)}"
shift || true
users=("$@")
[[ ${#users[@]} -gt 0 ]] || users=("teilnehmer13")

SOLVED_DIR="$(cd "$(dirname "$0")" && pwd)"
ACADEMY_DIR="$(cd "$SOLVED_DIR/.." && pwd)"
STAGING_DIR="$SOLVED_DIR/staging_day$day"

[[ "$(id -u)" -eq 0 ]] || { echo "Run as root: sudo bash $0 $day ${users[*]}" >&2; exit 1; }
case "$day" in
  2) tasks=(5 6 7) ;;
  3) tasks=(8 9 10 11) ;;
  4) tasks=(12 13) ;;
  *) echo "DAY must be 2, 3, or 4" >&2; exit 2 ;;
esac

# Full solved notebooks (copied verbatim, no stripping) + the day's intro pages.
notebooks=()
for task in "${tasks[@]}"; do
  match=("$SOLVED_DIR"/task"$task"_*.ipynb)
  [[ -f "${match[0]}" ]] || { echo "Missing solved Task $task notebook in $SOLVED_DIR" >&2; exit 1; }
  notebooks+=("${match[0]}")
done
intros=()
for task in "${tasks[@]}"; do
  match=("$STAGING_DIR"/task"$task"_*_intro.html)
  [[ -f "${match[0]}" ]] || { echo "Missing Task $task intro page in $STAGING_DIR" >&2; exit 1; }
  intros+=("${match[0]}")
done
slides="$STAGING_DIR/day${day}_slides.html"
[[ -f "$slides" ]] || { echo "Missing $slides" >&2; exit 1; }

for user in "${users[@]}"; do
  home_dir="$(getent passwd "$user" | cut -d: -f6)"
  [[ -n "$home_dir" ]] || { echo "Missing account: $user" >&2; exit 1; }
  destination="$home_dir/academy/day_$day"
  install -d -m 0755 "$destination"
  rsync -a "${notebooks[@]}" "${intros[@]}" "$slides" \
    "$ACADEMY_DIR/sdk_wrapper.py" "$ACADEMY_DIR/util.py" "$ACADEMY_DIR/slam_util.py" \
    "$destination/"
  chown -R "$user:$user" "$destination"
  echo "Staged SOLVED Day $day materials for $user at $destination"
done
