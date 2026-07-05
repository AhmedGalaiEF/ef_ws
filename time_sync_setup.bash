#!/usr/bin/env bash
# time_sync_setup.bash — chrony NTP client setup for the G1 companion computer (PC2)
#
# Syncs the companion computer's clock to G1's built-in time-sync source
# (PC1), reachable at 192.168.123.161 on the robot's internal network.
# Requires G1 firmware > 1.5.1.
#
# Run this ON the Jetson/companion computer (Ubuntu, systemd, apt) — not in
# a Termux/Alpine dev sandbox, which has neither.
#
# Usage:
#   sudo ./time_sync_setup.bash [PC1_IP]
#
# Reference: Unitree G1 SDK Development Guide > Services Interface >
# Time Sync Interface.
set -euo pipefail

PC1_IP="${1:-192.168.123.161}"
CHRONY_CONF="/etc/chrony/chrony.conf"
CHRONY_DEFAULT="/etc/default/chrony"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0 [PC1_IP]" >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1; then
  echo "This script expects systemd + apt (the companion computer's Ubuntu image)." >&2
  echo "systemctl or apt-get was not found on this host." >&2
  exit 1
fi

echo "== Installing chrony =="
if ! command -v chronyd >/dev/null 2>&1; then
  apt-get update
  apt-get install -y chrony
else
  echo "chrony already installed, skipping apt-get install."
fi

echo "== Writing ${CHRONY_CONF} (backing up any existing file once) =="
mkdir -p "$(dirname "${CHRONY_CONF}")"
if [[ -f "${CHRONY_CONF}" && ! -f "${CHRONY_CONF}.orig" ]]; then
  cp "${CHRONY_CONF}" "${CHRONY_CONF}.orig"
  echo "Backed up existing config to ${CHRONY_CONF}.orig"
fi
cat > "${CHRONY_CONF}" <<EOF
# Managed by time_sync_setup.bash — syncs to the G1 PC1 time-sync source.

# Use the G1 internal-network NTP server (PC1).
server ${PC1_IP} iburst prefer

# If the network is not ready at startup, allow subsequent hard-step sync.
makestep 1.0 3

# Allow hardware clock synchronization.
rtcsync

# Logs (useful for debugging).
log tracking measurements statistics
logdir /var/log/chrony
EOF

echo "== Enabling and restarting chrony =="
systemctl enable chrony >/dev/null 2>&1 || true

set +e
systemctl restart chrony
restart_status=$?
status_output="$(systemctl status chrony 2>&1 || true)"
set -e

if [[ "${restart_status}" -ne 0 ]] || grep -qi "core-dump\|signal=SYS" <<<"${status_output}"; then
  echo "== chrony failed to start (seccomp core-dump) — applying DAEMON_OPTS=\"-F 0\" =="
  if [[ -f "${CHRONY_DEFAULT}" ]]; then
    if grep -q '^DAEMON_OPTS=' "${CHRONY_DEFAULT}"; then
      sed -i 's/^DAEMON_OPTS=.*/DAEMON_OPTS="-F 0"/' "${CHRONY_DEFAULT}"
    else
      printf '\nDAEMON_OPTS="-F 0"\n' >> "${CHRONY_DEFAULT}"
    fi
  else
    printf 'DAEMON_OPTS="-F 0"\n' > "${CHRONY_DEFAULT}"
  fi
  systemctl restart chrony
fi

sleep 2
echo
echo "== chrony status =="
systemctl --no-pager status chrony || true
echo
echo "== chrony sources =="
chronyc sources -v || true
echo
echo "== chrony tracking =="
chronyc tracking || true

echo
echo "Done. If 'Leap status' above is not 'Normal' yet, wait a bit and re-check:"
echo "  chronyc tracking"
