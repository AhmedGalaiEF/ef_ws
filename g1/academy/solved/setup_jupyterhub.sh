#!/usr/bin/env bash
set -euo pipefail

# setup_jupyterhub.sh — install and configure JupyterHub (via "The Littlest
# JupyterHub", TLJH) on the headless Jetson so every teilnehmerN account
# gets its own browser-based JupyterLab, with the kernel actually running
# on the Jetson under that student's own Linux account and cloned venv.
#
# Architecture:
#   - Auth:    PAMAuthenticator against the existing teilnehmerN accounts
#              (the ones reset_users.sh creates), password academy2026.
#   - Serving: one shared JupyterLab install (TLJH's /opt/tljh/user env)
#              handles the web UI for everyone — that part does NOT
#              multiply per student.
#   - Kernels: SystemdSpawner starts each student's single-user server (and
#              their notebook kernels) as a transient systemd scope running
#              as *their* Linux user, using the unitree_sdk2 kernel already
#              registered against their own cloned venv (see
#              reset_users.sh) — so only the kernel processes multiply,
#              each cgroup-limited and idle-culled below.
#   - Robot access: intentionally NOT arbitrated here. Multiple students'
#     kernels can send actuation commands to the same physical robot at
#     the same time; coordinate whose turn it is verbally in class.
#
# Usage: sudo ./setup_jupyterhub.sh
#
# Re-running is safe: the TLJH bootstrap upgrades an existing install, and
# every tljh-config call below is idempotent.

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must be run as root (e.g. sudo $0)." >&2
  exit 1
fi

NUM_USERS="${NUM_USERS:-13}"
USER_PREFIX="${USER_PREFIX:-teilnehmer}"
ADMIN_USER="${ADMIN_USER:-unitree}"
MEM_LIMIT="${MEM_LIMIT:-1.5G}"      # per-student server memory cap
CPU_LIMIT="${CPU_LIMIT:-1}"         # per-student server CPU core cap
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}" # seconds of inactivity before culling (30 min)
CULL_EVERY="${CULL_EVERY:-300}"     # how often the culler checks (5 min)

USERS=()
for i in $(seq 1 "$NUM_USERS"); do
  USERS+=("${USER_PREFIX}${i}")
done

# --- 1. Install (or upgrade) TLJH ---------------------------------------
if ! command -v tljh-config >/dev/null 2>&1; then
  echo "Installing The Littlest JupyterHub..."
  curl -fsSL https://raw.githubusercontent.com/jupyterhub/the-littlest-jupyterhub/main/bootstrap/bootstrap.py \
    | python3 - --admin "$ADMIN_USER"
else
  echo "TLJH already installed; reconfiguring."
fi

# --- 2. Authentication: PAM against existing Linux accounts -------------
# TLJH defaults to PAMAuthenticator, which is what we want: it checks
# system accounts directly, so teilnehmerN / academy2026 (from
# reset_users.sh) works with no separate hub user database. Only accounts
# explicitly allow-listed here may log in.
tljh-config add-item users.admin "$ADMIN_USER"
for u in "${USERS[@]}"; do
  if id "$u" &>/dev/null; then
    tljh-config add-item users.allowed "$u"
  else
    echo "warning: $u does not exist yet (run reset_users.sh first); not allow-listing." >&2
  fi
done

# --- 3. Per-student resource limits (SystemdSpawner + cgroups) ----------
tljh-config set limits.memory "$MEM_LIMIT"
tljh-config set limits.cpu "$CPU_LIMIT"

# --- 4. Idle culling: stop servers/kernels nobody is using --------------
tljh-config set services.cull.enabled true
tljh-config set services.cull.timeout "$IDLE_TIMEOUT"
tljh-config set services.cull.every "$CULL_EVERY"
tljh-config set services.cull.users true   # also stop the idle single-user server, not just its kernels

tljh-config reload

cat <<EOF

Done. JupyterHub is up at: https://<jetson-ip>/  (self-signed cert by default)
Admin: $ADMIN_USER
Allowed students: ${USERS[*]}
Per-student limits: ${MEM_LIMIT} RAM, ${CPU_LIMIT} CPU core(s)
Idle culling: after ${IDLE_TIMEOUT}s inactivity, checked every ${CULL_EVERY}s

Each student's own "unitree_sdk2" kernel (registered by reset_users.sh
against their cloned venv) will show up automatically in their kernel
picker — no extra kernel configuration needed here.

No robot-access arbitration is configured: multiple students' kernels can
send commands to the physical robot concurrently. Coordinate turns
verbally, or revisit this if it becomes a problem in practice.
EOF
