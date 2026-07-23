#!/usr/bin/env bash

# Configure and start wpa_supplicant for a Jetson Wi-Fi interface.
# Usage: sudo ./wifi.sh [-i interface] [-s ssid] [-H]

set -Eeuo pipefail
umask 077

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_INTERFACE="wlan0"
readonly CONFIG_DIR="/etc/wpa_supplicant"

INTERFACE="$DEFAULT_INTERFACE"
SSID=""
HIDDEN_NETWORK=0

usage() {
    cat <<EOF
Usage: sudo ./$SCRIPT_NAME [options]

Options:
  -i INTERFACE  Wi-Fi interface (default: $DEFAULT_INTERFACE)
  -s SSID       Wi-Fi network name; otherwise you will be prompted
  -H            Mark the network as hidden (adds scan_ssid=1)
  -h            Show this help

The Wi-Fi password is always read interactively and is never stored in this script.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
}

while getopts ':i:s:Hh' option; do
    case "$option" in
        i) INTERFACE="$OPTARG" ;;
        s) SSID="$OPTARG" ;;
        H) HIDDEN_NETWORK=1 ;;
        h)
            usage
            exit 0
            ;;
        :)
            die "option -$OPTARG requires an argument"
            ;;
        \?)
            die "unknown option: -$OPTARG (use -h for help)"
            ;;
    esac
done

[[ $EUID -eq 0 ]] || die "run this script with sudo: sudo ./$SCRIPT_NAME"
[[ "$INTERFACE" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "invalid interface name: $INTERFACE"
[[ -d "/sys/class/net/$INTERFACE" ]] || die "network interface does not exist: $INTERFACE"
command -v wpa_passphrase >/dev/null 2>&1 || die "wpa_passphrase is not installed"
command -v systemctl >/dev/null 2>&1 || die "systemctl is not available"

if [[ -z "$SSID" ]]; then
    read -r -p 'Wi-Fi SSID: ' SSID
fi
[[ -n "$SSID" ]] || die "SSID cannot be empty"
[[ "$SSID" != *$'\n'* && "$SSID" != *$'\r'* ]] || die "SSID cannot contain newlines"

read -r -s -p 'Wi-Fi password: ' PASSWORD
printf '\n'
[[ -n "$PASSWORD" ]] || die "password cannot be empty"

CONFIG_FILE="$CONFIG_DIR/wpa_supplicant-${INTERFACE}.conf"
TEMP_FILE=''

cleanup() {
    if [[ -n "$TEMP_FILE" && -e "$TEMP_FILE" ]]; then
        rm -f -- "$TEMP_FILE"
    fi
    unset PASSWORD
}
trap cleanup EXIT

install -d -m 0755 "$CONFIG_DIR"
TEMP_FILE="$(mktemp "$CONFIG_FILE.tmp.XXXXXX")"
chmod 600 "$TEMP_FILE"

# wpa_passphrase reads the password from stdin. Remove the plaintext psk line
# so the configuration retains only the derived key.
if ! wpa_passphrase "$SSID" <<<"$PASSWORD" |
    sed '/^[[:space:]]*psk="/d' >"$TEMP_FILE"; then
    die "could not generate the wpa_supplicant configuration"
fi

unset PASSWORD

if [[ "$HIDDEN_NETWORK" -eq 1 ]]; then
    sed -i '/^[[:space:]]*}/i\	scan_ssid=1' "$TEMP_FILE"
fi

if [[ -f "$CONFIG_FILE" ]]; then
    BACKUP_FILE="${CONFIG_FILE}.backup.$(date +%Y%m%d-%H%M%S)"
    cp -a -- "$CONFIG_FILE" "$BACKUP_FILE"
    printf 'Backed up existing configuration to %s\n' "$BACKUP_FILE"
fi

mv -- "$TEMP_FILE" "$CONFIG_FILE"
TEMP_FILE=''
chmod 600 "$CONFIG_FILE"

SERVICE="wpa_supplicant@${INTERFACE}.service"
if ! systemctl cat "$SERVICE" >/dev/null 2>&1; then
    die "$SERVICE is not installed on this Jetson"
fi

if command -v rfkill >/dev/null 2>&1; then
    rfkill unblock wifi || warn "could not unblock Wi-Fi with rfkill"
fi

systemctl enable --now "$SERVICE"

printf 'Wi-Fi configuration installed: %s\n' "$CONFIG_FILE"
if systemctl is-active --quiet "$SERVICE"; then
    printf '%s is active.\n' "$SERVICE"
else
    warn "$SERVICE is not active; inspect it with: journalctl -u $SERVICE -n 50 --no-pager"
fi

if command -v iw >/dev/null 2>&1; then
    iw dev "$INTERFACE" link || true
else
    printf 'Check the connection with: ip addr show %s\n' "$INTERFACE"
fi
