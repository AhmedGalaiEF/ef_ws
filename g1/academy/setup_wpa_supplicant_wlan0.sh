#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./setup_wpa_supplicant_wlan0.sh SSID PASSPHRASE
SSID="$1"
PASSPHRASE="$2"
COUNTRY="${WPA_COUNTRY:-DE}"
CONF="/etc/wpa_supplicant/wpa_supplicant-wlan0.conf"

install -d -m 0755 /etc/wpa_supplicant
wpa_passphrase "$SSID" "$PASSPHRASE" | sed "1i country=$COUNTRY\nctrl_interface=DIR=/run/wpa_supplicant GROUP=netdev\nupdate_config=1" > "$CONF"
chmod 600 "$CONF"
systemctl enable --now wpa_supplicant@wlan0.service
systemctl status --no-pager wpa_supplicant@wlan0.service
