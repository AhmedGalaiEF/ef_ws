#!/bin/bash
set -euo pipefail

IFACE="${IFACE:-wlan0}"
SSID="MESSEacademy"
PSK="technology"
CONF="/etc/wpa_supplicant.conf"

ip link set "$IFACE" up

if ip -4 addr show dev "$IFACE" | grep -q ' inet '; then
    echo "$IFACE already has an IPv4 address"
    exit 0
fi

if ! pgrep -f "wpa_supplicant.*-i ?$IFACE" >/dev/null; then
    umask 077
    wpa_passphrase "$SSID" "$PSK" > "$CONF"
    wpa_supplicant -B -i "$IFACE" -c "$CONF"
else
    wpa_cli -i "$IFACE" reconfigure >/dev/null 2>&1 || true
fi

for _ in $(seq 1 20); do
    if iw dev "$IFACE" link 2>/dev/null | grep -q '^Connected'; then
        break
    fi
    sleep 1
done

if ip -4 addr show dev "$IFACE" | grep -q ' inet '; then
    echo "$IFACE already has an IPv4 address"
    exit 0
fi

dhclient -1 "$IFACE"
