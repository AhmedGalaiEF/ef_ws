#!/bin/bash
set -euo pipefail

IFACE="${IFACE:-wlan0}"
SSID="MESSEacademy"
PSK="technology"
CONF="/etc/wpa_supplicant.conf"
STATIC_IP="${STATIC_IP:-192.168.1.50}"
PREFIX_LEN="${PREFIX_LEN:-24}"
GATEWAY="${GATEWAY:-192.168.1.1}"
DNS_SERVERS="${DNS_SERVERS:-8.8.8.8 1.1.1.1}"

ip link set "$IFACE" up

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

if ! iw dev "$IFACE" link 2>/dev/null | grep -q '^Connected'; then
    echo "Wi-Fi association failed on $IFACE"
    exit 1
fi

ip addr flush dev "$IFACE"
ip addr add "${STATIC_IP}/${PREFIX_LEN}" dev "$IFACE"
ip route replace default via "$GATEWAY" dev "$IFACE"

if [ -w /etc/resolv.conf ]; then
    : > /etc/resolv.conf
    for dns in $DNS_SERVERS; do
        echo "nameserver $dns" >> /etc/resolv.conf
    done
fi
