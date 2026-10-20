#!/bin/bash
# Run as: sudo bash ~/wg-nano-install.sh
set -e
systemctl stop wg-quick@wg0 2>/dev/null || true
systemctl disable wg-quick@wg0 2>/dev/null || true
pkill -x wireguard-go 2>/dev/null || true; sleep 1
ip link del wg0 2>/dev/null || true
install -m 755 ~kitty/wg-nano-config /usr/local/sbin/wg-nano-config
install -m 644 ~kitty/wg-nano.service /etc/systemd/system/wg-nano.service
systemctl daemon-reload
systemctl enable --now wg-nano
sleep 3
systemctl is-active wg-nano
ip -br addr show wg0
ping -c 3 10.8.0.1
