#!/bin/bash
# Usage: sudo ./install.sh [--with-hardening] [--uninstall]
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
  systemctl disable --now gpu-oom-guard.service 2>/dev/null || true
  rm -f /usr/local/sbin/gpu-oom-guard /etc/systemd/system/gpu-oom-guard.service
  systemctl daemon-reload
  echo "removed (config /etc/gpu-oom-guard.toml and hardening drop-ins left in place)"
  exit 0
fi

command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found in PATH"
install -m 755 gpu_oom_guard.py /usr/local/sbin/gpu-oom-guard
install -m 644 systemd/gpu-oom-guard.service /etc/systemd/system/gpu-oom-guard.service
[ -e /etc/gpu-oom-guard.toml ] || install -m 644 gpu-oom-guard.example.toml /etc/gpu-oom-guard.toml.example

if [ "${1:-}" = "--with-hardening" ]; then
  for u in tailscaled.service ssh.service sshd.service; do
    if systemctl cat "$u" >/dev/null 2>&1; then
      install -D -m 644 hardening/vpn-ssh-resilience.conf "/etc/systemd/system/$u.d/90-resilience.conf"
      echo "hardened $u"
    fi
  done
  if [ -e /dev/watchdog ]; then
    install -D -m 644 hardening/90-watchdog.conf /etc/systemd/system.conf.d/90-watchdog.conf
    systemctl daemon-reexec; echo "hardware watchdog enabled"
  else
    echo "no /dev/watchdog, skipping watchdog"
  fi
  install -m 644 hardening/90-gpu-oom-guard.sysctl.conf /etc/sysctl.d/90-gpu-oom-guard.conf
  sysctl -q -p /etc/sysctl.d/90-gpu-oom-guard.conf
  echo "note: restart hardened services yourself when convenient (e.g. systemctl restart tailscaled)"
fi

systemctl daemon-reload
systemctl enable --now gpu-oom-guard.service
sleep 1
systemctl --no-pager --lines=3 status gpu-oom-guard.service | head -12
echo; /usr/bin/python3 -I /usr/local/sbin/gpu-oom-guard --status
