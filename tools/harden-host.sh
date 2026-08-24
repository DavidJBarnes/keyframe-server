#!/usr/bin/env bash
# Host guards for a GPU box you cannot physically reach.
#
# Learned the hard way 2026-08-24: two model servers on one 61GB host exhausted
# RAM and the machine went down. It self-recovered, but the previous boot's
# kernel log was gone, so the cause could not be confirmed.
set -uo pipefail

echo "=== 1. persistent journald (so the next incident leaves evidence) ==="
if journalctl --list-boots 2>/dev/null | wc -l | grep -qE '^\s*[01]$'; then
    echo "  journald is volatile — only the current boot is retained"
    echo "  enabling persistent storage:"
    echo "    sudo mkdir -p /var/log/journal"
    echo "    sudo sed -i 's/^#\\?Storage=.*/Storage=persistent/' /etc/systemd/journald.conf"
    echo "    sudo systemctl restart systemd-journald"
else
    echo "  already persistent ($(journalctl --list-boots 2>/dev/null | wc -l) boots retained)"
fi

echo "=== 2. auto-reboot on kernel panic (instead of hanging forever) ==="
CUR=$(sysctl -n kernel.panic 2>/dev/null || echo 0)
if [ "$CUR" = "0" ]; then
    echo "  kernel.panic=0 — a panic hangs indefinitely, needing a physical power cycle"
    echo "  to auto-reboot after 10s:"
    echo "    echo 'kernel.panic=10' | sudo tee /etc/sysctl.d/99-panic.conf"
    echo "    echo 'kernel.panic_on_oops=1' | sudo tee -a /etc/sysctl.d/99-panic.conf"
    echo "    sudo sysctl --system"
else
    echo "  kernel.panic=$CUR (already set to reboot)"
fi

echo "=== 3. earlyoom (kills the hog before the box becomes unresponsive) ==="
if systemctl is-active --quiet earlyoom 2>/dev/null; then
    echo "  earlyoom active"
else
    echo "  not installed. The kernel OOM killer often acts too late, after the"
    echo "  machine has already thrashed into unresponsiveness:"
    echo "    sudo dnf install -y earlyoom && sudo systemctl enable --now earlyoom"
fi

echo "=== 4. current memory headroom ==="
free -g | head -2 | sed 's/^/  /'
