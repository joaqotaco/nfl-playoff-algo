#!/usr/bin/env bash
# Run once as root, after setup.sh. Prepares the trader; it does NOT start trading.
set -euo pipefail

APP_DIR=/opt/nfl-playoff-algo

apt-get install -y python3-cryptography

id trader >/dev/null 2>&1 || useradd --system --home-dir /var/lib/nfl-trader --shell /usr/sbin/nologin trader
usermod -aG collector trader                                # reads the fair values
install -d -o trader -g trader -m 755 /var/lib/nfl-trader   # fills, read by the daily report
install -d -o trader -g trader -m 700 /etc/nfl-trader       # the API key lives here

if [ ! -f /etc/nfl-trader.env ]; then
    install -o root -g trader -m 640 "$APP_DIR/deploy/trader.env.example" /etc/nfl-trader.env
fi

install -m 644 "$APP_DIR/deploy/nfl-trader.service" "$APP_DIR/deploy/nfl-trader.timer" /etc/systemd/system/
systemctl daemon-reload

echo "Trader installed but not scheduled. Next: put the key in /etc/nfl-trader/, fill in"
echo "/etc/nfl-trader.env, try a dry run with: systemctl start nfl-trader"
