#!/usr/bin/env bash
# Run once as root on a fresh Ubuntu 24.04 server, from the copied repo.
# Safe to re-run after copying new code; it won't touch your env file or data.
set -euo pipefail

APP_DIR=/opt/nfl-playoff-algo
DATA_DIR=/var/lib/nfl-collector
ENV_FILE=/etc/nfl-collector.env

apt-get update
apt-get install -y python3 python3-numpy rclone curl sqlite3

id collector >/dev/null 2>&1 || useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin collector
install -d -m 750 "$DATA_DIR"
chown -R collector:collector "$DATA_DIR"

if [ ! -f "$ENV_FILE" ]; then
    install -o root -g collector -m 640 "$APP_DIR/deploy/collector.env.example" "$ENV_FILE"
fi

install -m 644 "$APP_DIR"/deploy/nfl-*.service "$APP_DIR"/deploy/nfl-*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now nfl-collector.timer nfl-fairvalue.timer

echo "Collector and fair-value timers are on."
echo "Then fill in $ENV_FILE and run: systemctl enable --now nfl-backup.timer"
