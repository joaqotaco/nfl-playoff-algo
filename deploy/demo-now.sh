#!/usr/bin/env bash
# Test the trader right now on DEMO, using fair values from the latest collector snapshot
# instead of waiting for the daily 14:00 UTC run. Run as root on the server:
#   bash /opt/nfl-playoff-algo/deploy/demo-now.sh          dry run: plan only
#   bash /opt/nfl-playoff-algo/deploy/demo-now.sh live     place the orders on demo
set -euo pipefail
cd /opt/nfl-playoff-algo

set -a; . /etc/nfl-trader.env; set +a
if [ "${KALSHI_ENV:-}" != "demo" ]; then
    echo "This test script only runs on demo (KALSHI_ENV is '${KALSHI_ENV:-unset}')." >&2
    exit 1
fi

# The latest finished half-hour snapshot.
AT=$(python3 -c "
from datetime import UTC, datetime, timedelta
n = datetime.now(UTC) - timedelta(minutes=5)
print(n.replace(minute=0 if n.minute < 30 else 30, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ'))")
echo "Fair values at $AT"

sudo -u collector COLLECTOR_DATA_DIR=/var/lib/nfl-collector \
    python3 -m model.fairvalue --at "$AT" --refresh --sims 20000 --quiet

FLAGS=""
[ "${1:-}" = "live" ] && FLAGS="--live"
sudo -u trader KALSHI_ENV="$KALSHI_ENV" KALSHI_KEY_ID="$KALSHI_KEY_ID" KALSHI_KEY_FILE="$KALSHI_KEY_FILE" \
    COLLECTOR_DATA_DIR=/var/lib/nfl-collector TRADER_DATA_DIR=/var/lib/nfl-trader \
    python3 -m trader.run --at "$AT" $FLAGS
