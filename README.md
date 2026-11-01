# NFL playoff markets

Pricing Kalshi's 32 NFL "make the playoffs" contracts from game markets, then scoring
paper quotes with markouts. This repo currently holds the data collector, which records
the market data that can't be recovered later. See [CLAUDE.md](CLAUDE.md) for the project plan.

## What the collector records

Every 30 minutes, on the hour and half hour UTC:

| Markets | Books | Trades |
|---|---|---|
| 32 playoff contracts (`KXNFLPLAYOFF-27`) | Full depth | All |
| 32 division winner contracts (8 events) | Full depth | All |
| Open `KXNFLGAME` and `KXNFLSPREAD` markets | Full depth in raw, top of book in SQLite | Not polled |

Game market trades are too heavy to poll, and paper fills don't need them. They stay
available from Kalshi's API if they're ever wanted. Order books have no history on
Kalshi, which is why this has to run continuously. It uses public endpoints and
needs no API key or Python packages, only Python 3.11+.

## Where the data goes

- `raw/YYYY/MM/DD/<round>.jsonl.gz`: the permanent record, one file per round. Each line is
  one request: URL, UTC times, status, attempts, errors and the exact response. Failed
  requests get a line too. The final line summarizes the round.
- `tidy.sqlite`: query tables built from raw. It can be deleted and rebuilt at any time.
- `state.json`: which tickers exist and how far each one's trades have been read.

These live in `data/` locally and in `/var/lib/nfl-collector` on the server.

## SQLite tables

| Table | Contents |
|---|---|
| `books` | One row per ticker per round: best bid, derived best ask, sizes, `book_status` |
| `book_levels` | Full depth for playoff and division markets |
| `trades` | Deduplicated on `trade_id`. Exclude `is_block_trade = 1` from paper fills |
| `markets` | Each market's status per round, including when it closes after a clinch |
| `rounds` | One row per round: `ok`, `degraded` or `crashed` |
| `fetches` | Every request and how it went |

`book_status` keeps gaps distinct from quiet markets:

- `ok`: there are orders. A side with no orders has a NULL price.
- `empty`: Kalshi returned no orders on either side.
- `missing`: no book was obtained for this ticker, so it says nothing about the market.

Asks are derived from the opposite side: `yes_ask = 1 - best no bid`. Kalshi's raw levels
are sorted ascending, so the best bid is the last one.

Each row shows the book at `received_at`, and quotes that came and went between rounds are
never seen. Books are fetched in a few batches, so cross-market checks are close to
simultaneous but not exactly.

## Running it

```sh
python3 -m unittest discover -s tests   # tests
python3 -m collector.round              # one round, saved in data/
python3 -m collector.backfill           # one-off: trades older than the live API holds
python3 -m collector.tidy               # update SQLite from raw
python3 -m collector.tidy --status      # recent rounds and any missing slots
```

## Deploying

For a fresh Ubuntu 24.04 server, from your laptop in this repo. This copies the code and any
local data, including the backfill, then starts the timer:

```sh
export SERVER=root@YOUR_SERVER_IP
rsync -a --exclude data --exclude .git --exclude __pycache__ ./ $SERVER:/opt/nfl-playoff-algo/
rsync -a --exclude collector.lock data/ $SERVER:/var/lib/nfl-collector/
ssh $SERVER 'bash /opt/nfl-playoff-algo/deploy/setup.sh && systemctl start nfl-collector'
```

Then fill in `/etc/nfl-collector.env` on the server, using
[deploy/collector.env.example](deploy/collector.env.example), and turn on backups:

```sh
systemctl enable --now nfl-backup.timer
```

To check on it:

```sh
systemctl list-timers 'nfl-*'
journalctl -u nfl-collector -n 20
cd /opt/nfl-playoff-algo && sudo -u collector COLLECTOR_DATA_DIR=/var/lib/nfl-collector python3 -m collector.tidy --status
```

## Monitoring

Each round pings an outside monitor such as [Healthchecks.io](https://healthchecks.io), and
pings its `/fail` URL if the round is degraded. Because the monitor expects the ping, it
also catches a dead server, which nothing on the server itself could report.
