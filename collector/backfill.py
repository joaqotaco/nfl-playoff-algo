"""One-off: fetch playoff and division trades older than the live API's cutoff.

Run once, after the first round: python3 -m collector.backfill
"""
from datetime import UTC, datetime

from collector.fetch import Fetcher, utc_now
from collector.rawlog import RoundLog
from collector.round import PAGE_LIMIT, Session, data_dir, load_state, run_lock


def main():
    base = data_dir()
    state = load_state(base / "state.json")
    tickers = state.get("futures_tickers")
    if not tickers:
        raise SystemExit("No tickers saved yet. Run collector.round first.")

    round_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-backfill"
    with run_lock(base):
        log = RoundLog(base / "raw", round_id)
        session = Session(Fetcher(), log, round_id)
        started_at, count, incomplete = utc_now(), 0, []
        for ticker in tickers:
            trades, complete = session.get_all(
                "trades_historical", "/historical/trades",
                {"ticker": ticker, "limit": PAGE_LIMIT}, "trades", ticker=ticker,
            )
            count += len(trades)
            if not complete:
                incomplete.append(ticker)
        log.write({
            "round_id": round_id,
            "kind": "round_summary",
            "slot": None,
            "started_at": started_at,
            "finished_at": utc_now(),
            "outcome": "degraded" if incomplete else "ok",
            "requests": session.requests,
            "errors": session.errors,
            "trades": count,
            "incomplete": incomplete,
        })
        log.close()
    print(f"{round_id}: {count} trades, {len(incomplete)} tickers incomplete")


if __name__ == "__main__":
    main()
