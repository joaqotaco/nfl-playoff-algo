"""A small synthetic season for end-to-end tests of fair values and joint exports.

Ten weeks: weeks 1-8 are played (lines and scores), weeks 9-10 are left. At T (14:00 UTC on
week 9's Sunday) one collector round holds Kalshi spread strikes for every week-9 game and a
playoff and division book for every team. Everything is plain arithmetic, no randomness.
"""
import csv
import sqlite3
from datetime import UTC, date, datetime, timedelta
from statistics import NormalDist

from collector import tidy
from model.league import TEAMS, division_ticker, playoff_ticker

GAMES_HEADER = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
                "location", "spread_line", "result", "home_score", "away_score"]
FIRST_SUNDAY = date(2026, 9, 13)
WEEKS, PLAYED_WEEKS = 10, 8
SIGMA = 11.0
STRIKES = (1.5, 4.5, 7.5, 10.5)
AT = datetime.combine(FIRST_SUNDAY + timedelta(weeks=PLAYED_WEEKS), datetime.min.time(), UTC).replace(hour=14)
SLOT = AT.strftime("%Y-%m-%dT%H:%M:%SZ")
NFLVERSE_NAME = f"games-{AT + timedelta(minutes=10, seconds=4):%Y%m%dT%H%M%SZ}.csv"


def strength(i):
    return (i * 7) % 13 - 6.0


def pairings(week):
    """Circle method: every team plays once a week, no pair repeats within 31 weeks."""
    n = len(TEAMS)
    order = [0] + [1 + (k + week) % (n - 1) for k in range(n - 1)]
    return [(order[k], order[n - 1 - k]) for k in range(n // 2)]


def games():
    rows = []
    for week in range(1, WEEKS + 1):
        day = FIRST_SUNDAY + timedelta(weeks=week - 1)
        for k, (a, b) in enumerate(pairings(week)):
            home, away = (a, b) if (week + k) % 2 else (b, a)
            line = strength(home) - strength(away) + 1.5
            played = week <= PLAYED_WEEKS
            result = round(line + ((home * 5 + away * 3 + week) % 21 - 10)) or 1
            rows.append({
                "game_id": f"2026_{week:02d}_{TEAMS[away]}_{TEAMS[home]}", "season": "2026", "game_type": "REG",
                "week": week, "gameday": day.isoformat(), "gametime": "13:00",
                "away_team": TEAMS[away], "home_team": TEAMS[home], "location": "Home",
                "spread_line": line if played else "", "result": result if played else "",
                "home_score": (20 + max(result, 0)) if played else "",
                "away_score": (20 + max(-result, 0)) if played else "",
            })
    return rows


def build(base):
    """Write nflverse/<file>, tidy.sqlite and an empty fairvalues/ under base. Returns the nflverse path."""
    (base / "nflverse").mkdir(parents=True)
    path = base / "nflverse" / NFLVERSE_NAME
    rows = games()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAMES_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    db = sqlite3.connect(base / "tidy.sqlite")
    db.executescript(tidy.SCHEMA)
    rid = AT.strftime("%Y%m%dT%H%M%SZ")
    db.execute("INSERT INTO rounds VALUES (?,?,?,?,?,?,?,?)", (rid, SLOT, SLOT, SLOT, "ok", 1, 0, 0))

    def book(ticker, title, bid, ask, event):
        db.execute("INSERT INTO books VALUES (?,?,?,?,?,?,?,?,?,?)", (rid, ticker, SLOT, "ok", 1, 1, bid, 5, ask, 5))
        db.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?)", (rid, ticker, event, title, "active", "", "", 0, 0))

    day = (FIRST_SUNDAY + timedelta(weeks=PLAYED_WEEKS)).strftime("%y%b%d").upper()
    for g in (r for r in rows if r["week"] == PLAYED_WEEKS + 1):
        home, away = g["home_team"], g["away_team"]
        line = strength(TEAMS.index(home)) - strength(TEAMS.index(away)) + 1.5 + 0.5
        event = f"KXNFLSPREAD-{day}{away}{home}"
        for k in STRIKES:
            for team, p in ((home, 1 - NormalDist(line, SIGMA).cdf(k)), (away, NormalDist(line, SIGMA).cdf(-k))):
                book(f"{event}-{team}{int(k)}", f"{team} wins by over {k} points", p - 0.01, p + 0.01, event)
    for i, team in enumerate(TEAMS):
        p = 0.1 + 0.8 * (strength(i) + 6) / 12
        book(playoff_ticker(team), team, round(p - 0.02, 2), round(p + 0.02, 2), "KXNFLPLAYOFF-27")
        book(division_ticker(team), team, round(p / 3, 2), round(p / 3 + 0.03, 2), "division")
    db.commit()
    db.close()
    (base / "fairvalues").mkdir()
    return path
