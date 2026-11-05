"""Inputs: the nflverse schedule (results and past lines) and Kalshi lines at time T."""
import csv
import re
import sqlite3
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SEASON = "2026"
EASTERN = ZoneInfo("America/New_York")
# Kalshi codes that differ from nflverse's.
KALSHI_TEAM = {"JAC": "JAX", "LAR": "LA"}
# A snapshot older than this isn't "lines as of T".
MAX_SNAPSHOT_AGE = timedelta(hours=1)
# Strikes this close to 0 or 1 say little about the line and distort the probit fit.
MIN_STRIKE_PROB = 0.03


@dataclass(frozen=True)
class Game:
    game_id: str
    week: int
    kickoff: datetime  # UTC
    home: str
    away: str
    neutral: bool
    spread_line: float | None  # nflverse: positive means the home team is favored
    result: float | None  # home score minus away score; None until played
    home_score: float | None = None
    away_score: float | None = None


@dataclass(frozen=True)
class KalshiLine:
    home: str
    away: str
    gameday: str  # ET date, YYYY-MM-DD
    line: float  # median home margin implied by the spread strikes
    sigma: float | None  # spread of a normal fitted to all strikes
    home_win_prob: float | None  # KXNFLGAME mid for the home team


def fetch_nflverse(data_dir):
    """Save a dated copy, so any past fit can be rerun on the file it actually used."""
    folder = Path(data_dir) / "nflverse"
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(NFLVERSE_URL, headers={"User-Agent": "nfl-playoff-algo/0.1"})
    body = urllib.request.urlopen(request, timeout=60).read()
    path = folder / f"games-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.csv"
    path.write_bytes(body)
    return path


def latest_nflverse(data_dir):
    files = sorted((Path(data_dir) / "nflverse").glob("games-*.csv"))
    return files[-1] if files else None


def load_games(path, season=SEASON):
    games = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["season"] != season or row["game_type"] != "REG":
                continue
            local = datetime.strptime(f"{row['gameday']} {row['gametime']}", "%Y-%m-%d %H:%M")
            games.append(Game(
                game_id=row["game_id"],
                week=int(row["week"]),
                kickoff=local.replace(tzinfo=EASTERN).astimezone(UTC),
                home=row["home_team"],
                away=row["away_team"],
                neutral=row["location"] == "Neutral",
                spread_line=float(row["spread_line"]) if row["spread_line"] else None,
                result=float(row["result"]) if row["result"] else None,
                home_score=float(row["home_score"]) if row["home_score"] else None,
                away_score=float(row["away_score"]) if row["away_score"] else None,
            ))
    return games


def median_margin(points):
    """Where P(home margin > x) crosses 50%, interpolating between strikes.

    points: (x, P(margin > x)) pairs. This is the betting line's definition: the
    margin the favorite is as likely to beat as to fall short of.
    """
    points = sorted(points)
    for (x0, p0), (x1, p1) in zip(points, points[1:]):
        if p0 >= 0.5 >= p1 and p0 > p1:
            return x0 + (p0 - 0.5) / (p0 - p1) * (x1 - x0)
    return None


def fitted_sigma(points):
    """Fit P(margin > x) = 1 - Phi((x - mu) / sigma) by regressing probits on x."""
    usable = [(x, NormalDist().inv_cdf(1 - p)) for x, p in points
              if MIN_STRIKE_PROB <= p <= 1 - MIN_STRIKE_PROB]
    if len(usable) < 3:
        return None
    n = len(usable)
    mean_x = sum(x for x, _ in usable) / n
    mean_z = sum(z for _, z in usable) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in usable)
    slope = sum((x - mean_x) * (z - mean_z) for x, z in usable) / sxx
    return 1 / slope if slope > 0 else None


def snapshot_round(db, at):
    """The latest collector round at or before T. Returns (round_id, slot)."""
    row = db.execute(
        "SELECT round_id, slot FROM rounds WHERE slot IS NOT NULL AND slot <= ? "
        "AND outcome != 'crashed' ORDER BY slot DESC LIMIT 1",
        (at.strftime("%Y-%m-%dT%H:%M:%SZ"),),
    ).fetchone()
    if row is None or parse_time(row[1]) < at - MAX_SNAPSHOT_AGE:
        found = f"latest is {row[1]}" if row else "none found"
        raise LookupError(f"no collector round within {MAX_SNAPSHOT_AGE} before {at:%Y-%m-%dT%H:%MZ} ({found})")
    return row


def kalshi_lines(db_path, at, games):
    """Kalshi lines for games kicking off after T, from the order books at T.

    Uses mids, which is fine for estimating a line. Trading comparisons use bids
    and asks instead.
    """
    db = sqlite3.connect(db_path)
    round_id, slot = snapshot_round(db, at)
    rows = db.execute(
        "SELECT b.ticker, b.yes_bid, b.yes_ask, m.event_ticker, m.title "
        "FROM books b JOIN markets m USING (round_id, ticker) "
        "WHERE b.round_id = ? AND (b.ticker LIKE 'KXNFLSPREAD-%' OR b.ticker LIKE 'KXNFLGAME-%')",
        (round_id,),
    ).fetchall()
    db.close()

    by_game = {}
    for ticker, bid, ask, event, title in rows:
        if bid is None or ask is None:
            continue
        match = re.fullmatch(r"KXNFL(?:SPREAD|GAME)-(\d{2}[A-Z]{3}\d{2})([A-Z]+)-([A-Z]+)\d*", ticker)
        if not match:
            continue
        date, pair, team = match.groups()
        entry = by_game.setdefault((date, pair), {"spread": [], "win": {}})
        mid = (bid + ask) / 2
        if ticker.startswith("KXNFLGAME"):
            entry["win"][KALSHI_TEAM.get(team, team)] = mid
        else:
            strike = float(re.search(r"over ([\d.]+) points", title).group(1))
            entry["spread"].append((KALSHI_TEAM.get(team, team), strike, mid))

    upcoming = {(g.kickoff.astimezone(EASTERN).strftime("%y%b%d").upper(), frozenset((g.home, g.away))): g
                for g in games if g.kickoff > at}
    lines = []
    for (date, _pair), entry in by_game.items():
        teams = {team for team, _, _ in entry["spread"]} | set(entry["win"])
        game = upcoming.get((date, frozenset(teams)))
        if game is None:
            continue
        # A home strike k gives P(margin > k); an away strike k gives P(margin < -k).
        points = [(k, p) if team == game.home else (-k, 1 - p) for team, k, p in entry["spread"]]
        line = median_margin(points)
        if line is None:
            continue
        lines.append(KalshiLine(
            home=game.home, away=game.away,
            gameday=game.kickoff.astimezone(EASTERN).strftime("%Y-%m-%d"),
            line=line, sigma=fitted_sigma(points), home_win_prob=entry["win"].get(game.home),
        ))
    return slot, lines


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def default_at(now=None):
    """Today's 14:00 UTC if it has passed, otherwise yesterday's."""
    now = now or datetime.now(UTC)
    today = now.replace(hour=14, minute=0, second=0, microsecond=0)
    return today if now >= today else today - timedelta(days=1)
