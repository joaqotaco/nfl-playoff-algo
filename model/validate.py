"""Check the tiebreakers against real seasons: seed each one from its regular-season
results, and compare with the playoff field that actually happened.

    python3 -m model.validate

From the playoff games we know each conference's field, the seeds with a bye, the division
winners (wild-card hosts plus byes) and the wild-card pairings. Since 2020 there are 7
teams, one bye and pairings 2v7, 3v6, 4v5. From 2002 (the current divisions) to 2019 there
were 6 teams, two byes and pairings 3v6, 4v5. Matching all of those pins down the seeding.
"""
import csv
import sys

import numpy as np

from collector.round import data_dir
from model import data
from model.league import CONFERENCE_OF, CONFERENCES, INDEX, TEAMS
from model.standings import Season, seed_conference

SEASONS = range(2002, 2026)
# Old nflverse codes for teams that moved.
OLD_CODES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def team(code):
    return OLD_CODES.get(code, code)


def wild_cards(season):
    return 3 if season >= 2020 else 2


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["home_team"], r["away_team"] = team(r["home_team"]), team(r["away_team"])
    return rows


def actual_field(rows, season, conference):
    wc = [r for r in rows if r["season"] == str(season) and r["game_type"] == "WC"
          and CONFERENCE_OF[r["home_team"]] == conference]
    div = [r for r in rows if r["season"] == str(season) and r["game_type"] == "DIV"
           and CONFERENCE_OF[r["home_team"]] == conference]
    hosts = {r["home_team"] for r in wc}
    visitors = {r["away_team"] for r in wc}
    in_div = {r["home_team"] for r in div} | {r["away_team"] for r in div}
    byes = in_div - hosts - visitors
    return {
        "field": hosts | visitors | byes,
        "byes": byes,
        "division_winners": hosts | byes,
        "pairings": {frozenset((r["home_team"], r["away_team"])) for r in wc},
    }


def model_field(seeds, winners):
    names = [TEAMS[t] for t in seeds]
    byes = 1 if len(names) == 7 else 2
    rest = names[byes:]
    return {
        "field": set(names),
        "byes": set(names[:byes]),
        "division_winners": {TEAMS[t] for t in winners},
        "pairings": {frozenset((rest[k], rest[-1 - k])) for k in range(len(rest) // 2)},
    }


def season_from_rows(rows, season, seed=0):
    games = [r for r in rows if r["season"] == str(season) and r["game_type"] == "REG" and r["result"]]
    return Season.from_games(
        [INDEX[r["home_team"]] for r in games], [INDEX[r["away_team"]] for r in games],
        [float(r["home_score"]) for r in games], [float(r["away_score"]) for r in games],
        rng=np.random.default_rng(seed),
    )


def validate(rows, seasons=SEASONS):
    failures = []
    for season in seasons:
        standings = season_from_rows(rows, season)
        for conference in CONFERENCES:
            seeds, winners = seed_conference(standings, conference, wild_cards(season))
            ours, actual = model_field(seeds, winners), actual_field(rows, season, conference)
            wrong = [k for k in actual if ours[k] != actual[k]]
            status = "ok" if not wrong else f"MISMATCH in {', '.join(wrong)}"
            print(f"{season} {conference}: {' '.join(TEAMS[t] for t in seeds)}  {status}")
            if wrong:
                failures.append((season, conference, ours, actual))
    return failures


def main():
    base = data_dir()
    path = data.latest_nflverse(base) or data.fetch_nflverse(base)
    failures = validate(load_rows(path))
    print(f"\n{len(failures)} mismatches")
    for season, conference, ours, actual in failures:
        for key in actual:
            if ours[key] != actual[key]:
                print(f"  {season} {conference} {key}: model {sorted(map(sorted, ours[key])) if key == 'pairings' else ours[key]} "
                      f"actual {sorted(map(sorted, actual[key])) if key == 'pairings' else actual[key]}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
