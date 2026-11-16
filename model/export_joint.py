"""Per-simulation playoff outcomes at T, for research on sizing across contracts.

    python3 -m model.export_joint --at 2026-11-08T14:00Z --data-dir ~/nfl-copy \\
        --nflverse auto --out ~/Developer/mvo/data/joint

Runs the same setup as model/fairvalue.py at uncertainty x1, x1.5 (central) and x2, keeping
every simulated season, and writes one .npz per scale:

    seed            sims x 32 int8: each team's seed, 0 = missed the playoffs (1-4 = division winner)
    teams           team codes, the column order (model.league.TEAMS)
    tickers         KXNFLPLAYOFF-27 ticker per column; division_tickers likewise
    p_playoffs      share of seasons each team made the playoffs; p_division likewise
    at, kalshi_slot, sims, rng_seed, scale, git_commit
    settings        JSON: every input that determines the run (see settings() below)

Load with np.load(path) (no pickle needed). Reads --data-dir only: never downloads nflverse and
never writes under <data-dir>/fairvalues/. If that day's fair-value CSV is in --data-dir, the
x1.5 run is checked against it: each team's count must be within 3 standard errors of
p_playoffs x sims (the CSV's p), and whether the counts match exactly is reported.

Run it on a copy: `sqlite3 tidy.sqlite ".backup copy.sqlite"` while the collector is idle.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np

from model import data
from model.fairvalue import BAND_FACTORS, setup, scaled_uncertainty
from model.league import TEAMS, division_ticker, playoff_ticker
from model.ratings import DEFAULT_HALF_LIFE
from model.simulate import MEAN_TOTAL, P_TIE, check_seed_matrix, simulate

SCALES = tuple(sorted({1.5, *BAND_FACTORS}))  # x1, x1.5 (central), x2
CHECK_SCALE = 1.5
MAX_Z = 3.0
WILD_CARDS = 3
# The daily job runs at 14:10 UTC and downloads nflverse as it starts.
NFLVERSE_DELAY = timedelta(minutes=10)
REPO = Path(__file__).resolve().parent.parent


def choose_nflverse(data_dir, at, choice, out=print):
    """An explicit path, or 'auto': the first file downloaded at or after T + 10 minutes."""
    if choice != "auto":
        path = Path(choice).expanduser()
        if not path.is_file():
            raise SystemExit(f"--nflverse {path}: no such file")
        return path
    stamped = []
    for path in (data_dir / "nflverse").glob("games-*.csv"):
        try:
            stamped.append((datetime.strptime(path.stem, "games-%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), path))
        except ValueError:
            continue
    after = sorted(p for p in stamped if p[0] >= at + NFLVERSE_DELAY)
    if not after:
        raise SystemExit(f"--nflverse auto: no file in {data_dir / 'nflverse'} downloaded at or after "
                         f"{at + NFLVERSE_DELAY:%Y-%m-%dT%H:%M:%SZ}")
    fetched, path = after[0]
    lag = fetched - at
    out(f"nflverse auto: chose {path.name} (downloaded {lag} after T)")
    if lag > timedelta(hours=1):
        out("  warning: more than an hour after T, so probably not the file the daily job used")
    return path


def check_out_dir(data_dir, out_dir):
    forbidden = (data_dir / "fairvalues").resolve()
    target = out_dir.expanduser().resolve()
    if target == forbidden or forbidden in target.parents:
        raise SystemExit(f"--out {out_dir} is inside {forbidden}; the daily fair values live there")
    return target


def git_commit():
    """(commit, dirty). 'unknown' without git, e.g. on the server, where code arrives by rsync."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "model", "collector"], cwd=REPO,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    """Hash of the model code, which identifies the version even without git."""
    digest = hashlib.sha256()
    for path in sorted((REPO / "model").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def settings(args, at, nflverse, s, scale, uncertainty, commit, dirty):
    return {
        "at": f"{at:%Y-%m-%dT%H:%M:%SZ}", "kalshi_slot": s.slot, "sims": args.sims, "rng_seed": args.seed,
        "scale": scale, "lasting_sd": uncertainty.lasting_sd, "drift_sd": uncertainty.drift_sd,
        "one_off_sd": uncertainty.one_off_sd, "feedback": uncertainty.feedback,
        "sigma": s.sigma, "half_life": args.half_life, "ratings_use_kalshi": args.ratings_use_kalshi,
        "moneylines": args.moneylines, "wild_cards": WILD_CARDS, "p_tie": P_TIE, "mean_total": MEAN_TOTAL,
        "games_left": int(len(s.schedule.home)),
        "games_on_kalshi_lines": int((~np.isnan(s.schedule.market_line)).sum()),
        "games_from_moneylines": sum(k.from_moneyline for k in s.kalshi),
        "home_field": s.ratings.home_field, "ratings": s.ratings.ratings,
        "nflverse_file": nflverse.name, "nflverse_sha256": sha256(nflverse),
        "git_commit": commit, "git_dirty": dirty, "model_code_sha256": code_hash(),
        "numpy": np.__version__, "python": platform.python_version(),
    }


def compare_with_csv(csv_path, result, out=print):
    """True if every team is within MAX_Z standard errors of the CSV. Reports the max z and
    whether the counts are bit-identical."""
    with open(csv_path, newline="") as f:
        rows = {r["team"]: r for r in csv.DictReader(f)}
    n = result.sims
    csv_sims = {int(r["sims"]) for r in rows.values()}
    if csv_sims != {n}:
        out(f"  note: the CSV used {sorted(csv_sims)} sims, this run {n}; the 3-SE check ignores the CSV's own "
            "noise, so it is too strict for runs of different sizes")
    zs, identical = [], csv_sims == {n}
    for i, team in enumerate(TEAMS):
        p = float(rows[team]["p_playoffs"])
        count = result.playoffs[i]
        expected = p * n
        identical &= round(p * int(rows[team]["sims"])) == count
        se = np.sqrt(n * p * (1 - p))
        zs.append(0.0 if count == expected else (abs(count - expected) / se if se > 0 else np.inf))
    worst = int(np.argmax(zs))
    passed = max(zs) <= MAX_Z
    out(f"  vs {csv_path.name}: max z {zs[worst]:.2f} ({TEAMS[worst]}), "
        f"{'PASS' if passed else 'FAIL'} (limit {MAX_Z:g}); bit-identical counts: {'yes' if identical else 'no'}")
    return passed


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--at", type=data.parse_time, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="a copy of /var/lib/nfl-collector")
    parser.add_argument("--nflverse", required=True, help="a games-*.csv path, or 'auto'")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    parser.add_argument("--ratings-exclude-kalshi", dest="ratings_use_kalshi", action="store_false")
    parser.add_argument("--no-moneylines", dest="moneylines", action="store_false")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    if not (data_dir / "tidy.sqlite").is_file():  # sqlite would otherwise create an empty one
        raise SystemExit(f"{data_dir / 'tidy.sqlite'}: no such file")
    out_dir = check_out_dir(data_dir, args.out)
    nflverse = choose_nflverse(data_dir, args.at, args.nflverse, out)
    games = data.load_games(nflverse)
    s = setup(args.at, data_dir, games, args.half_life, args.ratings_use_kalshi, args.moneylines)
    commit, dirty = git_commit()
    out(f"T = {args.at:%Y-%m-%d %H:%M} UTC   Kalshi snapshot {s.slot}   nflverse {nflverse.name}   "
        f"sigma {s.sigma:.2f}   sims {args.sims}   seed {args.seed}   commit {commit[:12]}{' (dirty)' if dirty else ''}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.at:%Y%m%dT%H%MZ}"
    passed = True
    for scale in SCALES:
        uncertainty = scaled_uncertainty(scale)
        result = simulate(s.schedule, s.sigma, uncertainty, args.sims, args.seed, wild_cards=WILD_CARDS,
                          keep_outcomes=True)
        check_seed_matrix(result.seed_matrix, WILD_CARDS)
        path = out_dir / f"{stamp}-x{scale:g}-s{args.sims}-seed{args.seed}.npz"
        np.savez_compressed(
            path, seed=result.seed_matrix, teams=np.array(TEAMS),
            tickers=np.array([playoff_ticker(t) for t in TEAMS]),
            division_tickers=np.array([division_ticker(t) for t in TEAMS]),
            p_playoffs=result.probability(result.playoffs), p_division=result.probability(result.division),
            at=np.array(f"{args.at:%Y-%m-%dT%H:%M:%SZ}"), kalshi_slot=np.array(s.slot),
            sims=np.int64(args.sims), rng_seed=np.int64(args.seed), scale=np.float64(scale),
            git_commit=np.array(commit),
            settings=np.array(json.dumps(settings(args, args.at, nflverse, s, scale, uncertainty, commit, dirty))),
        )
        out(f"x{scale:g}: saved {path}")
        if scale == CHECK_SCALE:
            name = f"{stamp}-calibrated-hl{args.half_life:g}{'-k' if args.ratings_use_kalshi else ''}.csv"
            csv_path = data_dir / "fairvalues" / name
            if csv_path.is_file():
                passed = compare_with_csv(csv_path, result, out)
            else:
                out(f"  no {name} in {data_dir / 'fairvalues'} to check against")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
