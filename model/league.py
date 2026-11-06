"""Teams, divisions and conferences, using nflverse team codes."""

DIVISIONS = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"),
    "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"),
    "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"),
    "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"),
    "NFC West": ("ARI", "LA", "SEA", "SF"),
}
TEAMS = tuple(sorted(team for teams in DIVISIONS.values() for team in teams))
INDEX = {team: i for i, team in enumerate(TEAMS)}
DIVISION_OF = {team: division for division, teams in DIVISIONS.items() for team in teams}
CONFERENCE_OF = {team: division.split()[0] for team, division in DIVISION_OF.items()}
CONFERENCES = ("AFC", "NFC")

# Kalshi uses its own codes for two teams.
TO_KALSHI = {"JAX": "JAC", "LA": "LAR"}


def kalshi_code(team):
    return TO_KALSHI.get(team, team)


def playoff_ticker(team):
    return f"KXNFLPLAYOFF-27-{kalshi_code(team)}"


def division_ticker(team):
    conference, region = DIVISION_OF[team].split()
    return f"KXNFL{conference}{region.upper()}-27-{kalshi_code(team)}"
