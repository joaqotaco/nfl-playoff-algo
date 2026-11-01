# NFL playoff-qualifier markets: consistency pricing + markouts

## What this project is
Price Kalshi's 32 NFL "make the playoffs" contracts (event KXNFLPLAYOFF-27) from the
efficient game markets (spreads / moneylines), via a season Monte Carlo with the
official NFL tiebreakers. Compare to the thin playoff markets, paper-quote where the
gap exceeds fees + model uncertainty, and score every hypothetical fill with markouts.

- Claimed edge: consistency/computation, NOT better football knowledge or information.
- Decision it ends in: per team, fair value + uncertainty band -> quote or don't, at what price.
- Deadline: end of the 2026 regular season (early Jan 2027). "No edge after fees, and here's
  why" is an acceptable result.
- Artifact: short writeup + this repo. Bridge to credit SMM: fair value from sparse, noisy
  quotes in a thin book, scored by markouts.

## How I want to work with you (important)
- I write the modeling, simulation and tiebreaker code myself. Do not write those for me.
  Review it, explain, point out bugs, ask me questions, and write tests when I ask.
- For the data collector: help with Kalshi API details and robustness, review my code. I write it.
- Push back. If my model solves a slightly different problem than the one I stated, say so.
- Before I compute anything, make me state: what is random, what I condition on (with timestamp),
  and the exact quantity. Ask me to guess the answer first.
- When I'm unsure about a probability claim, suggest a quick Monte Carlo check.
- Point me to human-written sources (papers, official docs), never content farms.

## Phase plan
1. Collector (now): record data every day, starting immediately; lost days can't be recovered.
2. Ratings: back out team ratings from current game lines; spread -> win prob (Stern 1991).
3. Season sim + tiebreakers; validate against past seasons' actual seeding.
4. Daily fair values, paper quotes, markouts (1h / 24h / after next game day). Writeup.

## Collector requirements
- Markets: all 32 KXNFLPLAYOFF contracts; the 8 division-winner events; upcoming game markets.
- Record: order book snapshots (full depth, both sides) and trades, with UTC timestamps.
- Keep raw API responses (append-only), plus a tidy table derived from them.
- Log gaps and errors explicitly; a missing snapshot must be distinguishable from an empty book.
- Must run on an always-on machine (VPS or scheduled job), not a laptop that sleeps.
- Read Kalshi's official API docs for endpoints, auth and rate limits; don't guess.

## Paper-fill and fee rules
- A passive paper order counts as filled only if the market trades THROUGH my price.
- Kalshi taker fee: round_up(0.07 * C * P * (1-P)). Maker fees apply only on some series;
  check Kalshi's fee schedule for these markets.
- Consistency checks use bids/asks, never mids: sum of bids vs 14, sum of asks vs 14,
  7 per conference, P(division) <= P(playoffs).

## Model spec: status
In progress. Open items:
- Settlement: read the KXNFLPLAYOFF rules page (cancellations, forfeits, timing).
- Random: game outcomes (incl. rare ties), team-strength uncertainty (source of correlation),
  injuries. Tiebreakers are deterministic rules, not randomness (except the coin toss).
- Tiebreakers: division, wild-card and 3+-club procedures all differ. Source of truth:
  NFL Football Operations tiebreaking page. Test my implementation against nflseedr
  (nflverse) on past seasons.
- Games with no market line yet (weeks 12-18): ratings fit to current lines; temporary
  effects (QB out 2 weeks, weather, Week 18 rested starters) must not leak into later weeks.
- Avoid double counting: lines already price injuries known at the snapshot time.

## Error log categories (for my own mistakes)
- Answered an adjacent question
- Conflated two concepts
- Simulation doesn't match the model I wrote down
- Used mids where a tradeable price was needed
