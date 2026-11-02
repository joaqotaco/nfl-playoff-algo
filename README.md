# NFL playoff markets

The purpose of this repo is to explore the following question: **Are Kalshi's NFL Playoff Qualifiers markets priced consistently NFL game markets? If not, can a patient quoter capture the gap post fees?**

## Why this market
- **It's relatively thin.** Trading volume is ~$7k/day across all 32 contracts (vs. ~$321,737,091 total volume in individual game outcomes in the last 7 days, as of 1 Nov 2026).
- **Insiders likely to have small edge.** Prices are arguably mostly mechanical, given that they are informed by individual game outcome probabilities. This is not the case in markets such as coach-firing or draft markets, where insiders certainly have an edge.
- **It has a built-in efficient input.** Game lines are among the most efficiently-priced markets in sports.

## Hypothesis
Edge comes from computation and patience, not information. Playoff prices are derived from game prices, and in a market this thin, few may be performing detailed derivations from game prices.

## Methodology
- 1. Record order books and trades for the 32 playoff markets, the 8 division-winner markets, and upcoming games.
- 2. Back out team ratings from current game lines, and turn a spread into a win probability (see: [Hal Stern's "On the Probability of Winning a Football Game"](https://www.tandfonline.com/doi/abs/10.1080/00031305.1991.10475798)).
- 3. Simulate the rest of the season, including the uncertainty in team ratings, with any official tiebreakers. Validate against past seasons, using nflseedr as reference.
- 4. Produce a daily fair value and an uncertainty band for each team.
- 5. Paper-quote only where a gap is bigger than fees plus the band. A paper fill counts only if the market trades through the quoted price.
- 6. Score each paper fill with markouts at 1 hour, 24 hours, and after the next game day.

## How results are scored
- **Success:** markouts are positive after fees, the sample is large enough to seperate them from noise, and the gaps can be explained meaningfully.
- **Failure:** markets are consistent within fees, or fills get adversely selected.
- **Deadline:** end of regular season (e.g. early Jan 2027).

## (Rough) timeline
- 1. Design model spec
- 2. Build data collector
- 3. Back out team ratings from game lines and turn spreads into win probabilities
- 4. Build season simulation (including tiebreakers), checked against how past seasons actually finished
- 5. Compute daily fair value for each team, paper quotes, and markouts
- 6. Post-mortem writeup

## Status
1 & 2 completed above; starting on 3.

## Traps to avoid
- Overconfident simulation that looks like favorite-longshot edge (see ["The Economics of the Kalshi prediction market"](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market)) when it's actually a modeling error.
- Temporary effects must not carry into future weeks. E.g. a QB out for two weeks should not necessarily change ratings for the entire season.
