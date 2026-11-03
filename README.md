# NFL playoff markets

The purpose of this repo is to explore the following question: **Are Kalshi's NFL Playoff Qualifiers markets priced consistently with NFL game markets? If not, can a patient quoter capture the gap post fees?**

## Why this market
- **It's relatively thin.** Trading volume is ~$7k/day across all 32 contracts (vs. ~$321,737,091 total volume in individual game outcomes in the last 7 days, as of 1 Nov 2026).
- **Insiders likely have little edge.** Prices are arguably mostly mechanical, given that they are informed by individual game outcome probabilities. This is not the case in markets such as coach-firing or draft markets, where insiders certainly have an edge.
- **It has a built-in efficient input.** Game lines are among the most efficiently-priced markets in sports.

## Hypothesis
Edge comes from computation and patience, not information. Playoff prices are derived from game prices, and in a market this thin, few may be performing detailed derivations from game prices.

## Model spec

**Quantity.** Every day at T = 14:00UTC, for each team X:
P(X is one of the 7 playoff seeds in its conference | information at T), as implied by game lines. Each value comes with an uncertainty band covering model inputs (rating noise, game-margin spread, home field) and simulation error.

**Sources of randomness**
- Game outcomes given team strengths: the margin is modeled to be normal around the line with SD = 13.86 (Stern 1991). Ties are possible but rare.
- Each team's true strength, drawn once per simulated season. Sharing one draw across a team's games is what makes its results correlated.
- The coin toss, the only random step in tiebreakers.

**What is conditioned on information at T**
- Results of all games final before T, used for standings and tiebreakers only.
- Game lines as of T, used for team strength.
- The remaining schedule.

**Team ratings.** Each game line gives one equation:
line = rating_home - rating_away + home_field (home field is 0 at neutral sites).
- **Solving:** All equations are solved together via weighted least squares, with ratings averaging zero. That gives 33 unknowns (32 ratings plus home field) against >134 lines in 2026.
- **Weights:** weight = 0.5^(age in weeks / half-life). The half-life is chosen by fitting on lines before week k and predicting weeks in k's lines.
- **Upcoming games:** Kalshi lines at T. The lines is where 'KXNFLSPREAD' strikes cross 50%.
- **Past games:** nflverse 'spread_line'.

** Assumptions**
- Team strength is fixed from T to the end of the season (no future injuries)
- nflverse's past lines are roughly closing lines. Its documentation doesn't say.
- Temporary effects in old lines, such as a backup QB, are reduced by the decay but not removed.

## Methodology
- 1. Record order books and trades for the 32 playoff markets, the 8 division-winner markets, and upcoming games.
- 2. Back out team ratings from past and current game lines, and turn a spread into a win probability (see: [Hal Stern's "On the Probability of Winning a Football Game"](https://www.tandfonline.com/doi/abs/10.1080/00031305.1991.10475798)).
- 3. Simulate the rest of the season, including the uncertainty from model inputs and simulation error, with any official tiebreakers. Validate against past seasons, using nflseedr as reference.
- 4. Produce a daily fair value and an uncertainty band for each team.
- 5. Paper-quote only where a gap is bigger than fees plus the band. A paper fill counts only if the market trades through the quoted price.
- 6. Score each paper fill with markouts at 1 hour, 24 hours, and after the next game day.

## How results are scored
- **Success:** markouts are positive after fees, the sample is large enough to separate them from noise, and the gaps can be explained meaningfully.
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
