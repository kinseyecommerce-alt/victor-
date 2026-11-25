# VRP-STRANGLE-016-WINGED on real option prices (22 Nov 2026)

**Question.** The strangle passed the lab protocol on MODELLED option prices (Black-Scholes, sigma = the previous
INDIA VIX close, one flat volatility for every strike, calendar clock). Its own readiness bar asks for a real credit of
at least 80 % of the model credit. Do real prices support it?

**Method** (`python -m stratlab.strangle_realprice <lab scratch>`). For each of the 152 lab trades
(`dd/vrp/final_run/trades/VRP-STRANGLE-016-WINGED.csv`, 2023-11 .. 2026-11): on the entry day, the real NSE F&O
bhavcopy CLOSE of each of the four legs, against the lab's own model priced at the same moment (the day's close, the
bhavcopy's underlying, the trade's VIX). Same instant, so timing does not bias the comparison. The ratio real / model
net credit is applied to the lab's 10:20 model credit to estimate the real-priced net. The exit (settlement at
intrinsic from the index) is the same in both.

**Result.** 142 of 152 trades priced (10 in 2023 have no bhavcopy file here).

| | value |
|---|---|
| real / model net credit (pooled) | **0.755** (mean 0.743, median 0.700) |
| short legs (sold) real / model | 0.898 |
| wing legs (bought) real / model | **1.728** |
| lab net on the model | +Rs 1,95,065, PF 1.26 |
| real-priced net (pooled ratio) | **-Rs 96,685, PF 0.88** |
| real-priced net (per-trade ratio) | -Rs 1,12,790, PF 0.86 |
| by year (pooled) | 2024 -46,649 · 2025 -3,541 · 2026 -46,496 |

**Why.** The flat-volatility model ignores the volatility skew: far out-of-the-money options (the bought wings, mostly
puts) trade much richer than one flat VIX implies, and the sold strikes a little cheaper. The protection costs far more
than modelled, so the net credit is about a quarter smaller, and the edge disappears.

**Verdict.** Fails its own readiness bar (credit ratio 0.755 < 0.80) and is negative at real prices. Not proven.
The paper agent fills at real Kite quotes, so its paper P&L is already real-priced: expect it to lose. It stays in
paper only as a measurement until the owner decides to stop it. Limits of this check: end-of-day closes (the lab enters
at 10:20); a few deep wings may have stale closes, but every leg counted in the leg ratios traded that day.
