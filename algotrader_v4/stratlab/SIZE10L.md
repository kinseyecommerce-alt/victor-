# Every lab strategy at Rs 10 lakh of capital per agent

Pre-registered before any result at the new size: `stratlab/PREREGISTERED_SIZE10L.json` (sha256 `02f654e6975ad1b18384b16e04354e2b5cf26b6666547840a8b84519ff31ddd4`) and, before any option result, `stratlab/PREREGISTERED_SIZE10L_AMENDMENT1.json` (sha256 `af07d1c43f1723afde2b552d02502f700196c09edbe093e31cb631c4029e7d56`), which supersedes it and adds the options strategies. Code: `stratlab/size10l_run.py`, `stratlab/size10l_options.py`. Research only; no live rule or verdict is changed.

## Result

- **Proven at the Rs 10 lakh size: 0**.
- Re-tested as new trials: **K = 435** strategies (299 intraday / scalping, 97 stock futures, 39 options). The luck correction counts all of them: N = 5434.
- 7 strategies pass DISCOVERY and are confirmed on the FINAL year at the new size (3 stock futures, 4 options). None of them clears the remaining gates (below).
- No intraday or scalping strategy passes DISCOVERY at Rs 5 lakh per trade either.
- The live rules keep their verdicts and do better per rupee at the new size (report only): AUC-FADE-2PCT's FINAL year goes from PF 1.065 (net Rs 19,341 at Rs 1 lakh) to PF 1.136 (net Rs 197,545 at Rs 5 lakh).

## All types

| Type | Strategies | At Rs 10 lakh per agent | DISCOVERY pass: before -> now | FINAL confirmed | Old-era OOS confirmed | Proven (all gates incl. DSR >= 0.95) |
|---|---|---|---|---|---|---|
| Intraday + scalping (MIS) | 299 | re-tested at Rs 5 lakh per trade | 0 -> 0 | 0 | 0 | 0 |
| Stocks at futures costs | 97 | re-tested at Rs 5 lakh per trade | 7 -> 9 | 3 | 0 | 0 |
| Options | 39 | re-tested, lots from a Rs 1 lakh premium / risk or Rs 5 lakh margin budget | 12 -> 11 | 4 | not available at the new size | 0 |
| Delivery swing (CNC) | 107 | identical at Rs 10 lakh - existing result stands | 8 | 3 | 1 | 0 |
| Index futures | 105 | identical (one real lot) - existing result stands | 0 | 0 | 0 | 0 |
| Timeframe grid (tfgrid) | 668 (+ 608 catalogue rules with the 1:3 bracket) | already traded from Rs 10 lakh accounts - existing result stands | 7 | see tfgrid report | 1 | 0 |

Why three types are not re-run: delivery fees are a fixed percentage of the trade (no flat brokerage), so the profit factor is the same at any size and a Rs 10 lakh account simply holds 10 positions of Rs 1 lakh; an index future is already one real lot, where the flat brokerage is under 0.3 bps; tfgrid sized every trade from a Rs 10 lakh account (median fee per intraday trade Rs 465 = trades of Rs 10-25 lakh).

## Fees: what the size changes

| Product | Round trip at Rs 1 lakh | per Rs 1 lakh at Rs 5 lakh | saving per Rs 1 lakh traded |
|---|---|---|---|
| Intraday (MIS) | Rs 82.68 (8.27 bps) | Rs 44.92 (4.49 bps) | Rs 37.76 |
| Stock at futures costs | Rs 103.75 (10.38 bps) | Rs 65.99 (6.60 bps) | Rs 37.76 |

- Median intraday / scalping strategy (299 with trades), per Rs 1 lakh traded: gross after slippage Rs -24; fees Rs 82 at Rs 1 lakh and Rs 45 at Rs 5 lakh. Net positive in both halves of DISCOVERY: 1 -> 2 of 299.
- Median stock-futures strategy (97 with trades), per Rs 1 lakh traded: gross after slippage Rs -100; fees Rs 103 at Rs 1 lakh and Rs 66 at Rs 5 lakh. Net positive in both halves of DISCOVERY: 13 -> 13 of 97.
- Options: the median strategy's fee per lot falls from Rs 112 to Rs 45 (approximate: fees per trade divided by the median lots). Net positive in both halves: 13 -> 12 of 39.
- Slippage (2 bps a side on stocks, the option fill haircut per unit) does not shrink with size and is unchanged.

## Stock strategies that pass DISCOVERY at Rs 5 lakh per trade

| Strategy | DISCOVERY trades / PF / net | at Rs 1 lakh: passed, PF | FINAL trades / PF / net | FINAL | 2016-23 OOS at the new size | DSR | Net per year |
|---|---|---|---|---|---|---|---|
| `cs_gravestone_doji_d` | 84 / 1.470 / Rs 190,568 | yes, 1.411 | 32 / 1.158 / Rs 26,054 | CONFIRMED | OOS-FAILED | 0.0000 | Rs 72,207 |
| `r3_futures_stk_tom_long` | 2417 / 1.666 / Rs 8,975,924 | yes, 1.625 | 775 / 0.713 / Rs -2,161,776 | rejected | not run (FINAL rejected) | 0.0000 | Rs 2,271,383 |
| `r3_tfdaily_bb_upper_short_downtrend` | 172 / 1.683 / Rs 651,862 | yes, 1.641 | 153 / 0.795 / Rs -256,550 | rejected | not run (FINAL rejected) | 0.0000 | Rs 131,771 |
| `r4_delivery_brk_fake_short` | 827 / 1.313 / Rs 1,744,356 | yes, 1.291 | 324 / 1.117 / Rs 246,323 | CONFIRMED | OOS-FAILED | 0.0000 | Rs 663,560 |
| `r4_delivery_lowdel_rally_short` | 172 / 1.464 / Rs 590,140 | yes, 1.455 | 110 / 0.530 / Rs -503,778 | rejected | not run (FINAL rejected) | 0.0000 | Rs 28,787 |
| `r5_kite_vars_bb20_d` | 2696 / 1.240 / Rs 6,277,943 | yes, 1.221 | 1243 / 0.902 / Rs -1,281,818 | rejected | not run (FINAL rejected) | 0.0000 | Rs 1,665,375 |
| `r5_kite_vars_cs_bull_engulfing` | 704 / 1.253 / Rs 1,397,463 | yes, 1.228 | 266 / 1.052 / Rs 116,577 | CONFIRMED | OOS-FAILED | 0.0000 | Rs 504,680 |
| `r5_kite_vars_cs_bull_marubozu_nxo` | 577 / 1.175 / Rs 703,699 | no, 1.147 | 284 / 0.942 / Rs -125,637 | rejected | not run (FINAL rejected) | 0.0000 | Rs 192,687 |
| `r7_gh2_tsp1_stock` | 1090 / 1.154 / Rs 985,287 | no, 1.121 | 577 / 0.977 / Rs -90,783 | rejected | not run (FINAL rejected) | 0.0000 | Rs 298,168 |

All of them are stock rules booked at futures costs and held for days. Net per year = (DISCOVERY + FINAL net) / 3 at Rs 5 lakh per trade; the number of positions open together is not capped by the account in this test, so it is not a return on Rs 10 lakh (tfgrid is the account-level test).

- `cs_gravestone_doji_d`: Gravestone doji after an up-trend on daily candles: enter next open, stop beyond the pattern extreme (>= 0.3 ATR), 2R, max 10 sessions
- `r3_futures_stk_tom_long`: First session with day-of-month >= 25: buy the stock future at 15:20, 5 % stop, hold 5 sessions
- `r3_tfdaily_bb_upper_short_downtrend`: Close above the upper Bollinger band (20, 2 sd) and close < SMA100: short stock future, stop 2.5 ATR, hold 5
- `r4_delivery_brk_fake_short`: Close > prior 50-session high close with deliv_pct <= 25th pct of own prior 60, not banned: short, 5 % stop, hold 5
- `r4_delivery_lowdel_rally_short`: 5d return >= +8 % with 5d mean deliv_pct < 20th pct of prior 60, not banned: short future, 8 % stop, hold 5
- `r5_kite_vars_bb20_d`: Varsity: daily low touches the lower BB(20,2) -> long, high touches the upper -> short (futures); exit at the 20-day SMA
- `r5_kite_vars_cs_bull_engulfing`: Varsity bullish engulfing after a downtrend, stop at the P1-P2 low; checked at 15:20, filled at the 15:20 open, target 1.5R (stock futures)
- `r5_kite_vars_cs_bull_marubozu_nxo`: 2012-23 daily-data twin of the Varsity bull_marubozu: same rule on the full daily candle, filled at the NEXT open
- `r7_gh2_tsp1_stock`: Turtle Soup +1 (stock futures): day 1 new 20d low closing at/below the prior low (>= 3 sessions old) -> day-2 buy stop at it; stop under the lower low; 5 sessions (mirror)

### The FINAL-confirmed stock strategies

- `cs_gravestone_doji_d`. FINAL: 32 trades, PF 1.158, net Rs 26,054 (the same trades at Rs 1 lakh: PF 1.117, net Rs 3,925). 2016-23 minute at the new size: **OOS-FAILED** - H1 net <= 0, H2 net <= 0, PF 0.948 < 1.15 (era: 294 trades, PF 0.948, net Rs -101,323; at Rs 1 lakh it was OOS-FAILED). 2012-23 daily (report): OOS-FAILED - H1 net <= 0, PF 1.074 < 1.15. Sharpe 0.84 a year over 2023-26, DSR 0.0000 (needs >= 0.95).
- `r4_delivery_brk_fake_short`. FINAL: 324 trades, PF 1.117, net Rs 246,323 (the same trades at Rs 1 lakh: PF 1.089, net Rs 37,277). 2016-23 minute at the new size: **OOS-FAILED** - H2 net <= 0, PF 0.986 < 1.15 (era: 3301 trades, PF 0.986, net Rs -373,769; at Rs 1 lakh it was OOS-FAILED). 2012-23 daily (report): OOS-FAILED - PF 1.069 < 1.15. Sharpe 1.42 a year over 2023-26, DSR 0.0000 (needs >= 0.95).
- `r5_kite_vars_cs_bull_engulfing`. FINAL: 266 trades, PF 1.052, net Rs 116,577 (the same trades at Rs 1 lakh: PF 1.024, net Rs 10,845). 2016-23 minute at the new size: **OOS-FAILED** - H1 net <= 0, PF 1.068 < 1.15 (era: 2038 trades, PF 1.068, net Rs 1,448,600; at Rs 1 lakh it was OOS-FAILED). Sharpe 0.99 a year over 2023-26, DSR 0.0000 (needs >= 0.95).

## Options strategies at the new lots

| Strategy | Lot rule (median lots) | DISCOVERY trades / PF / net | one lot: passed, PF, net | FINAL PF / net | FINAL | DSR | One-lot old-era verdict |
|---|---|---|---|---|---|---|---|
| `VRP-STRANGLE-016-NAKED` | margin (2) | 107 / 1.484 / Rs 276,288 | yes, 1.473, Rs 135,300 | 1.268 / Rs 75,170 | CONFIRMED | 0.0000 | OOS-CONFIRMED (ledger) |
| `r3_options_ss16_stop2x` | margin (2) | 107 / 1.343 / Rs 180,765 | yes, 1.328, Rs 86,972 | 0.870 / Rs -33,782 | rejected | 0.0000 | OOS-CONFIRMED (ledger) |
| `r3_options_ss16_stop2x_tp50` | margin (2) | 107 / 1.461 / Rs 171,787 | yes, 1.431, Rs 80,843 | 0.905 / Rs -19,478 | rejected | 0.0000 | OOS-CONFIRMED (ledger) |
| `r3_options_ss16_stop2x_vix33` | margin (2) | 73 / 1.975 / Rs 267,827 | yes, 1.951, Rs 131,660 | 0.834 / Rs -37,887 | rejected | 0.0000 | OOS-CONFIRMED (ledger) |
| `r3_options_wc_fri` | max_risk (3) | 99 / 1.730 / Rs 425,731 | yes, 1.742, Rs 135,006 | 3.168 / Rs 364,376 | CONFIRMED | 0.0000 | OOS-CONFIRMED (ledger) |
| `r4_realopt_bnf_mo_oiwall_beyond_h5` | margin (2) | 81 / 1.219 / Rs 146,090 | yes, 1.207, Rs 69,222 | 0.799 / Rs -77,519 | rejected | 0.0000 | OOS-FAILED (oldera_oos_ledger) |
| `r4_realopt_nf_exp_eve_strangle_1pct` | margin (2) | 108 / 1.251 / Rs 105,371 | yes, 1.236, Rs 49,782 | 1.045 / Rs 13,998 | rejected | 0.0000 | OOS-FAILED (oldera_oos_ledger) |
| `r4_realopt_nf_expwk_last2_strangle_1pct` | margin (2) | 108 / 1.397 / Rs 276,378 | yes, 1.387, Rs 135,086 | 1.656 / Rs 191,507 | CONFIRMED | 0.0000 | OOS-FAILED (oldera_oos_ledger) |
| `r4_realopt_nf_wk_oiwall_at` | margin (2) | 104 / 2.491 / Rs 354,716 | yes, 2.464, Rs 174,727 | 0.775 / Rs -78,490 | rejected | 0.0000 | none on record |
| `r4_realopt_nf_wk_oiwall_beyond` | margin (2) | 84 / 3.465 / Rs 228,084 | yes, 3.413, Rs 111,977 | 0.853 / Rs -34,453 | rejected | 0.0000 | none on record |
| `r4_realopt_nf_wk_strangle_3pct` | margin (2) | 104 / 1.968 / Rs 198,904 | yes, 1.940, Rs 96,879 | 2.481 / Rs 156,239 | CONFIRMED | 0.0000 | none on record |
| `r3_options_ic_25_10` | max_risk (7) | 107 / 1.268 / Rs 566,099 | yes, 1.290, Rs 89,480 | - / Rs - | not opened (failed DISCOVERY) | - | OOS-CONFIRMED (ledger) |

Passed DISCOVERY at one lot but not at the new lots: `r3_options_ic_25_10` (H1 net <= 0): the lot count now varies from trade to trade, which re-weights the weeks.

An option strategy cannot be proven by this re-test: no old-era option record carries per-leg prices, so its out-of-sample years cannot be re-sized (pre-registered). The four FINAL-confirmed option strategies were already confirmed at one lot; all four sell option premium, and their DSR is far below 0.95.

## The live rules at the new size (report only; no verdict is re-opened)

| Rule | Size | DISCOVERY net / PF | FINAL net / PF | FINAL gross and fee per trade | 3-year net | Sharpe a year |
|---|---|---|---|---|---|---|
| AUC-FADE-2PCT | Rs 1 lakh | Rs 265,186 / 1.449 | Rs 19,341 / 1.065 | Rs 118 and Rs 82 | Rs 284,526 | 1.59 |
| AUC-FADE-2PCT | Rs 5 lakh | Rs 1,545,583 / 1.535 | Rs 197,545 / 1.136 | Rs 594 and Rs 224 | Rs 1,743,128 | 1.92 |
| AUC-FADE-3PCT | Rs 1 lakh | Rs 239,796 / 1.910 | Rs 48,412 / 1.316 | Rs 272 and Rs 82 | Rs 288,208 | 2.03 |
| AUC-FADE-3PCT | Rs 5 lakh | Rs 1,307,002 / 2.013 | Rs 293,641 / 1.391 | Rs 1,376 and Rs 224 | Rs 1,600,643 | 2.21 |
| SWING-ATTN-ON-3X | Rs 1 lakh | Rs 157,888 / 2.126 | Rs 10,615 / 1.172 | Rs 158 and Rs 103 | Rs 168,504 | 2.09 |
| SWING-ATTN-ON-3X | Rs 5 lakh | Rs 900,914 / 2.367 | Rs 93,244 / 1.318 | Rs 810 and Rs 329 | Rs 994,158 | 2.45 |

The Rs 1 lakh rows reproduce the stored corrected-data results: {'AUC-FADE-2PCT': True, 'AUC-FADE-3PCT': True, 'SWING-ATTN-ON-3X': True}.
- VRP-STRANGLE-016-WINGED at the max-risk lot rule (median 3 lots; the lab used 2): DISCOVERY PF 1.245, net Rs 217,750 (2 lots: PF 1.238, Rs 135,643); FINAL PF 1.269, net Rs 119,911 (2 lots: PF 1.273, Rs 61,060).
- EV-PEAD-LONG-H60 is delivery: identical at any size.

## Types whose result stands unchanged

**Delivery swing** (107 strategies): 8 pass DISCOVERY, 3 confirmed on FINAL, 1 confirmed on 2016-23.
- `r3_swing_bb_lower_sma100`: DISCOVERY 254 trades, PF 1.533, net Rs 163,804 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 0.588, rejected; 2016-23: OOS-FAILED.
- `r3_swing_close_down4`: DISCOVERY 705 trades, PF 2.200, net Rs 1,192,260 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 0.932, rejected; 2016-23: OOS-FAILED.
- `r3_swing_gapdown_recover_1520`: DISCOVERY 186 trades, PF 2.903, net Rs 404,846 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 0.790, rejected; 2016-23: OOS-FAILED.
- `r3_swing_quiet_loser_rev`: DISCOVERY 191 trades, PF 2.708, net Rs 377,385 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 1.645, confirmed; 2016-23: OOS-CONFIRMED.
- `r3_swing_vix_capitulation`: DISCOVERY 195 trades, PF 6.417, net Rs 999,837 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 1.613, confirmed; 2016-23: OOS-FAILED.
- `r3_tfweekly_wk_loser_rev`: DISCOVERY 72 trades, PF 1.346, net Rs 64,317 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 1.843, confirmed; 2016-23: OOS-FAILED.
- `r4_delivery_lowdel_selloff_long`: DISCOVERY 69 trades, PF 1.684, net Rs 78,763 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 0.903, rejected; 2016-23: OOS-CONFIRMED.
- `r6_gh_wvf_stock`: DISCOVERY 2273 trades, PF 1.296, net Rs 906,738 at the lab size (per Rs 10 lakh (10 positions of Rs 1 lakh): the same PF); FINAL PF 0.859, rejected; 2016-23: OOS-FAILED.

**Index futures** (105 strategies): 0 pass DISCOVERY, 0 confirmed on FINAL, 0 confirmed on 2016-23.

`r3_swing_quiet_loser_rev` is the one catalogue rule that is confirmed on both the FINAL year and 2016-23 on corrected data; its Sharpe on the FINAL year is 1.18 and its DSR 0.1096 (needs 0.95), so it is not proven. Its size does not change, so nothing here changes that.

**Timeframe grid**: 7 of 1276 combos passed stage 1 in Rs 10 lakh accounts; 2016-23 confirmed: `nat:r3_swing_quiet_loser_rev:rr3`; proven (with DSR): 0. See the tfgrid report.

## Near misses (failures)

| Strategy | Kind | Trades | PF | Net | Why it failed DISCOVERY at Rs 5 lakh |
|---|---|---|---|---|---|
| `r4_mechanics_ban_entry_short_id` | intraday | 6 | 6.478 | Rs 72,434 | 6 trades < 150; IRCTC is 99% of net (> 35%) |
| `r5_kite_vars_cs_evening_star` | futures | 20 | 2.263 | Rs 182,810 | 20 trades < 60 |
| `r3_intraday_downday_strong_fade` | intraday | 234 | 1.253 | Rs 97,314 | SIEMENS is 61% of net (> 35%) |
| `r5_kite_vars_cs_bull_marubozu` | futures | 766 | 1.131 | Rs 688,972 | PF 1.131 < 1.15 |
| `r7_gh2_smash_stock` | futures | 3010 | 1.127 | Rs 4,301,571 | PF 1.127 < 1.15 |
| `r5_kite_vars_cs_morning_star_nxo` | futures | 30 | 1.119 | Rs 49,601 | PF 1.119 < 1.15; 30 trades < 60; OFSS is 146% of net (> 35%) |

Passed DISCOVERY and rejected on the FINAL year: `r3_futures_stk_tom_long`, `r3_tfdaily_bb_upper_short_downtrend`, `r4_delivery_lowdel_rally_short`, `r5_kite_vars_bb20_d`, `r5_kite_vars_cs_bull_marubozu_nxo`, `r7_gh2_tsp1_stock`.
Best intraday / scalping strategies at Rs 5 lakh (all fail): `r3_intraday_downday_strong_fade` PF 1.253 (SIEMENS is 61% of net (> 35%)); `r3_intraday_3day_reversal` PF 1.072 (H2 net <= 0, PF 1.072 < 1.15, RECLTD is 49% of net (> 35%)); `r3_intraday_idio_cont_1100` PF 1.071 (H2 net <= 0, PF 1.071 < 1.15, ADANIPOWER is 61% of net (> 35%)).

## Controls and checks

- Before the plan was frozen, the size wrapper forced to Rs 1 lakh reproduced the stored DISCOVERY results of `afternoon_vwap_trend`, `r3_scalping_5m_bigbar_fade`, `cs_bear_engulfing_d` to the paisa: **yes**.
- In the main pass every strategy's original-size twin was recomputed next to its new-size result: 435 of 435 equal their stored results.
- Random-entry placebos at Rs 5 lakh: 0 of 10 pass DISCOVERY (expected 0). Planted look-ahead edge at Rs 5 lakh passes: **yes**. Controls gate: **met**.
- DSR: N = 5434 (= 4979 earlier trials + K 435 + 20 FINAL looks), V = 0.027812; the luck bar is a Sharpe of 9.82 a year.
- The pre-registered V is large because most catalogue strategies trade often and lose their fees every day, which spreads their Sharpes widely; with it no strategy here can reach DSR 0.95 (all are 0.00). Report only, not a verdict: with the builder's much smaller V (0.0013) the FINAL-confirmed strategies' DSR would be `cs_gravestone_doji_d` 0.012 (Sharpe 0.84), `r4_delivery_brk_fake_short` 0.104 (Sharpe 1.42), `r5_kite_vars_cs_bull_engulfing` 0.025 (Sharpe 0.99), `VRP-STRANGLE-016-NAKED` 0.037 (Sharpe 0.91), `r3_options_wc_fri` 0.358 (Sharpe 1.84), `r4_realopt_nf_expwk_last2_strangle_1pct` 0.016 (Sharpe 0.90), `r4_realopt_nf_wk_strangle_3pct` 0.138 (Sharpe 1.30).
- Trials ledger: 458 rows appended as new trials (ids `sz10l:<id>`); ledger N for 2023-26 2552 -> 2987. Backup: `trials_v2.csv.bak_size10l_20261116_172735`.
- Two code fixes after the plans were hashed, neither touching a rule or a gate: a missing catalogue import in the FINAL stage and a wrong read of the option simulator's detail records (both crashed before any result).

## What this does and does not show

- Same strategies, entries, exits, data and gates; only the quantity per trade changed, so only the fee per rupee (and, for options, the weight of each trade) changed.
- The FINAL year and the 2016-23 years were already looked at once at the old size. This is a second, strongly correlated look, counted in N; it is not an independent confirmation.
- The minute universe is today's 97 NIFTY-100 names (survivorship-biased), as in every earlier minute test.
- At Rs 5 lakh per trade the 2 bps-a-side slippage assumption is untested; option freeze quantities are not modelled; the option model prices of options_vrp / options_r3 are Black-Scholes values, not quotes.
