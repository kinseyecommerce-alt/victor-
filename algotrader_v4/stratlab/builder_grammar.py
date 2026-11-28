"""
stratlab/builder_grammar.py — the GRAMMAR of the automatic strategy builder. Pure: no data, no returns.
Two frozen versions live side by side (a version is never edited, only added):
  bg1 (rule builder/v1)  agent x market type x signal block (+ its round parameters) x entry timing x exit rule x
                         instrument — the stock universe is implicit (minute grid: the 97 NIFTY-100 names; daily grid:
                         the point-in-time top 500)
  bg2 (rule builder/v2)  the same sentence x UNIVERSE (UNIVERSES below; a dimension of the draw):
                           NIFTY100  the 97 NIFTY-100 names of the lab (forward_test.UNIVERSE)
                           NIFTY200  NIFTY100 + the data layer's added NIFTY 200 names (the current NIFTY 200 list and
                                     the historical NIFTY 100 members still listed); on the minute grid only the added
                                     names with COMPLETE minute data over the candidate's whole split (engine rule)
                           TOP500    the point-in-time top 500 by traded value (daily-grid stock blocks only)
                           INDEX     NIFTY / BANKNIFTY instruments (no stock universe)
                         A bg2 sentence whose universe is the one bg1 used implicitly IS the bg1 sentence: it is never
                         drawn when its bg1 key was drawn (Candidate.v1_key); every other bg2 sentence is a NEW candidate
                         (e.g. the NIFTY200 versions of the two proven lab rules get no credit from their NIFTY-100 proof)
  bg3 (rule builder/v3)  bg2's sentences under the SIZE regime "c10" (Rs 10 lakh capital per agent): a sentence whose
                         trades the engine books at a size builder/v3 changed (size_regime(): every MIS stock sentence
                         of intraday / scalping, every stock sentence at futures costs = the futures agent's stock
                         sentences and the swing sentences that can go short; every options sentence, whose lots
                         now follow the capital) carries size "c10" and a NEW key (['bg3', ..., universe, 'c10']): it
                         can be drawn again and every draw is a new trial in N. A sentence whose size is unchanged
                         (swing long-only = CNC delivery at proportional fees; index futures at 1 real lot) IS its
                         bg2 sentence (same key, same JSON; through it its bg1 twin): never drawn again once that
                         key (or its bg1 twin) was drawn - its existing result IS its Rs 10 lakh result. The size
                         numbers live in the engine (stratlab/builder_engine.SIZE_TABLES, builder_options.OPTION_SIZES)
                         and in builder_rule_v3.json costs.size_rs.
Building blocks come from the lab's own data only (stratlab/datagen_features.py: opening auction gap / path, volume
shocks, index expiries, India VIX, delivery %, participant OI, the F&O ban list, futures OI quadrants, bulk / block
deals by client class, results timing, corporate actions) plus the NIFTY / VIX daily series. The DIRECTION of every
block is fixed by its economic story (written next to it); the builder never chooses a side from returns.

AGENTS (product, horizon)
  intraday  NIFTY-100 stocks, MIS, minute grid; entry at a decision minute of the session, exit by 15:15 (time exit,
            fixed stop + target incl. a 1:3 bracket, or a signal exit = a 1-minute close back through the session
            VWAP); signals = the minute blocks and the previous evening's end-of-day blocks
  scalping  NIFTY-100 stocks, MIS, 1-5-minute signals of the opening; exit <= 30 minutes after entry
  swing     stocks of the point-in-time top-500 universe, decided in the evening, entered at the next open; long = CNC
            delivery, short = stock futures (F&O names only); 2-60 sessions (time, bracket on daily high / low,
            signal = close through its 10-session mean, or the event's own date)
  options   NIFTY / BANKNIFTY options on REAL end-of-day prices (nsedata's option simulator: strike chosen on the
            previous session's chain, filled at the entry session's official close, expiry settlement at intrinsic)
  futures   NIFTY / BANKNIFTY futures (index spot stands in, as in the lab) and stock futures (cash price, futures
            costs): overnight / positional rules of both directions

MARKET TYPES (point-in-time, frozen; stratlab/builder_engine.market_types)
  RISING    NIFTY 50 close > its 50-session SMA and the SMA above its value 10 sessions earlier
  FALLING   close < SMA and the SMA below its value 10 sessions earlier
  SIDEWAYS  neither
  HIGH_VOL  India VIX close in the top quartile (>= 75th percentile) of its previous 252 sessions; overrides the three
A candidate trades only on sessions whose market type (known at its decision time) is its own.

THE DRAW (deterministic, no replacement across batches): each (agent, market type) cell's full grammar list is
ordered by sha256(<grammar version> | batch id | cell | candidate key); a batch takes, in that order, the first 20 whose
key (bg2: and whose bg1-equivalent key) was never drawn in an earlier batch and that is not a near-duplicate of one
already taken in this batch.
NEAR-DUPLICATE (fixed): same agent, market type, block, exit kind and instrument, and at most ONE other field of the
signature (each block parameter, the entry time, each exit parameter; bg2: also the universe) differs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

GRAMMAR_V1 = "bg1"
GRAMMAR_V2 = "bg2"
GRAMMAR_V3 = "bg3"
GRAMMAR_V4 = "bg4"
GRAMMAR_V5 = "bg5"
GRAMMAR_V6 = "bg6"
GRAMMAR_V7 = "bg7"
VERSIONS = (GRAMMAR_V1, GRAMMAR_V2, GRAMMAR_V3, GRAMMAR_V4, GRAMMAR_V5, GRAMMAR_V6, GRAMMAR_V7)
GRAMMAR_VERSION = GRAMMAR_V7             # the current version (new batches); bg1 / bg2 / bg3 stay reproducible byte for byte
# bg4 (builder/v4, 25 Nov 2026): bg3 + TRAILING-STOP exits for every minute-grid stock sentence of intraday and scalping
#   trail / trail_min        a stop x % behind the best price since entry (long: the highest high; short: the lowest
#                            low), moved after each bar closes (never inside a bar)
#   trail_be / trail_be_min  a fixed stop x % from entry until the price has moved x % in favour, then the stop goes to
#                            the entry price (breakeven) and trails x % behind the best price
#   every trailing exit keeps the agent's hard time limit (intraday 15:15, scalping 30 minutes after entry)
TRAIL_KINDS = ("trail", "trail_min", "trail_be", "trail_be_min")
# bg5 (builder/v5, 26 Nov 2026): bg4 + ATR TRAILING-STOP exits for the same sentences: the trailing distance is
#   k x ATR(14) / previous close, ATR(14) = the mean true range of the 14 sessions before the entry day (session high /
#   low / close from the symbol's own 1-minute bars; special sessions skipped); a trade without 15 earlier sessions is
#   not taken. Fills exactly as the bg4 kinds (builder_engine.walk_session): trail_atr = trail, trail_be_atr =
#   trail_be, trail_atr_min = trail_min, trail_be_atr_min = trail_be_min, at that distance.
ATR_KINDS = ("trail_atr", "trail_atr_min", "trail_be_atr", "trail_be_atr_min")
ATR_BASE = {"trail_atr": "trail", "trail_atr_min": "trail_min", "trail_be_atr": "trail_be",
            "trail_be_atr_min": "trail_be_min"}
SIZE_LEGACY = ""                         # bg1 / bg2: Rs 1 lakh per stock trade
SIZE_C10 = "c10"                         # bg3: Rs 10 lakh capital per agent (builder_engine.SIZE_TABLES)
SIZE_TEXT = {SIZE_C10: "Rs 5 lakh per trade (Rs 10 lakh capital per agent)"}
OPTION_SIZE_TEXT = {SIZE_C10: ("lots from Rs 10 lakh capital: Rs 1 lakh of premium (long / debit) or of maximum loss "
                               "(iron condor), Rs 5 lakh of margin (naked strangle)")}
UNIVERSES = ("NIFTY100", "NIFTY200", "TOP500", "INDEX")
MINUTE_UNIVERSES = ("NIFTY100", "NIFTY200")
DAILY_UNIVERSES = ("TOP500", "NIFTY100", "NIFTY200")
UNIVERSE_TEXT = {"NIFTY100": "NIFTY 100 stocks (the lab's 97)",
                 "NIFTY200": "NIFTY 200 stocks (the 97 + the added NIFTY 200 names)",
                 "TOP500": "the point-in-time top-500 stocks", "INDEX": "index"}
AGENTS = ("intraday", "scalping", "swing", "options", "futures")
MARKET_TYPES = ("RISING", "FALLING", "SIDEWAYS", "HIGH_VOL")
PER_CELL = 20
BATCH_SIZE = len(AGENTS) * len(MARKET_TYPES) * PER_CELL          # 400

ENTRY_MIN = {"o0915": 555, "o0920": 560, "o0930": 570, "o0945": 585, "o1430": 870, "o1515": 915, "o1521": 921}
HHMM = {k: f"{v // 60:02d}:{v % 60:02d}" for k, v in ENTRY_MIN.items()}


# ══════════════════════════════════════════════════════════════════════════
# signal blocks
# ══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class Block:
    name: str
    grid: str                 # minute | daily | idx_daily | idx_minute | prev_eod (daily block read on the minute grid)
    fn: str                   # function in datagen_features or builder_engine (prefix "be:")
    params: tuple             # ((name, (values...)), ...) : the round grid
    ready: str                # minute grid: the earliest entry field (decision taken at the close of the bar before)
    side: str                 # the story's direction: long | short | with | against | neutral (options premium)
    story: str
    text: str                 # plain English template, {param} placeholders

    def combos(self):
        names = [p[0] for p in self.params]
        vals = [p[1] for p in self.params]
        for v in itertools.product(*vals) if vals else [()]:
            yield tuple(zip(names, v))


def _b(name, grid, fn, params, ready, side, story, text):
    return Block(name, grid, fn, tuple((k, tuple(v)) for k, v in params.items()), ready, side, story, text)


S_GAP = "Opening call-auction imbalances overshoot; the first five minutes tell information (continues) from imbalance."
S_VOL = "Abnormal volume is attention / news; attention-driven pressure moves prices and partly reverts."
S_DEL = ("Delivery % separates investors who take shares home from intraday speculators: a delivery-heavy move is "
         "informed and drifts; a delivery-light spike is churn and fades.")
S_OI = ("Futures OI tells new positions from closed ones: build-ups are new money (continue), short covering / long "
        "unwinding are forced closes (revert).")
S_BAN = ("The F&O ban forbids new positions until OI falls below 80 % of the limit: entry forces a crowded trade to "
         "unwind; exit re-opens speculation.")
S_DEALS = "Large bulk / block trades by a client class reveal who accumulates; block price pressure partly reverts."
S_RES = "Results move prices; under-reaction drifts, over-reaction reverts."
S_VIX = "A VIX spike is forced de-risking; the fear premium is paid to whoever supplies liquidity."
S_FII = "FIIs are the large, informed index-futures players; retail (Client) positioning is the contrarian side."
S_EXP = "Option writers' hedging and its unwinding around index expiries move the index."
S_CA = "Splits / bonuses draw retail attention to 'cheaper' shares before the ex-date."
S_PREM = "Option premium is richest after fear spikes; selling it collects the variance risk premium."

BLOCKS = [
    # ── minute grid (stocks) ──
    _b("gap_go", "minute", "r_gap_path", {"g": (0.01, 0.02, 0.03), "mode": ("go",)}, "o0920", "with", S_GAP,
       "the stock opens >= {g:%} away from yesterday's close and the first five minutes move further the same way"),
    _b("gap_rev", "minute", "r_gap_path", {"g": (0.01, 0.02, 0.03), "mode": ("revfollow",)}, "o0920", "against", S_GAP,
       "the stock opens >= {g:%} away from yesterday's close and the first five minutes move back against the gap"),
    _b("gap_fade", "minute", "r_gap_path", {"g": (0.01, 0.02, 0.03), "mode": ("fade_day",)}, "o0915", "against", S_GAP,
       "the stock opens >= {g:%} away from yesterday's close (call auction)"),
    _b("open5", "minute", "r_open5_shock", {"r": (0.01, 0.02), "mode": ("cont", "fade")}, "o0920", "mode", S_GAP,
       "no real gap but the first five minutes move >= {r:%} ({mode})"),
    _b("idio_gap", "minute", "r_idio_gap", {"g": (0.01, 0.02)}, "o0915", "against", S_GAP,
       "the stock gaps >= {g:%} against NIFTY 50's own opening gap"),
    _b("early_vol", "minute", "r_early_vol", {"k": (3.0, 5.0), "mode": ("cont", "fade")}, "o0945", "mode", S_VOL,
       "09:15-09:44 volume >= {k}x its 20-session mean with a >= 0.5 % move ({mode})"),
    _b("gap_volume", "minute", "r_gap_volume", {"mode": ("lowvol_fade", "hivol_go")}, "o0930", "mode", S_VOL,
       "a >= 2 % gap with first-15-minute volume {mode}"),
    _b("results_gap", "minute", "r_results_gap", {"g": (0.02, 0.04), "mode": ("cont", "fade")}, "o0915", "mode", S_RES,
       "results came out after yesterday's close and the stock gaps >= {g:%} ({mode})"),
    _b("lasthour_vol", "minute", "r_lasthour_vol", {"k": (3.0, 5.0), "mode": ("cont", "fade")}, "o1521", "mode", S_VOL,
       "last-hour volume >= {k}x normal with a >= 0.5 % last-hour move ({mode})"),
    _b("down_attention", "minute", "r_down_attention", {"mode": ("rev", "cont")}, "o1521", "mode", S_VOL,
       "volume >= 3x normal on a down day ({mode})"),
    _b("attention_swing", "minute", "r_attention_swing", {"day": ("up", "down")}, "o1521", "long", S_VOL,
       "volume >= 3x normal on an {day} day"),
    # ── daily grid (stocks), decided in the evening ──
    _b("deliv_up", "daily", "r_delivery_shock", {"pp": (15.0, 25.0), "day": ("up",)}, "", "long", S_DEL,
       "delivery % >= its 20-day mean + {pp:g} points on >= 2x volume and a >= 2 % up day"),
    _b("deliv_down", "daily", "r_delivery_shock", {"pp": (15.0, 25.0), "day": ("down",)}, "", "short", S_DEL,
       "delivery % >= its 20-day mean + {pp:g} points on >= 2x volume and a >= 2 % down day"),
    _b("low_deliv", "daily", "r_low_delivery_spike", {}, "", "against", S_DEL,
       "delivery % >= 15 points below normal on >= 3x volume and a >= 3 % move"),
    _b("deliv_trend", "daily", "r_delivery_trend", {}, "", "long", S_DEL,
       "5-day delivery % >= its 60-day mean + 10 points and a positive 20-day return"),
    _b("oi_long_buildup", "daily", "r_oi_quadrant", {"quad": ("long_buildup",), "thr": (0.10, 0.20)}, "", "long", S_OI,
       "price up >= 2 % and stock-futures OI up >= {thr:%} (long build-up)"),
    _b("oi_short_buildup", "daily", "r_oi_quadrant", {"quad": ("short_buildup",), "thr": (0.10, 0.20)}, "", "short",
       S_OI, "price down >= 2 % and stock-futures OI up >= {thr:%} (short build-up)"),
    _b("oi_short_covering", "daily", "r_oi_quadrant", {"quad": ("short_covering",), "thr": (0.10, 0.20)}, "", "short",
       S_OI, "price up >= 2 % and stock-futures OI down >= {thr:%} (short covering)"),
    _b("oi_long_unwinding", "daily", "r_oi_quadrant", {"quad": ("long_unwinding",), "thr": (0.10, 0.20)}, "", "long",
       S_OI, "price down >= 2 % and stock-futures OI down >= {thr:%} (long unwinding)"),
    _b("ban_entry_short", "daily", "r_ban", {"event": ("entry",), "side": (-1,)}, "", "short", S_BAN,
       "the stock enters the F&O ban list for the next session"),
    _b("ban_entry_long", "daily", "r_ban", {"event": ("entry",), "side": (1,)}, "", "long", S_BAN,
       "the stock enters the F&O ban list for the next session (rebound after the forced unwinding)"),
    _b("ban_exit_long", "daily", "r_ban", {"event": ("exit",), "side": (1,)}, "", "long", S_BAN,
       "the stock leaves the F&O ban list for the next session"),
    _b("ban_exit_short", "daily", "r_ban", {"event": ("exit",), "side": (-1,)}, "", "short", S_BAN,
       "the stock leaves the F&O ban list for the next session (speculation re-opens on the short side)"),
    _b("inst_buy", "daily", "r_deals", {"cls": ("inst",), "flow": ("buy",), "side": (1,)}, "", "long", S_DEALS,
       "institutions net-bought >= Rs 10 crore in bulk / block deals today"),
    _b("inst_sell_rev", "daily", "r_deals", {"cls": ("inst",), "flow": ("sell",), "side": (1,)}, "", "long", S_DEALS,
       "institutions net-sold >= Rs 10 crore in bulk / block deals today (block pressure reverts)"),
    _b("other_buy", "daily", "r_deals", {"cls": ("other",), "flow": ("buy",), "side": (1,)}, "", "long", S_DEALS,
       "other clients net-bought >= Rs 10 crore in bulk / block deals today"),
    _b("other_sell", "daily", "r_deals", {"cls": ("other",), "flow": ("sell",), "side": (-1,)}, "", "short", S_DEALS,
       "other clients net-sold >= Rs 10 crore in bulk / block deals today"),
    _b("results_up", "daily", "r_results_reaction", {"a": (0.04, 0.08), "mode": ("up",)}, "", "long", S_RES,
       "on its results reaction day the stock beats NIFTY by >= {a:%}"),
    _b("results_down", "daily", "r_results_reaction", {"a": (0.04, 0.08), "mode": ("down",)}, "", "short", S_RES,
       "on its results reaction day the stock lags NIFTY by >= {a:%}"),
    _b("results_fade", "daily", "r_results_reaction", {"a": (0.08,), "mode": ("fade",)}, "", "against", S_RES,
       "on its results reaction day the stock moves >= {a:%} against NIFTY (over-reaction)"),
    _b("vix_spike_losers", "daily", "r_vix_spike_losers", {"x": (0.10, 0.15)}, "", "long", S_VIX,
       "India VIX rises >= {x:%} today: the day's biggest losers"),
    _b("vix_high_losers", "daily", "r_vix_high_losers", {}, "", "long", S_VIX,
       "India VIX is in the top decile of its last year and the stock fell >= 5 % in 5 sessions"),
    _b("vol_shock", "daily", "be:r_vol_shock", {"k": (3.0, 5.0), "day": ("up", "down")}, "", "day", S_VOL,
       "volume >= {k}x its 20-day mean on a >= 2 % {day} day"),
    _b("split_runup", "daily", "r_split_bonus", {"mode": ("runup",)}, "", "long", S_CA,
       "a split / bonus became known today (hold to the session before the ex-date)"),
    # ── index blocks ──
    _b("fii_flow", "idx_daily", "r_fii_flow", {"z": (1.0, 1.5)}, "", "with", S_FII,
       "FIIs' net index-futures position changes by >= {z} standard deviations today"),
    _b("client_contra", "idx_daily", "r_client_contrarian", {}, "", "against", S_FII,
       "retail (Client) index-futures long share is in the top / bottom decile of its last year"),
    _b("vix_high_index", "idx_daily", "r_vix_high_index", {}, "", "long", S_VIX,
       "India VIX is in the top decile of its last year"),
    _b("vix_spike_index", "idx_daily", "be:r_vix_spike_index", {"x": (0.10, 0.15)}, "", "long", S_VIX,
       "India VIX rises >= {x:%} today"),
    _b("nifty_drop", "idx_daily", "be:r_nifty_move", {"x": (0.015, 0.025), "mode": ("rebound",)}, "", "against",
       S_VIX, "NIFTY 50 falls >= {x:%} today"),
    _b("expiry", "idx_minute", "r_expiry", {"when": ("on", "pre"), "side": (1, -1)}, "o1521", "side", S_EXP,
       "{when} the index's weekly / monthly expiry session"),
    _b("idx_gap", "idx_minute", "be:r_index_gap", {"g": (0.005, 0.01), "mode": ("fade", "go")}, "o0915", "mode",
       S_GAP, "the index opens >= {g:%} away from yesterday's close ({mode})"),
]
S_PRICE = ("Price-history anomalies documented in many markets: one-week extremes revert (liquidity providers are "
           "paid to absorb over-reaction); 52-week highs and 12-1 month winners keep drifting (anchoring, slow "
           "diffusion of news); trend crosses and range breakouts on volume continue (time-series momentum).")
# builder/v6 (bg6): NEW daily stock blocks. Kept OUT of BLOCKS so bg1 .. bg5 stay byte for byte; their sentences are
# generated by the same bg1 -> bg2 -> bg3 rules (swing: every exit; futures: the short-side blocks) and appended.
BLOCKS_BG6 = [
    _b("st_rev_losers", "daily", "be:r_st_reversal", {"x": (0.08, 0.12), "mode": ("losers",)}, "", "long", S_PRICE,
       "the stock fell >= {x:%} over the last 5 sessions (one-week loser)"),
    _b("st_rev_winners", "daily", "be:r_st_reversal", {"x": (0.08, 0.12), "mode": ("winners",)}, "", "short", S_PRICE,
       "the stock rose >= {x:%} over the last 5 sessions (one-week winner)"),
    _b("high52", "daily", "be:r_high52", {"x": (0.01, 0.03)}, "", "long", S_PRICE,
       "the stock closes within {x:%} of its 52-week closing high"),
    _b("mom12_1", "daily", "be:r_mom12_1", {"m": (0.5, 1.0)}, "", "long", S_PRICE,
       "the stock rose >= {m:%} from 12 months to 1 month ago (12-1 momentum)"),
    _b("ma200_up", "daily", "be:r_ma200", {"k": (1.0, 2.0), "mode": ("up",)}, "", "long", S_PRICE,
       "the stock closes above its 200-day average after closing below it, on >= {k}x its 20-day volume"),
    _b("ma200_down", "daily", "be:r_ma200", {"k": (1.0, 2.0), "mode": ("down",)}, "", "short", S_PRICE,
       "the stock closes below its 200-day average after closing above it, on >= {k}x its 20-day volume"),
    _b("breakout20", "daily", "be:r_breakout20", {"k": (1.5, 2.5), "mode": ("up",)}, "", "long", S_PRICE,
       "the stock closes above its 20-day high on >= {k}x its 20-day volume"),
    _b("breakdown20", "daily", "be:r_breakout20", {"k": (1.5, 2.5), "mode": ("down",)}, "", "short", S_PRICE,
       "the stock closes below its 20-day low on >= {k}x its 20-day volume"),
]
S_MR = ("Short-term over-reaction inside a long-term uptrend reverts (RSI(2) / internal bar strength / down-streak "
        "dips above the 200-day mean); a volatility contraction or a gap that holds through the session resolves in "
        "its direction.")
# builder/v7 (bg7): NEW daily stock blocks, buy and sell side (swing longs in CNC, shorts in stock futures; the
# futures agent takes the sell-side ones): tradeable live in PAPER by bld_promoted the day they pass. Kept out of BLOCKS / BLOCKS_BG6 so every earlier grammar stays byte for byte.
BLOCKS_BG7 = [
    _b("rsi2_dip", "daily", "be:r_rsi2_dip", {"thr": (5.0, 10.0)}, "", "long", S_MR,
       "RSI(2) <= {thr} while the stock closes above its 200-day average"),
    _b("ibs_low", "daily", "be:r_ibs_low", {"x": (0.1, 0.2)}, "", "long", S_MR,
       "the stock closes in the bottom {x:%} of its day's range while above its 200-day average"),
    _b("down_streak", "daily", "be:r_down_streak", {"n": (3, 4)}, "", "long", S_MR,
       "{n} lower closes in a row while the stock is above its 200-day average"),
    _b("nr7_up", "daily", "be:r_nr7_up", {"p": (0.5, 0.75)}, "", "long", S_MR,
       "the narrowest day of the last 7 (NR7) with the close in the top part of the range (>= {p:%})"),
    _b("gap_up_hold", "daily", "be:r_gap_up_hold", {"g": (0.03, 0.05)}, "", "long", S_MR,
       "the stock opens >= {g:%} above yesterday's close and closes at or above its open"),
    # the SELL side: mirror images (swing shorts in stock futures; also the futures agent's short-side rule)
    _b("rsi2_spike", "daily", "be:r_rsi2_spike", {"thr": (5.0, 10.0)}, "", "short", S_MR,
       "RSI(2) >= 100 - {thr} while the stock closes below its 200-day average"),
    _b("ibs_high", "daily", "be:r_ibs_high", {"x": (0.1, 0.2)}, "", "short", S_MR,
       "the stock closes in the top {x:%} of its day's range while below its 200-day average"),
    _b("up_streak", "daily", "be:r_up_streak", {"n": (3, 4)}, "", "short", S_MR,
       "{n} higher closes in a row while the stock is below its 200-day average"),
    _b("nr7_down", "daily", "be:r_nr7_down", {"p": (0.5, 0.75)}, "", "short", S_MR,
       "the narrowest day of the last 7 (NR7) with the close in the bottom part of the range"),
    _b("gap_down_hold", "daily", "be:r_gap_down_hold", {"g": (0.03, 0.05)}, "", "short", S_MR,
       "the stock opens >= {g:%} below yesterday's close and closes at or below its open"),
]
S_IDX = ("Index-level versions of the same effects: short-term index over-reaction inside its trend reverts, trend "
         "crosses / breakouts and a gap that holds continue, month-end / month-start flows lift the index, a VIX drop "
         "is fear unwinding.")
S_MOM = ("Intraday momentum: the market's move over the day so far predicts the last hour, as late traders and "
         "hedgers trade in the same direction (Gao, Han, Li and Zhou 2018).")
# bg7 index blocks (options and index futures, NIFTY / BANKNIFTY on NIFTY 50's signal) and the 14:30 intraday
# momentum block (intraday / scalping); plus every bg6 / bg7 daily stock block traded INTRADAY the next morning.
BLOCKS_BG7_IDX = [
    _b("idx_rsi2_dip", "idx_daily", "be:r_idx_rsi2", {"thr": (5.0, 10.0), "mode": ("dip",)}, "", "long", S_IDX,
       "NIFTY 50's RSI(2) <= {thr} while it closes above its 200-day average"),
    _b("idx_rsi2_spike", "idx_daily", "be:r_idx_rsi2", {"thr": (5.0, 10.0), "mode": ("spike",)}, "", "short", S_IDX,
       "NIFTY 50's RSI(2) >= 100 - {thr} while it closes below its 200-day average"),
    _b("idx_down_streak", "idx_daily", "be:r_idx_streak", {"n": (3, 4), "mode": ("down",)}, "", "long", S_IDX,
       "NIFTY 50 closes lower {n} days in a row while above its 200-day average"),
    _b("idx_up_streak", "idx_daily", "be:r_idx_streak", {"n": (3, 4), "mode": ("up",)}, "", "short", S_IDX,
       "NIFTY 50 closes higher {n} days in a row while below its 200-day average"),
    _b("idx_tom", "idx_daily", "be:r_idx_tom", {}, "", "long", S_IDX,
       "today is the last session of the month (turn of the month)"),
    _b("idx_ma200_up", "idx_daily", "be:r_idx_ma200", {"mode": ("up",)}, "", "long", S_IDX,
       "NIFTY 50 closes above its 200-day average after closing below it"),
    _b("idx_ma200_down", "idx_daily", "be:r_idx_ma200", {"mode": ("down",)}, "", "short", S_IDX,
       "NIFTY 50 closes below its 200-day average after closing above it"),
    _b("idx_breakout", "idx_daily", "be:r_idx_breakout", {"mode": ("up",)}, "", "long", S_IDX,
       "NIFTY 50 closes above its highest close of the last 20 sessions"),
    _b("idx_breakdown", "idx_daily", "be:r_idx_breakout", {"mode": ("down",)}, "", "short", S_IDX,
       "NIFTY 50 closes below its lowest close of the last 20 sessions"),
    _b("idx_gap_up_hold", "idx_daily", "be:r_idx_gap_hold", {"g": (0.005, 0.01), "mode": ("up",)}, "", "long", S_IDX,
       "NIFTY 50 opens >= {g:%} up and closes at or above its open"),
    _b("idx_gap_down_hold", "idx_daily", "be:r_idx_gap_hold", {"g": (0.005, 0.01), "mode": ("down",)}, "", "short",
       S_IDX, "NIFTY 50 opens >= {g:%} down and closes at or below its open"),
    _b("vix_drop", "idx_daily", "be:r_vix_drop", {"x": (0.10, 0.15)}, "", "long", S_IDX,
       "India VIX falls >= {x:%} today"),
]
BLOCKS_BG7_MIN = [
    _b("halfhour_mom", "minute", "be:r_halfhour_mom", {"x": (0.005, 0.01), "src": ("open30", "day")}, "o1430", "with",
       S_MOM, "the stock is >= {x:%} away from yesterday's close at {src} (open30 = 09:44, day = 14:29)"),
]
BLOCK = {b.name: b for b in BLOCKS + BLOCKS_BG6 + BLOCKS_BG7 + BLOCKS_BG7_IDX + BLOCKS_BG7_MIN}
BG6_BLOCKS = frozenset(b.name for b in BLOCKS_BG6)
BG7_BLOCKS = frozenset(b.name for b in BLOCKS_BG7 + BLOCKS_BG7_IDX + BLOCKS_BG7_MIN)

PREV_EOD_FOR_INTRADAY = ("deliv_up", "deliv_down", "oi_long_buildup", "oi_short_buildup", "oi_short_covering",
                         "oi_long_unwinding", "ban_entry_short", "ban_entry_long", "inst_buy", "other_sell",
                         "results_up", "results_down", "vix_spike_losers", "vol_shock")


# ══════════════════════════════════════════════════════════════════════════
# exits
# ══════════════════════════════════════════════════════════════════════════
def _exits(agent: str) -> list:
    """(kind, params...) — params are round values; every exit has a hard time limit."""
    if agent == "intraday":
        out = [("time", "o1430"), ("time", "o1515")]
        out += [("bracket", s, rr) for s in (0.005, 0.01, 0.015) for rr in (1.5, 2.0, 3.0)]
        out += [("signal", "vwap")]
        return out
    if agent == "scalping":
        out = [("time_min", m) for m in (5, 10, 15, 25)]
        out += [("bracket_min", s, rr, 30) for s in (0.003, 0.005) for rr in (2.0, 3.0)]
        out += [("signal_min", "vwap", 30)]
        return out
    if agent == "swing":
        out = [("hold", h) for h in (2, 3, 5, 10, 20, 40, 60)]
        out += [("bracket_d", s, rr, mh) for s in (0.03, 0.05, 0.08) for rr in (2.0, 3.0) for mh in (20, 60)]
        out += [("signal_d", "sma10", 60)]
        return out
    if agent == "futures":
        out = [("hold", h) for h in (1, 2, 3, 5, 10)]
        out += [("bracket_d", s, rr, 10) for s in (0.015, 0.03) for rr in (2.0, 3.0)]
        out += [("signal_d", "sma10", 10)]
        return out
    if agent == "options":
        out = [("opt_hold", h) for h in (1, 3, 5)] + [("opt_expiry",)]
        out += [("opt_bracket", "1:3")]
        out += [("opt_signal", "sma10")]
        return out
    raise KeyError(agent)


FUT_MINUTE_EXITS = [("overnight",), ("hold_m", 5), ("time", "o1515"), ("bracket", 0.005, 3.0), ("signal", "vwap")]
OPTION_STRUCTURES = {"directional": ("long_atm", "debit_spread"), "neutral": ("short_strangle", "iron_condor")}
OPTION_EXPIRY = ("weekly", "monthly")


# ══════════════════════════════════════════════════════════════════════════
# candidates
# ══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class Candidate:
    agent: str
    mtype: str
    block: str
    params: tuple             # ((name, value), ...)
    entry: str                # entry field (minute grid), "next_open" (daily), "close" (options)
    exit: tuple               # (kind, params...)
    instrument: str           # STOCKS | NIFTY | BANKNIFTY
    product: str              # mis | cnc | fut | opt (side-dependent for swing: "cnc/fut")
    structure: str = ""       # options: long_atm | debit_spread | short_strangle | iron_condor (+ expiry)
    universe: str = ""        # bg2 / bg3 (UNIVERSES); "" = a bg1 sentence (its implicit universe: implied_universe())
    size: str = ""            # bg3 only: the size regime of a sentence whose trade size builder/v3 changed (SIZE_C10);
    #                           "" = Rs 1 lakh per stock trade (every bg1 / bg2 sentence and bg3's unchanged twins)

    @property
    def version(self) -> str:
        if self.block in BG7_BLOCKS or (self.block in BG6_BLOCKS and self.agent == "intraday"):
            return GRAMMAR_V7                 # bg7 blocks, and bg6 blocks traded intraday: only in bg7, always new
        if self.block in BG6_BLOCKS:
            return GRAMMAR_V6                 # a bg6 block exists only in bg6: always a new candidate
        if self.exit and self.exit[0] in ATR_KINDS:
            return GRAMMAR_V5                 # an ATR-trailing sentence exists only in bg5: always a new candidate
        if self.exit and self.exit[0] in TRAIL_KINDS:
            return GRAMMAR_V4                 # a trailing-exit sentence exists only in bg4: always a new candidate
        if self.size:
            return GRAMMAR_V3
        return GRAMMAR_V2 if self.universe else GRAMMAR_V1

    def _key(self, version: str, universe: str, size: str = "") -> str:
        body = [version, self.agent, self.mtype, self.block, [list(p) for p in self.params], self.entry,
                list(self.exit), self.instrument, self.product, self.structure]
        if universe:
            body.append(universe)
        if size:
            body.append(size)
        return json.dumps(body, separators=(",", ":"))

    @property
    def key(self) -> str:
        """The identity across batches (never re-tested). A bg1 / bg2 sentence keeps its key byte for byte; a bg3
        sentence of unchanged size IS its bg2 sentence (size ""), a re-sized one has its own bg3 key."""
        return self._key(self.version, self.universe, self.size)

    @property
    def v1_key(self) -> str | None:
        """The bg1 key of the SAME sentence: a bg2 candidate whose universe is the one bg1 used implicitly is the bg1
        candidate (never drawn again once its bg1 key was drawn). None when the universe is new (no bg1 twin)."""
        if self.size or (self.exit and self.exit[0] in TRAIL_KINDS + ATR_KINDS) or self.block in BG6_BLOCKS | BG7_BLOCKS:
            return None                      # a re-sized bg3 / a bg4 trailing sentence is new: no twin in bg1 / bg2
        if not self.universe:
            return self.key
        if self.universe != implied_universe(self.agent, self.block, self.instrument):
            return None
        return self._key(GRAMMAR_V1, "")

    @property
    def stock_universe(self) -> str:
        """The universe the engine applies (bg1: the implicit one)."""
        return self.universe or implied_universe(self.agent, self.block, self.instrument)

    @property
    def cell(self) -> str:
        return f"{self.agent}|{self.mtype}"

    @property
    def kw(self) -> dict:
        return dict(self.params)

    @property
    def grid(self) -> str:
        return BLOCK[self.block].grid if not (self.agent == "intraday" and BLOCK[self.block].grid == "daily") \
            else "prev_eod"

    @property
    def horizon(self) -> str:
        return "intraday" if self.agent in ("intraday", "scalping") or self.exit[0] in ("time", "bracket", "signal") \
            else "positional"

    @property
    def single(self) -> bool:
        return self.instrument in ("NIFTY", "BANKNIFTY")

    def signature(self) -> tuple:
        """Fields compared by the near-duplicate rule (besides the fixed agent / market / block / exit kind)."""
        sig = tuple(v for _k, v in self.params) + (self.entry, self.structure) + tuple(self.exit[1:])
        return sig + (self.universe,) if self.universe else sig

    def to_json(self) -> dict:
        d = {"agent": self.agent, "mtype": self.mtype, "block": self.block, "params": [list(p) for p in self.params],
             "entry": self.entry, "exit": list(self.exit), "instrument": self.instrument, "product": self.product,
             "structure": self.structure}
        if self.universe:                    # a bg1 candidate's JSON is unchanged
            d["universe"] = self.universe
        if self.size:                        # a bg1 / bg2 candidate's JSON is unchanged
            d["size"] = self.size
        return d

    @staticmethod
    def from_json(d: dict) -> "Candidate":
        return Candidate(d["agent"], d["mtype"], d["block"], tuple(tuple(p) for p in d["params"]), d["entry"],
                         tuple(d["exit"]), d["instrument"], d["product"], d.get("structure", ""),
                         d.get("universe", "") or "", d.get("size", "") or "")


def exec_grid(agent: str, block: str) -> str:
    """The grid a sentence EXECUTES on: an intraday sentence on a daily block runs on the minute grid (prev_eod)."""
    g = BLOCK[block].grid
    return "prev_eod" if (agent == "intraday" and g == "daily") else g


def implied_universe(agent: str, block: str, instrument: str) -> str:
    """The universe a bg1 sentence used implicitly: index instruments -> INDEX; stock sentences on the minute grid
    (incl. intraday on a daily block) -> NIFTY100 (the 97); stock sentences on the daily grid -> TOP500."""
    if instrument in ("NIFTY", "BANKNIFTY"):
        return "INDEX"
    return "NIFTY100" if exec_grid(agent, block) in ("minute", "prev_eod", "idx_minute") else "TOP500"


def universes_for(agent: str, block: str, instrument: str) -> tuple:
    """bg2's universe choices of a sentence."""
    if instrument in ("NIFTY", "BANKNIFTY"):
        return ("INDEX",)
    return MINUTE_UNIVERSES if exec_grid(agent, block) in ("minute", "prev_eod", "idx_minute") else DAILY_UNIVERSES


def size_regime(agent: str, block: str, params, instrument: str) -> str:
    """bg3: the size regime of a sentence, from how the engine BOOKS its trades (builder_engine.cost_kind):
      SIZE_C10  at least one trade is an MIS stock trade (intraday, scalping) or a stock trade at futures costs (the
                futures agent's stock sentences; swing shorts), or the sentence is an OPTIONS sentence (its lots
                follow the capital: builder_options.option_lots) - the sizes builder/v3 changed
      ""        every trade keeps its builder/v1 size: swing sentences that only go long (CNC delivery; the block's
                story side is "long", or a "day" block on an up day) and index FUTURES (1 real lot)."""
    if agent == "options":
        return SIZE_C10
    if instrument in ("NIFTY", "BANKNIFTY"):
        return SIZE_LEGACY
    if agent in ("intraday", "scalping", "futures"):
        return SIZE_C10
    b = BLOCK[block]
    kw = dict(params)
    long_only = b.side == "long" or (b.side == "day" and kw.get("day") == "up") or \
        (b.side == "side" and kw.get("side", 0) > 0)
    return SIZE_LEGACY if long_only else SIZE_C10


def _side_word(b: Block, params: dict) -> str:
    s = b.side
    if s == "mode":
        m = params.get("mode")
        return {"cont": "with", "fade": "against", "go": "with", "hivol_go": "with", "lowvol_fade": "against",
                "rev": "long", "rebound": "against"}.get(m, m)
    if s == "day":
        return "with"
    if s == "side":
        return "long" if params.get("side", 1) > 0 else "short"
    return s


def _entries(agent: str, b: Block) -> list:
    if agent == "intraday":
        if b.grid == "daily":
            return ["o0915", "o0920"]
        order = ["o0915", "o0920", "o0930", "o0945"]
        if b.ready not in order:
            return []
        i = order.index(b.ready)
        return order[i:i + 2]
    if agent == "scalping":
        order = ["o0915", "o0920", "o0930", "o0945"]
        if b.ready not in order:
            return []
        i = order.index(b.ready)
        return order[i:i + 2]
    return []


def grammar(agent: str, version: str | None = None) -> list:
    """Every sentence of one agent's grammar, without the market type (a fixed order). bg1: 7-tuples (instrument,
    product, block, params, entry, exit, structure); bg2: the same rows, each repeated once per universe choice
    (universes_for) with the universe appended (8-tuples); bg3: bg2's rows with the size regime appended (9-tuples,
    size_regime(): "c10" or "")."""
    version = version or GRAMMAR_VERSION
    if version == GRAMMAR_V1:
        return _grammar_bg1(agent)
    if version == GRAMMAR_V4:
        return _grammar_bg4(agent)
    if version == GRAMMAR_V5:
        return _grammar_bg5(agent)
    if version == GRAMMAR_V6:
        return _grammar_bg6(agent)
    if version == GRAMMAR_V7:
        return _grammar_bg7(agent)
    if version not in (GRAMMAR_V2, GRAMMAR_V3):
        raise KeyError(version)
    out = []
    for row in _grammar_bg1(agent):
        for u in universes_for(agent, row[2], row[0]):
            out.append(row + (u,))
    if version == GRAMMAR_V3:
        out = [row + (size_regime(agent, row[2], row[3], row[0]),) for row in out]
    return out


TRAIL_EXITS = {"intraday": [("trail", 0.005), ("trail", 0.01), ("trail_be", 0.005), ("trail_be", 0.01)],
               "scalping": [("trail_min", 0.003, 30), ("trail_min", 0.005, 30), ("trail_be_min", 0.003, 30),
                            ("trail_be_min", 0.005, 30)]}


def _grammar_bg4(agent: str) -> list:
    """bg3 byte for byte, then (intraday / scalping only) one row per distinct bg3 minute-grid sentence without its
    exit, for each trailing exit of TRAIL_EXITS, in bg3's order (first occurrence)."""
    base = grammar(agent, GRAMMAR_V3)
    if agent not in TRAIL_EXITS:
        return base
    seen, extra = set(), []
    for row in base:
        inst, prod, block, p, entry, _ex, st, u, size = row
        if BLOCK[block].grid != "minute" or inst != "STOCKS":
            continue
        k = (inst, prod, block, p, entry, st, u, size)
        if k in seen:
            continue
        seen.add(k)
        for ex in TRAIL_EXITS[agent]:
            extra.append((inst, prod, block, p, entry, ex, st, u, size))
    return base + extra


ATR_EXITS = {"intraday": [("trail_atr", 0.5), ("trail_atr", 1.0), ("trail_be_atr", 0.5), ("trail_be_atr", 1.0)],
             "scalping": [("trail_atr_min", 0.25, 30), ("trail_atr_min", 0.5, 30), ("trail_be_atr_min", 0.25, 30),
                          ("trail_be_atr_min", 0.5, 30)]}


def _grammar_bg5(agent: str) -> list:
    """bg4 byte for byte, then the same sentences as bg4's trailing block (one per distinct bg3 minute-grid stock
    sentence, bg3's order) with each ATR trailing exit of ATR_EXITS."""
    base = grammar(agent, GRAMMAR_V4)
    if agent not in ATR_EXITS:
        return base
    seen, extra = set(), []
    for row in grammar(agent, GRAMMAR_V3):
        inst, prod, block, p, entry, _ex, st, u, size = row
        if BLOCK[block].grid != "minute" or inst != "STOCKS":
            continue
        k = (inst, prod, block, p, entry, st, u, size)
        if k in seen:
            continue
        seen.add(k)
        for ex in ATR_EXITS[agent]:
            extra.append((inst, prod, block, p, entry, ex, st, u, size))
    return base + extra


def _grammar_bg7(agent: str) -> list:
    """bg6 byte for byte, then, in this order, the new bg7 sentences of each agent under bg1's exits / bg2's universes
    / bg3's size regime:
      swing, futures   BLOCKS_BG7 (every block; the futures rule takes the sell side) and BLOCKS_BG7_IDX (futures:
                       NIFTY / BANKNIFTY futures, both sides)
      options          BLOCKS_BG7_IDX (NIFTY / BANKNIFTY options, directional structures, every expiry and exit)
      intraday         every BLOCKS_BG6 / BLOCKS_BG7 daily STOCK block traded intraday the next morning (entries
                       09:15 / 09:20, intraday exits: bg1's previous-evening rule) and BLOCKS_BG7_MIN from 14:30
                       (exits: 15:15 time, every bracket, the VWAP signal)
      scalping         BLOCKS_BG7_MIN from 14:30 (scalping exits, all within 30 minutes)"""
    base = grammar(agent, GRAMMAR_V6)
    rows = []
    if agent in ("swing", "futures"):
        rows += _grammar_bg1(agent, BLOCKS_BG7)
        if agent == "futures":
            rows += _grammar_bg1(agent, BLOCKS_BG7_IDX)
    elif agent == "options":
        rows += _grammar_bg1(agent, BLOCKS_BG7_IDX)
    elif agent == "intraday":
        for b in BLOCKS_BG6 + BLOCKS_BG7:
            for p in b.combos():
                for en in ("o0915", "o0920"):
                    for ex in _exits(agent):
                        if ex[0] == "time" and ENTRY_MIN[ex[1]] <= ENTRY_MIN[en] + 30:
                            continue
                        rows.append(("STOCKS", "mis", b.name, p, en, ex, ""))
    if agent in ("intraday", "scalping"):
        for b in BLOCKS_BG7_MIN:
            for p in b.combos():
                for ex in _exits(agent):
                    if ex[0] == "time" and ENTRY_MIN[ex[1]] <= ENTRY_MIN[b.ready] + 30:
                        continue
                    rows.append(("STOCKS", "mis", b.name, p, b.ready, ex, ""))
    extra = []
    for row in rows:
        for u in universes_for(agent, row[2], row[0]):
            extra.append(row + (u, size_regime(agent, row[2], row[3], row[0])))
    return base + extra


def _grammar_bg6(agent: str) -> list:
    """bg5 byte for byte, then the BLOCKS_BG6 sentences of the swing / futures agents under the same rules as every
    earlier block (bg1's exits, bg2's universes, bg3's size regime), in BLOCKS_BG6 order."""
    base = grammar(agent, GRAMMAR_V5)
    if agent not in ("swing", "futures"):
        return base
    extra = []
    for row in _grammar_bg1(agent, BLOCKS_BG6):
        for u in universes_for(agent, row[2], row[0]):
            extra.append(row + (u, size_regime(agent, row[2], row[3], row[0])))
    return base + extra


def _grammar_bg1(agent: str, blocks: list | None = None) -> list:
    """bg1 exactly as frozen in builder/v1 (never edited); `blocks` (bg6) runs the same rules over a new block list."""
    BLOCKS = blocks if blocks is not None else globals()["BLOCKS"]   # noqa: N806 - the frozen loop text below
    out = []
    if agent in ("intraday", "scalping"):
        for b in BLOCKS:
            if b.grid == "minute" and b.ready != "o1521":
                pass
            elif agent == "intraday" and b.name in PREV_EOD_FOR_INTRADAY:
                pass
            else:
                continue
            for p in b.combos():
                for en in _entries(agent, b):
                    for ex in _exits(agent):
                        if agent == "intraday" and ex[0] == "time" and ENTRY_MIN[ex[1]] <= ENTRY_MIN[en] + 30:
                            continue
                        out.append(("STOCKS", "mis", b.name, p, en, ex, ""))
        return out
    if agent == "swing":
        for b in BLOCKS:
            if b.grid != "daily":
                continue
            for p in b.combos():
                exits = _exits(agent)
                if b.name == "split_runup":
                    exits = [("event",)] + [("bracket_d", s, rr, 40) for s in (0.05, 0.08) for rr in (2.0, 3.0)]
                for ex in exits:
                    out.append(("STOCKS", "cnc/fut", b.name, p, "next_open", ex, ""))
        return out
    if agent == "futures":
        for b in BLOCKS:
            if b.grid == "daily" and b.side in ("short", "against", "day", "with") and b.name != "split_runup":
                for p in b.combos():
                    for ex in _exits(agent):
                        out.append(("STOCKS", "fut", b.name, p, "next_open", ex, ""))
            elif b.grid == "idx_daily":
                for inst in ("NIFTY", "BANKNIFTY"):
                    for p in b.combos():
                        for ex in _exits(agent):
                            out.append((inst, "fut", b.name, p, "next_open", ex, ""))
            elif b.grid == "minute" and b.ready == "o1521":
                for p in b.combos():
                    for ex in (("overnight",), ("hold_m", 5)):
                        out.append(("STOCKS", "fut", b.name, p, "o1521", ex, ""))
            elif b.grid == "idx_minute":
                for inst in ("NIFTY", "BANKNIFTY"):
                    for p in b.combos():
                        exits = [("overnight",), ("hold_m", 5)] if b.ready == "o1521" else \
                            [("time", "o1515"), ("bracket", 0.005, 3.0), ("bracket", 0.0025, 3.0), ("signal", "vwap")]
                        for ex in exits:
                            en = b.ready if b.ready == "o1521" else "o0920" if p and dict(p).get("mode") == "go" \
                                else "o0915"
                            out.append((inst, "fut", b.name, p, en, ex, ""))
        return out
    if agent == "options":
        for b in BLOCKS:
            if b.grid != "idx_daily" and b.name != "expiry":
                continue
            if b.name == "expiry":
                combos = [(("when", "pre"), ("side", 0))]
                kinds = ("neutral",)
            else:
                combos = list(b.combos())
                kinds = ("directional", "neutral") if b.name in ("vix_high_index", "vix_spike_index") else \
                    ("directional",)
            for inst in ("NIFTY", "BANKNIFTY"):
                for p in combos:
                    for kind in kinds:
                        for st in OPTION_STRUCTURES[kind]:
                            for exp in OPTION_EXPIRY:
                                for ex in _exits(agent):
                                    if ex[0] == "opt_signal" and kind == "neutral":
                                        ex = ("opt_signal", "vix_pct50")
                                    out.append((inst, "opt", b.name, p, "close", ex, f"{st}:{exp}"))
        return out
    raise KeyError(agent)


def cell_grammar(agent: str, mtype: str, version: str | None = None) -> list:
    return [Candidate(agent, mtype, row[2], row[3], row[4], row[5], row[0], row[1], row[6],
                      row[7] if len(row) > 7 else "", row[8] if len(row) > 8 else "")
            for row in grammar(agent, version)]


def grammar_sizes(version: str | None = None) -> dict:
    return {a: len(grammar(a, version)) for a in AGENTS}


def near_duplicate(a: Candidate, b: Candidate) -> bool:
    """The fixed similarity rule (module docstring)."""
    if (a.agent, a.mtype, a.block, a.exit[0], a.instrument) != (b.agent, b.mtype, b.block, b.exit[0], b.instrument):
        return False
    sa, sb = a.signature(), b.signature()
    if len(sa) != len(sb):
        return False
    return sum(1 for x, y in zip(sa, sb) if x != y) <= 1


def _order_key(batch_id: str, cell: str, key: str, version: str | None = None) -> str:
    return hashlib.sha256(f"{version or GRAMMAR_VERSION}|{batch_id}|{cell}|{key}".encode()).hexdigest()


def never_drawn(c: Candidate, drawn: set) -> bool:
    """True when neither the candidate's key nor the key of its bg1 twin was drawn in an earlier batch (bg2 and bg3's
    unchanged-size sentences; a re-sized bg3 sentence has no twin: only its own bg3 key counts)."""
    if c.key in drawn:
        return False
    v1 = c.v1_key
    return not (v1 is not None and v1 in drawn)


def draw_batch(batch_id: str, drawn_keys, per_cell: int = PER_CELL, version: str | None = None) -> list:
    """The batch's candidates: per (agent, market type) cell the first `per_cell` of the hash-ordered grammar list that
    were never drawn before (drawn_keys; bg2: nor their bg1 twin) and are not near-duplicates of one already taken.
    Deterministic."""
    version = version or GRAMMAR_VERSION
    drawn = set(drawn_keys)
    out = []
    for agent in AGENTS:
        for mt in MARKET_TYPES:
            pool = cell_grammar(agent, mt, version)
            pool.sort(key=lambda c: _order_key(batch_id, c.cell, c.key, version))
            taken = []
            for c in pool:
                if len(taken) >= per_cell:
                    break
                if not never_drawn(c, drawn):
                    continue
                if any(near_duplicate(c, t) for t in taken):
                    continue
                taken.append(c)
            out.extend(taken)
    return out


def candidate_id(batch_id: str, c: Candidate, k: int) -> str:
    return f"bld{batch_id}_{c.agent[:3]}_{c.mtype[:3].lower()}_{k:02d}_{hashlib.sha1(c.key.encode()).hexdigest()[:6]}"


# ══════════════════════════════════════════════════════════════════════════
# plain English (generated from the grammar)
# ══════════════════════════════════════════════════════════════════════════
MT_TEXT = {"RISING": "in a RISING market (NIFTY above a rising 50-day average)",
           "FALLING": "in a FALLING market (NIFTY below a falling 50-day average)",
           "SIDEWAYS": "in a SIDEWAYS market (neither rising nor falling)",
           "HIGH_VOL": "in a HIGH-VOLATILITY market (India VIX in the top quarter of its last year)"}


MODE_TEXT = {"cont": "trade with it", "fade": "trade against it", "lowvol_fade": "at most half the normal (no news: fade)",
             "hivol_go": "at least 3x the normal (news: follow)", "rev": "buy (reversal)", "cont_short": "sell",
             "go": "follow", "rebound": "rebound", "on": "on", "pre": "the session before"}


def _fmt_params(b: Block, params: dict) -> str:
    t = b.text
    for k, v in params.items():
        if isinstance(v, float):
            t = t.replace(f"{{{k}:%}}", f"{v * 100:g}%")
        if k == "mode":
            t = t.replace("{mode}", MODE_TEXT.get(str(v), str(v)))
        if k == "when":
            t = t.replace("{when}", MODE_TEXT.get(str(v), str(v)))
    try:
        return t.format(**params)
    except Exception:
        return t


def _exit_text(c: Candidate) -> str:
    e = c.exit
    k = e[0]
    if k == "time":
        return f"exit at {HHMM[e[1]]}"
    if k == "time_min":
        return f"exit {e[1]} minutes after entry"
    if k == "bracket":
        return f"stop {e[1]:.2%} / target {e[1] * e[2]:.2%} from entry (1:{e[2]:g}), else exit at 15:15"
    if k == "bracket_min":
        return f"stop {e[1]:.2%} / target {e[1] * e[2]:.2%} (1:{e[2]:g}), else exit {e[3]} minutes after entry"
    if k == "signal":
        return "exit when a 1-minute close crosses back through the session VWAP, else at 15:15"
    if k == "trail":
        return f"trailing stop {e[1]:.2%} behind the best price since entry, else exit at 15:15"
    if k == "trail_min":
        return f"trailing stop {e[1]:.2%} behind the best price since entry, else exit {e[2]} minutes after entry"
    if k == "trail_be":
        return (f"stop {e[1]:.2%}; after a {e[1]:.2%} move in favour the stop goes to breakeven and trails {e[1]:.2%} "
                f"behind the best price, else exit at 15:15")
    if k == "trail_be_min":
        return (f"stop {e[1]:.2%}; after a {e[1]:.2%} move in favour the stop goes to breakeven and trails {e[1]:.2%} "
                f"behind the best price, else exit {e[2]} minutes after entry")
    if k in ATR_KINDS:
        be = "trail_be" in k
        lim = f"exit {e[2]} minutes after entry" if k.endswith("_min") else "exit at 15:15"
        d = f"{e[1]:g} x ATR(14)"
        if be:
            return (f"stop {d}; after a {d} move in favour the stop goes to breakeven and trails {d} behind the best "
                    f"price, else {lim}")
        return f"trailing stop {d} behind the best price since entry, else {lim}"
    if k == "signal_min":
        return f"exit when a 1-minute close crosses back through the session VWAP, at the latest {e[2]} minutes after entry"
    if k == "hold":
        return f"hold {e[1]} session{'s' if e[1] > 1 else ''}, exit at the close"
    if k == "bracket_d":
        return (f"stop {e[1]:.0%} / target {e[1] * e[2]:.0%} on the daily range (1:{e[2]:g}), at the latest after "
                f"{e[3]} sessions")
    if k == "signal_d":
        return f"exit at the next open after a close through its 10-day average, at the latest after {e[2]} sessions"
    if k == "event":
        return "exit at the close of the session before the ex-date"
    if k == "overnight":
        return "hold overnight, exit at the next 09:15 auction"
    if k == "hold_m":
        return f"exit at 15:21 {e[1]} sessions later"
    if k == "opt_hold":
        return f"close at the official close {e[1]} session{'s' if e[1] > 1 else ''} later (or settle at expiry)"
    if k == "opt_expiry":
        return "hold to expiry (settled at intrinsic value)"
    if k == "opt_bracket":
        return ("debit: stop at -50 % / target at +150 % of the premium (1:3); credit: stop when the value doubles / "
                "target at half the credit; checked on official closes")
    if k == "opt_signal":
        return ("close when NIFTY's close crosses its 10-day average against the position" if e[1] == "sma10"
                else "close when India VIX falls below the median of its last year")
    return str(e)


def describe(c: Candidate) -> str:
    """Plain English, generated from the grammar (no free text)."""
    b = BLOCK[c.block]
    p = c.kw
    cond = _fmt_params(b, p)
    side = _side_word(b, p)
    if c.agent == "options":
        st, exp = c.structure.split(":")
        what = {"long_atm": "buy the at-the-money call (put if the signal is bearish)",
                "debit_spread": "buy a 2 %-wide debit spread in the signal's direction (bull call / bear put)",
                "short_strangle": "sell a 4 %-out-of-the-money strangle",
                "iron_condor": "sell a 4 % strangle with 3 % wings (iron condor)"}[st]
        lots = f" ({OPTION_SIZE_TEXT[c.size]})" if c.size else ""
        return (f"{c.instrument} options {MT_TEXT[c.mtype]}: when {cond}, {what} on the {exp} expiry at the next "
                f"session's official close (strikes from the previous session's real chain){lots}; {_exit_text(c)}.")
    inst = {"STOCKS": "stock", "NIFTY": "NIFTY future", "BANKNIFTY": "BANKNIFTY future"}[c.instrument]
    if c.universe and c.instrument == "STOCKS":
        inst = f"stock (universe: {UNIVERSE_TEXT[c.universe]})"
    dirn = {"long": "buy", "short": "sell short", "with": "trade", "against": "trade"}.get(side, side)
    tail = {"with": " in the direction of the move", "against": " against the move"}.get(side, "")
    if c.agent in ("intraday", "scalping"):
        when = ("the previous evening's data show that " + cond) if c.grid == "prev_eod" else cond
        prod = "MIS"
        ent = f"at {HHMM[c.entry]}"
    elif c.entry in ENTRY_MIN:
        when, prod, ent = cond, "futures", f"at {HHMM[c.entry]}"
    else:
        when = cond
        prod = {"cnc/fut": "longs in delivery (CNC), shorts in stock futures", "fut": "futures"}.get(c.product,
                                                                                                    c.product)
        ent = "at the next session's open"
    if c.size and c.agent == "swing":        # delivery longs keep Rs 1 lakh; only the futures shorts are re-sized
        prod = ("longs in delivery (CNC, Rs 1 lakh per trade), shorts in stock futures (Rs 5 lakh per trade); "
                "Rs 10 lakh capital per agent")
    elif c.size:
        prod = f"{prod}; {SIZE_TEXT[c.size]}"
    return (f"{c.agent.capitalize()} {MT_TEXT[c.mtype]}: when {when}, {dirn} the {inst}{tail} {ent} ({prod}); "
            f"{_exit_text(c)}.")


def grammar_sha256(version: str | None = None) -> str:
    """Fingerprint of the whole grammar (every agent's list in order) of a version (default: the current one)."""
    version = version or GRAMMAR_VERSION
    h = hashlib.sha256(version.encode())
    for a in AGENTS:
        for row in grammar(a, version):
            h.update(repr(row).encode())
    return h.hexdigest()


def proven_rule_twins() -> list:
    """The bg2 sentences that are the two proven lab rules widened to NIFTY200 (one per market type each): the 09:15
    auction gap fade of >= 2 % / >= 3 % exited at 09:20 (scalping, AUC-FADE-2PCT / -3PCT) and the 3x-volume up-day
    overnight long (futures carrier, SWING-ATTN-ON-3X). ORDINARY members of the grammar: drawn when the hash order
    reaches them, judged like every other candidate, no credit from their NIFTY-100 proof. (Listed for tests and
    reports only; the draw never looks at this list.)"""
    out = []
    for mt in MARKET_TYPES:
        for g in (0.02, 0.03):
            out.append(Candidate("scalping", mt, "gap_fade", (("g", g), ("mode", "fade_day")), "o0915",
                                 ("time_min", 5), "STOCKS", "mis", "", "NIFTY200"))
        out.append(Candidate("futures", mt, "attention_swing", (("day", "up"),), "o1521", ("overnight",), "STOCKS",
                             "fut", "", "NIFTY200"))
    return out
