"""
stratlab/builder_grammar.py — the GRAMMAR of the automatic strategy builder (builder/v1). Pure: no data, no returns.

A candidate is one sentence of the grammar:
    agent x market type x signal block (+ its round parameters) x entry timing x exit rule x instrument
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
ordered by sha256(GRAMMAR_VERSION | batch id | cell | candidate key); a batch takes, in that order, the first 20 whose
key was never drawn in an earlier batch and that is not a near-duplicate of one already taken in this batch.
NEAR-DUPLICATE (fixed): same agent, market type, block, exit kind and instrument, and at most ONE other field of the
signature (each block parameter, the entry time, each exit parameter) differs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

GRAMMAR_VERSION = "bg1"
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
BLOCK = {b.name: b for b in BLOCKS}

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

    @property
    def key(self) -> str:
        """The identity across batches (never re-tested)."""
        return json.dumps([GRAMMAR_VERSION, self.agent, self.mtype, self.block, [list(p) for p in self.params],
                           self.entry, list(self.exit), self.instrument, self.product, self.structure],
                          separators=(",", ":"))

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
        return tuple(v for _k, v in self.params) + (self.entry, self.structure) + tuple(self.exit[1:])

    def to_json(self) -> dict:
        return {"agent": self.agent, "mtype": self.mtype, "block": self.block, "params": [list(p) for p in self.params],
                "entry": self.entry, "exit": list(self.exit), "instrument": self.instrument, "product": self.product,
                "structure": self.structure}

    @staticmethod
    def from_json(d: dict) -> "Candidate":
        return Candidate(d["agent"], d["mtype"], d["block"], tuple(tuple(p) for p in d["params"]), d["entry"],
                         tuple(d["exit"]), d["instrument"], d["product"], d.get("structure", ""))


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


def grammar(agent: str) -> list:
    """Every sentence of one agent's grammar, without the market type (a fixed order)."""
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


def cell_grammar(agent: str, mtype: str) -> list:
    return [Candidate(agent, mtype, blk, p, en, ex, inst, prod, st)
            for inst, prod, blk, p, en, ex, st in grammar(agent)]


def grammar_sizes() -> dict:
    return {a: len(grammar(a)) for a in AGENTS}


def near_duplicate(a: Candidate, b: Candidate) -> bool:
    """The fixed similarity rule (module docstring)."""
    if (a.agent, a.mtype, a.block, a.exit[0], a.instrument) != (b.agent, b.mtype, b.block, b.exit[0], b.instrument):
        return False
    sa, sb = a.signature(), b.signature()
    if len(sa) != len(sb):
        return False
    return sum(1 for x, y in zip(sa, sb) if x != y) <= 1


def _order_key(batch_id: str, cell: str, key: str) -> str:
    return hashlib.sha256(f"{GRAMMAR_VERSION}|{batch_id}|{cell}|{key}".encode()).hexdigest()


def draw_batch(batch_id: str, drawn_keys, per_cell: int = PER_CELL) -> list:
    """The batch's candidates: per (agent, market type) cell the first `per_cell` of the hash-ordered grammar list that
    were never drawn before (drawn_keys) and are not near-duplicates of one already taken. Deterministic."""
    drawn = set(drawn_keys)
    out = []
    for agent in AGENTS:
        for mt in MARKET_TYPES:
            pool = cell_grammar(agent, mt)
            pool.sort(key=lambda c: _order_key(batch_id, c.cell, c.key))
            taken = []
            for c in pool:
                if len(taken) >= per_cell:
                    break
                if c.key in drawn:
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
        return (f"{c.instrument} options {MT_TEXT[c.mtype]}: when {cond}, {what} on the {exp} expiry at the next "
                f"session's official close (strikes from the previous session's real chain); {_exit_text(c)}.")
    inst = {"STOCKS": "stock", "NIFTY": "NIFTY future", "BANKNIFTY": "BANKNIFTY future"}[c.instrument]
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
    return (f"{c.agent.capitalize()} {MT_TEXT[c.mtype]}: when {when}, {dirn} the {inst}{tail} {ent} ({prod}); "
            f"{_exit_text(c)}.")


def grammar_sha256() -> str:
    """Fingerprint of the whole grammar (every agent's list in order)."""
    h = hashlib.sha256(GRAMMAR_VERSION.encode())
    for a in AGENTS:
        for row in grammar(a):
            h.update(repr(row).encode())
    return h.hexdigest()
