"""
strategies/phase1_backtest.py
Phase-1 backtest + validation for the positional strategies, per the
spec's "Backtesting & Validation Methodology" section:

  1. Daily-bar event simulation of the ACTUAL strategy classes with
     next-bar execution lag, per-side costs, spec sizing formulas and a
     low-leverage cap.
  2. Out-of-sample hold-out: the most recent 15% of data is evaluated
     separately; gate = OOS Sharpe > 50% of in-sample Sharpe.
  3. Walk-forward blocks: consecutive out-of-sample windows (parameters
     are fixed by design — no optimization — so walk-forward here
     measures regime consistency, not parameter fit).
  4. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, JPM 40(5)):
     corrects the observed Sharpe for multiple testing, non-normality
     and sample length. Record how many strategy variants you tried and
     pass it as n_trials.
  5. Monte Carlo bootstrap of the trade sequence → drawdown/terminal
     distribution → suggested kill threshold (max DD × 1.5).

Pure python, no numpy — same convention as the rest of the package.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from strategies.base import Action, PositionalStrategy
from strategies.indicators import DailyBar

COST_SIDE_DEFAULT = 0.0005     # 0.05% of traded notional per side
MAX_LEVERAGE      = 2.0        # low-leverage profile: notional ≤ 2× equity
EVAL_WINDOW       = 320        # bars passed to evaluate() (max lookback ~253)
WARMUP            = 310
TRADING_DAYS      = 252


# ── Event simulator ────────────────────────────────────────────────────────

@dataclass
class SimResult:
    eq_curve: list[float] = field(default_factory=list)
    trades:   list[float] = field(default_factory=list)   # per-round-trip return
    years:    float = 0.0


def _unit_size(name: str, sig, equity: float, price: float) -> float:
    """Spec sizing: Turtle 1%/N (donchian), 1%/(2.5×ATR) (ma_crossover),
    20%-of-equity vol-target notional (tsmom)."""
    if name == "tsmom":
        if not sig.weight:
            return 0.0
        vol = 1.0 / sig.weight
        return equity * 0.20 / (vol * price) if vol > 0 else 0.0
    n = sig.n_atr or 0.0
    if n <= 0:
        return 0.0
    stop_mult = 2.5 if name == "ma_crossover" else 1.0
    return equity * 0.01 / (n * stop_mult)


def simulate(strategy_factory, name: str, bars: list[DailyBar],
             cost_side: float = COST_SIDE_DEFAULT,
             max_leverage: float = MAX_LEVERAGE) -> SimResult:
    """Run one strategy over one instrument. Signals fire on bar i and
    fill at bar i+1's close (conservative vs next-open execution)."""
    strat: PositionalStrategy = strategy_factory()
    res = SimResult()
    equity, qty, entry_eq = 1.0, 0.0, 1.0
    pending = None

    def _trade(dq: float, price: float) -> None:
        nonlocal equity, qty, entry_eq
        equity -= abs(dq) * price * cost_side
        if qty == 0:
            entry_eq = equity
        qty += dq
        if abs(qty) < 1e-12:
            qty = 0.0
            res.trades.append(equity / entry_eq - 1.0)

    for i in range(WARMUP, len(bars)):
        close, prev = bars[i].close, bars[i - 1].close
        if qty:
            equity += qty * (close - prev)
        res.eq_curve.append(equity)
        if equity <= 0.05:
            break

        if pending:
            act, sig = pending
            pending = None
            if act in (Action.ENTER_LONG, Action.ENTER_SHORT, Action.PYRAMID):
                q = _unit_size(name, sig, equity, close)
                room = max(0.0, max_leverage * equity / close - abs(qty))
                q = min(q, room)
                if act == Action.ENTER_SHORT or (act == Action.PYRAMID and qty < 0):
                    q = -q
                if q:
                    _trade(q, close)
            elif act == Action.EXIT and qty:
                _trade(-qty, close)

        sig = strat.evaluate("SYM", bars[max(0, i - EVAL_WINDOW):i + 1])
        if sig.action != Action.HOLD:
            pending = (sig.action, sig)

    if qty:
        _trade(-qty, bars[-1].close)
    if len(bars) > WARMUP:
        res.years = max((bars[-1].date - bars[WARMUP].date).days / 365.25, 1e-9)
    return res


# ── Metrics ────────────────────────────────────────────────────────────────

def daily_rets(eq: list[float]) -> list[float]:
    return [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]


def sharpe(eq: list[float]) -> float:
    rets = daily_rets(eq)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return mean / math.sqrt(var) * math.sqrt(TRADING_DAYS) if var > 0 else 0.0


def max_drawdown(eq: list[float]) -> float:
    peak, dd = float("-inf"), 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            dd = max(dd, 1 - v / peak)
    return dd


def metrics(res: SimResult) -> dict:
    eq = res.eq_curve
    if len(eq) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0,
                "win_rate": 0.0, "expectancy": 0.0}
    total = eq[-1] / eq[0]
    wins = [t for t in res.trades if t > 0]
    return {
        "cagr":       total ** (1 / res.years) - 1 if total > 0 else -1.0,
        "sharpe":     sharpe(eq),
        "max_dd":     max_drawdown(eq),
        "trades":     len(res.trades),
        "win_rate":   len(wins) / len(res.trades) if res.trades else 0.0,
        "expectancy": (sum(res.trades) / len(res.trades)) if res.trades else 0.0,
    }


# ── Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ──────────────────

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(rets: list[float]) -> tuple[float, float]:
    """(skewness, kurtosis) of a return series; (0, 3) when degenerate."""
    n = len(rets)
    if n < 4:
        return 0.0, 3.0
    mean = sum(rets) / n
    m2 = sum((r - mean) ** 2 for r in rets) / n
    if m2 <= 0:
        return 0.0, 3.0
    m3 = sum((r - mean) ** 3 for r in rets) / n
    m4 = sum((r - mean) ** 4 for r in rets) / n
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def deflated_sharpe(eq: list[float], n_trials: int,
                    sr_var_across_trials: float = 0.25) -> dict:
    """Probability that the observed Sharpe exceeds the max Sharpe expected
    from n_trials of pure noise. dsr > 0.5 ≈ the spec's 'DSR > 0'; > 0.95
    is strong evidence the Sharpe is not a selection artifact."""
    rets = daily_rets(eq)
    T = len(rets)
    if T < 10:
        return {"dsr": 0.0, "sr_daily": 0.0, "sr_benchmark": 0.0}
    mean = sum(rets) / T
    var = sum((r - mean) ** 2 for r in rets) / (T - 1)
    sr = mean / math.sqrt(var) if var > 0 else 0.0     # daily (non-annualized)
    skew, kurt = _moments(rets)

    euler = 0.5772156649015329
    n = max(n_trials, 1)
    sr0 = 0.0
    if n > 1:
        sd_sr = math.sqrt(max(sr_var_across_trials, 1e-12) / TRADING_DAYS)
        sr0 = sd_sr * ((1 - euler) * normal_ppf(1 - 1.0 / n)
                       + euler * normal_ppf(1 - 1.0 / (n * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr
    if denom <= 0:
        return {"dsr": 0.0, "sr_daily": sr, "sr_benchmark": sr0}
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)
    return {"dsr": normal_cdf(z), "sr_daily": sr, "sr_benchmark": sr0}


# ── Monte Carlo bootstrap of the trade sequence ────────────────────────────

def bootstrap_drawdowns(trade_returns: list[float], n_sims: int = 1000,
                        seed: int = 42) -> dict:
    """Resample the trade sequence with replacement; return the drawdown
    and terminal-return distribution (worst-case understanding → set the
    live kill threshold from this, not from the single realized path)."""
    if not trade_returns:
        return {"dd_p50": 0.0, "dd_p95": 0.0, "dd_worst": 0.0,
                "ret_p5": 0.0, "ret_p50": 0.0}
    rng = random.Random(seed)
    dds, finals = [], []
    k = len(trade_returns)
    for _ in range(n_sims):
        eq, peak, dd = 1.0, 1.0, 0.0
        for _ in range(k):
            eq *= 1.0 + rng.choice(trade_returns)
            peak = max(peak, eq)
            dd = max(dd, 1 - eq / peak)
        dds.append(dd)
        finals.append(eq - 1.0)
    dds.sort()
    finals.sort()

    def pct(sorted_vals, p):
        return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)]

    return {"dd_p50": pct(dds, 0.50), "dd_p95": pct(dds, 0.95),
            "dd_worst": dds[-1], "ret_p5": pct(finals, 0.05),
            "ret_p50": pct(finals, 0.50)}


# ── Walk-forward & hold-out ────────────────────────────────────────────────

def holdout_split(bars: list[DailyBar], holdout_frac: float = 0.15
                  ) -> tuple[list[DailyBar], list[DailyBar]]:
    """(in_sample, full_series_for_oos). The OOS run uses the full series
    but only the last holdout_frac of the equity curve is scored, so the
    strategy enters the hold-out with realistic warm state."""
    cut = int(len(bars) * (1 - holdout_frac))
    return bars[:cut], bars


def walk_forward(strategy_factory, name: str, bars: list[DailyBar],
                 block_years: float = 2.0) -> list[dict]:
    """Metrics per consecutive out-of-sample block (fixed parameters —
    consistency check across regimes, not an optimization loop)."""
    block = int(block_years * TRADING_DAYS)
    out = []
    start = WARMUP
    while start + block // 2 < len(bars):
        end = min(start + block, len(bars))
        seg = bars[max(0, start - WARMUP):end]
        res = simulate(strategy_factory, name, seg)
        m = metrics(res)
        m["from"] = bars[start].date.isoformat()
        m["to"]   = bars[end - 1].date.isoformat()
        out.append(m)
        start = end
    return out


# ── Phase-1 report with go/no-go gates ─────────────────────────────────────

def phase1_report(strategy_factory, name: str, bars: list[DailyBar],
                  n_trials: int = 9) -> dict:
    """Full spec Phase-1 evaluation. Gates:
      DSR > 0.5 ('DSR > 0'), OOS Sharpe > 50% of in-sample,
      ≥ 30 trades (min sample), plus the Monte Carlo DD distribution and
      the suggested live kill threshold (realized max DD × 1.5)."""
    full = simulate(strategy_factory, name, bars)
    m_full = metrics(full)

    is_bars, oos_bars = holdout_split(bars)
    m_is = metrics(simulate(strategy_factory, name, is_bars))
    oos_res = simulate(strategy_factory, name, oos_bars)
    oos_tail = oos_res.eq_curve[len(is_bars) - WARMUP:]
    oos_sharpe = sharpe(oos_tail) if len(oos_tail) > 10 else 0.0

    dsr = deflated_sharpe(full.eq_curve, n_trials)
    mc = bootstrap_drawdowns(full.trades)
    wf = walk_forward(strategy_factory, name, bars)
    wf_positive = sum(1 for b in wf if b["sharpe"] > 0)

    gates = {
        "dsr_positive":    dsr["dsr"] > 0.5,
        "oos_half_of_is":  oos_sharpe > 0.5 * m_is["sharpe"] if m_is["sharpe"] > 0
                           else oos_sharpe > 0,
        "min_30_trades":   m_full["trades"] >= 30,
        "wf_majority_positive": wf_positive * 2 > len(wf) if wf else False,
    }
    return {
        "strategy":       name,
        "full":           m_full,
        "in_sample":      m_is,
        "oos_sharpe":     oos_sharpe,
        "dsr":            dsr,
        "monte_carlo":    mc,
        "walk_forward":   wf,
        "gates":          gates,
        "go":             all(gates.values()),
        "kill_dd_threshold": round(m_full["max_dd"] * 1.5, 4),
    }
