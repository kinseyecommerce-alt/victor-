"""
shadow_registry.py — frozen shadow-incubation registry: forward testing that cannot be mined.

A *shadow* is a strategy that is watched on live market data, in PAPER, after it was frozen. Each shadow has
  id, strategy reference, frozen config + sha256 config hash, registration date, and its own append-only,
  hash-chained JSONL ledger of signals and fills (every record carries the previous record's hash, so an edit,
  deletion or reordering anywhere in the file breaks verify_chain()).

The registry itself (registrations, retirements, look results, promotions, demotions, ledger checkpoints) is a
second hash-chained JSONL file. A checkpoint pins each ledger's head (seq, hash), so truncating a ledger's tail is
detected as well.

What makes it hard to mine
  * the config is frozen: re-registering an id with a different config raises FrozenConfigError — a changed
    strategy is a NEW shadow (new id, new trials-ledger row, one more test in the Bonferroni family);
  * forward only: a trade is accepted only when it was entered after registration (exit date >= registration
    date + 1 + max_hold_days); there is no backfill;
  * the promotion rule (PROMOTION_RULE) is written here BEFORE any forward result exists, and its hash is stored in
    every registration, so a later change of the rule is visible;
  * a fixed number of looks at pre-registered trade counts, each evaluated exactly once and written to the
    registry; the significance level is Bonferroni-split over a FIXED family — every shadow ever registered
    under the rule version (demoted, retired and drifted ones included) — AND the looks, so demoting or retiring
    losers can never loosen the threshold of the survivors.

PAPER only. Nothing in this module places, modifies or cancels an order, or touches an agent's book: it reads the
forward test's ledger (forward_test.py) and the performance ledger rows of the validated agents and copies them.
A promotion is a recommendation recorded in the registry; it never changes trading behaviour by itself.
Scheduler wiring (platform_scheduler) exists only with SHADOW_REGISTRY_ENABLED=true (default false).
See SHADOW_REGISTRY.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

GENESIS_PREV = "0" * 64

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED PROMOTION / DEMOTION RULE — written 2026-11-14, before any shadow had a single forward trade.
# Do not tune it on results. Any change must bump "version"; its hash is stored in every registration.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
PROMOTION_RULE: dict = {
    "version": "shadow-promotion/v2",   # v2 (2026-11-14, before any forward trade): fixed Bonferroni family
    "written": "2026-11-14",
    # promotion: evaluated only at these looks (cumulative forward trade counts); each look runs exactly once
    "looks_trades": (50, 100, 200),
    "min_forward_trades": 50,            # = the first look: no promotion is possible before this many trades
    "min_forward_days": 20,              # and at least this many distinct trading days in the look sample
    "family_alpha": 0.05,                # family-wise error rate over (registered shadows × looks)
    "test": "one-sided bootstrap of the mean daily forward net P&L; H0: mean <= 0; daily totals resampled "
            "with replacement after centring on the observed mean; p = (1 + #{boot mean >= observed}) / (B + 1)",
    "bootstrap_resamples": 20000,
    "bootstrap_seed": "first 8 bytes of sha256('<shadow id>|look<k>')",
    "threshold": "p <= family_alpha / (n_family × len(looks_trades)); n_family = number of shadows ever "
                 "registered under this rule version (incl. demoted, retired, drifted); never decreases",
    # demotion: checked at every evaluation (no multiple-testing cost: it only removes)
    "demotion_dd_multiple": 1.5,         # forward max drawdown (daily totals) > 1.5 × backtest max drawdown
    "demotion_reference": "backtest max drawdown of daily net totals over the lab period, at lab size; forward "
                          "P&L normalised to lab size when the config gives per_trade_notional",
}


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_hex(obj) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


RULE_HASH = sha256_hex(PROMOTION_RULE)


def config_hash(config: dict) -> str:
    """sha256 of the canonical JSON of a frozen config."""
    return sha256_hex(config)


class FrozenConfigError(ValueError):
    """An id was registered again with a different config (a changed strategy must get a new id)."""


class ChainError(ValueError):
    pass


# ── paper fill model (the same as forward_test / the validated agents) ─────────────────────────────────────────
HALF_SPREAD_BPS = 2.1          # forward_test.SLIPPAGE_BPS / stratlab.auction: fallback when there is no touch


def paper_fill(side: str, quote: Optional[dict] = None, ref_px: float = 0.0,
               half_spread_bps: float = HALF_SPREAD_BPS) -> float:
    """BUY at the ask / SELL at the bid when the quote has a touch; otherwise the reference price (quote LTP, or
    ref_px) ± half_spread_bps — forward_test's auction ± 2.1 bps. Raises when no price is known."""
    q = quote or {}
    touch = float((q.get("ask") if side == "BUY" else q.get("bid")) or 0.0)
    if touch > 0:
        return touch
    ref = float(ref_px or q.get("ltp") or q.get("last_price") or 0.0)
    if ref <= 0:
        raise ValueError("no price for a paper fill")
    s = half_spread_bps / 1e4
    return ref * (1 + s) if side == "BUY" else ref * (1 - s)


# ── hash-chained JSONL ───────────────────────────────────────────────────────────────────────────────────────
def _record_hash(rec: dict) -> str:
    return sha256_hex({k: v for k, v in rec.items() if k != "hash"})


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def verify_chain(path, expect_head: Optional[tuple[int, str]] = None) -> dict:
    """Check a hash-chained JSONL file. Returns {"ok", "records", "head": (seq, hash) | None, "error", "bad_seq"}.
    Detects an edited field, a deleted / inserted / reordered line and a broken first link; with expect_head
    (a checkpoint) also a truncated tail."""
    path = Path(path)
    prev, n, head = GENESIS_PREV, 0, None
    for i, ln in enumerate(_read_lines(path)):
        try:
            rec = json.loads(ln)
        except Exception:
            return {"ok": False, "records": n, "head": head, "bad_seq": i, "error": f"line {i}: not JSON"}
        if rec.get("seq") != i:
            return {"ok": False, "records": n, "head": head, "bad_seq": i,
                    "error": f"line {i}: seq {rec.get('seq')} (a line was deleted, inserted or moved)"}
        if rec.get("prev_hash") != prev:
            return {"ok": False, "records": n, "head": head, "bad_seq": i, "error": f"line {i}: prev_hash mismatch"}
        if _record_hash(rec) != rec.get("hash"):
            return {"ok": False, "records": n, "head": head, "bad_seq": i,
                    "error": f"line {i}: hash mismatch (record edited)"}
        prev, n, head = rec["hash"], n + 1, (i, rec["hash"])
    if expect_head is not None:
        seq, h = expect_head
        lines = _read_lines(path)
        if seq >= len(lines):
            return {"ok": False, "records": n, "head": head, "bad_seq": seq,
                    "error": f"truncated: checkpoint seq {seq} beyond the last record"}
        if json.loads(lines[seq]).get("hash") != h:
            return {"ok": False, "records": n, "head": head, "bad_seq": seq, "error": f"checkpoint seq {seq} differs"}
    return {"ok": True, "records": n, "head": head, "bad_seq": None, "error": None}


class ChainedLog:
    """Append-only hash-chained JSONL file (one writer process; a lock for threads)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def records(self) -> list[dict]:
        return [json.loads(ln) for ln in _read_lines(self.path)]

    def _tail(self) -> tuple[int, str]:
        lines = _read_lines(self.path)
        if not lines:
            return -1, GENESIS_PREV
        last = json.loads(lines[-1])
        return int(last["seq"]), str(last["hash"])

    def append(self, kind: str, data: dict, ts: Optional[str] = None) -> dict:
        with self._lock:
            seq, prev = self._tail()
            rec = {"seq": seq + 1, "ts": ts or _now_iso(), "kind": kind, "data": data, "prev_hash": prev}
            rec["hash"] = _record_hash(rec)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canon(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return rec

    def verify(self, expect_head=None) -> dict:
        return verify_chain(self.path, expect_head)


def _now_iso() -> str:
    try:
        from ist_clock import now_ist
        return now_ist().isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat(timespec="seconds")


def _today() -> date:
    try:
        from ist_clock import now_ist
        return now_ist().date()
    except Exception:
        return date.today()


# ── statistics: bootstrap + Bonferroni (pure) ────────────────────────────────────────────────────────────────
def bootstrap_pvalue(daily_nets: Iterable[float], resamples: int = PROMOTION_RULE["bootstrap_resamples"],
                     seed: int = 0) -> float:
    """One-sided bootstrap p-value of H0: mean daily net <= 0 (centred resampling)."""
    import numpy as np
    x = np.asarray(list(daily_nets), dtype=float)
    if x.size < 2:
        return 1.0
    obs = float(x.mean())
    y = x - obs
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = 2000
    done = 0
    while done < resamples:
        b = min(chunk, resamples - done)
        idx = rng.integers(0, x.size, size=(b, x.size))
        hits += int((y[idx].mean(axis=1) >= obs).sum())
        done += b
    return (1 + hits) / (resamples + 1)


def bonferroni_threshold(n_family: int, alpha: float = PROMOTION_RULE["family_alpha"],
                         n_looks: int = len(PROMOTION_RULE["looks_trades"])) -> float:
    return alpha / (max(1, int(n_family)) * max(1, int(n_looks)))


def look_seed(shadow_id: str, look: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{shadow_id}|look{look}".encode()).digest()[:8], "big")


def daily_totals(trades: list[dict], key: str = "net") -> list[tuple[str, float]]:
    out: dict[str, float] = {}
    for t in trades:
        out[t["day"]] = out.get(t["day"], 0.0) + float(t.get(key, 0.0))
    return sorted(out.items())


def max_drawdown(values: Iterable[float]) -> float:
    cum = peak = dd = 0.0
    for v in values:
        cum += v
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return round(dd, 2)


def look_sample(trades: list[dict], n_trades: int, min_days: int) -> Optional[list[dict]]:
    """The deterministic sample of a look: all trades up to and including the first trading day at which the
    cumulative trade count reaches n_trades AND the sample spans min_days days; None until then."""
    by_day: dict[str, list[dict]] = {}
    for t in trades:
        by_day.setdefault(t["day"], []).append(t)
    sample: list[dict] = []
    keys: set = set()
    for i, d in enumerate(sorted(by_day), start=1):
        sample.extend(by_day[d])
        keys.update(t["trade_key"] for t in by_day[d])
        if len(keys) >= n_trades and i >= min_days:
            return sample
    return None


def judge_look(shadow_id: str, look: int, sample: list[dict], n_family: int,
               rule: dict = PROMOTION_RULE) -> dict:
    """One look's verdict (pure): bootstrap p of the mean daily net vs the Bonferroni threshold."""
    days = daily_totals(sample)
    nets = [v for _, v in days]
    p = bootstrap_pvalue(nets, rule["bootstrap_resamples"], look_seed(shadow_id, look))
    thr = bonferroni_threshold(n_family, rule["family_alpha"], len(rule["looks_trades"]))
    mean = sum(nets) / len(nets) if nets else 0.0
    return {"look": look, "trades": len({t["trade_key"] for t in sample}), "days": len(nets),
            "net": round(sum(nets), 2), "mean_daily_net": round(mean, 2), "p_value": round(p, 6),
            "n_family": int(n_family), "threshold": thr, "first_day": days[0][0] if days else None,
            "last_day": days[-1][0] if days else None, "promote": bool(mean > 0 and p <= thr)}


# ── spec ─────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ShadowSpec:
    id: str
    family: str
    strategy_ref: str              # "module:object" / "forward_test:AUC-FADE-3PCT/live" — what is being watched
    source: str                    # "forward_test" | "validated_agent" | "manual"
    config: dict = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)


# ── trials ledger hook ───────────────────────────────────────────────────────────────────────────────────────
_LAB_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"   # as stratlab
TRIALS_COLUMNS = ("id", "family", "protocol", "kind", "registered", "config_hash", "source", "note")


def trials_csv_path() -> Optional[Path]:
    p = os.environ.get("SHADOW_TRIALS_CSV")
    if p:
        return Path(p)
    base = os.environ.get("STRATLAB_CACHE") or os.path.join(_LAB_SCRATCH, "stratlab_cache")
    return Path(base) / "trials.csv"


def append_trial(spec: ShadowSpec, registered: str, path: Optional[Path] = None) -> bool:
    """Append one row (protocol=shadow) to the lab's trials ledger IF that file exists (optional; never creates
    it). Unknown columns of an existing header are left blank; a row with the same id + protocol is not repeated."""
    path = Path(path) if path is not None else trials_csv_path()
    if path is None or not path.exists():
        return False
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header = rows[0] if rows else list(TRIALS_COLUMNS)
        idx = {c: i for i, c in enumerate(header)}
        for r in rows[1:]:
            if r and r[idx.get("id", 0)] == spec.id and ("protocol" not in idx or
                                                          (len(r) > idx["protocol"] and r[idx["protocol"]] == "shadow")):
                return False
        vals = {"id": spec.id, "family": spec.family, "protocol": "shadow", "kind": spec.config.get("kind", ""),
                "registered": registered, "config_hash": spec.config_hash, "source": spec.source,
                "note": f"shadow incubation; rule {PROMOTION_RULE['version']}"}
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if not rows:
                w.writerow(header)
            w.writerow([vals.get(c, "") for c in header])
        return True
    except Exception as exc:
        logger.warning("[shadow] trials ledger {} not updated: {}", path, exc)
        return False


# ── the registry ─────────────────────────────────────────────────────────────────────────────────────────────
def default_root() -> Path:
    p = os.getenv("SHADOW_REGISTRY_DIR")
    if p:
        return Path(p)
    perf = os.getenv("PERFORMANCE_FILE")
    base = Path(perf).parent if perf else Path(__file__).resolve().parent / "data"
    return base / "shadow"


class ShadowRegistry:
    ACTIVE = ("incubating", "promotion_eligible")

    def __init__(self, root=None, rule: dict = PROMOTION_RULE) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.rule = rule
        self.index = ChainedLog(self.root / "registry.jsonl")

    # ── registry state (folded from the registry chain) ──
    def ledger(self, shadow_id: str) -> ChainedLog:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in shadow_id)
        return ChainedLog(self.root / "ledgers" / f"{safe}.jsonl")

    def entries(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for rec in self.index.records():
            d, k = rec["data"], rec["kind"]
            sid = d.get("id")
            if k == "register":
                out[sid] = {**d, "status": "incubating", "looks": {}, "checkpoint": None, "status_at": rec["ts"]}
            elif sid in out:
                e = out[sid]
                if k == "look":
                    e["looks"][str(d["look"])] = d
                elif k == "status":
                    e["status"], e["status_at"], e["status_reason"] = d["status"], rec["ts"], d.get("reason")
                elif k == "checkpoint":
                    e["checkpoint"] = (int(d["seq"]), d["hash"])
        return out

    def active_ids(self) -> list[str]:
        return [i for i, e in self.entries().items() if e["status"] in self.ACTIVE]

    def family_ids(self) -> list[str]:
        """The Bonferroni family: every shadow ever registered under this rule version, whatever its status
        (demoted, retired and config-drifted shadows stay in). Only grows, so it cannot be shrunk by results."""
        v = self.rule["version"]
        return [i for i, e in self.entries().items() if e.get("rule_version") == v]

    # ── registration ──
    def register(self, spec: ShadowSpec, registered_on: Optional[date] = None) -> dict:
        """Register (idempotent for the same config). A different config under the same id raises."""
        cur = self.entries().get(spec.id)
        h = spec.config_hash
        if cur is not None:
            if cur["config_hash"] != h:
                raise FrozenConfigError(f"{spec.id}: config hash {h[:12]} differs from the frozen "
                                        f"{cur['config_hash'][:12]} — register the change under a new id")
            return cur
        reg_day = (registered_on or _today()).isoformat()
        hold = int(spec.config.get("max_hold_days", 0) or 0)
        forward_from = (date.fromisoformat(reg_day) + timedelta(days=1 + hold)).isoformat()
        data = {"id": spec.id, "family": spec.family, "strategy_ref": spec.strategy_ref, "source": spec.source,
                "config": spec.config, "config_hash": h, "registered_on": reg_day, "forward_from": forward_from,
                "rule_version": self.rule["version"], "rule_hash": sha256_hex(self.rule)}
        self.index.append("register", data)
        self.ledger(spec.id).append("genesis", {"id": spec.id, "config_hash": h, "registered_on": reg_day,
                                                "forward_from": forward_from, "rule_hash": data["rule_hash"]})
        append_trial(spec, reg_day)
        logger.info("[shadow] registered {} (config {}, forward from {})", spec.id, h[:12], forward_from)
        return self.entries()[spec.id]

    def retire(self, shadow_id: str, reason: str) -> None:
        """Retired shadows leave the active set (no more trades or looks) but stay in the Bonferroni family, so
        retiring losers cannot loosen anyone's threshold. Recorded with its reason in the registry chain."""
        if shadow_id not in self.entries():
            raise KeyError(shadow_id)
        self.index.append("status", {"id": shadow_id, "status": "retired", "reason": reason})

    # ── ledger writes ──
    def _known_keys(self, shadow_id: str, kind: str) -> set:
        return {r["data"].get("key") for r in self.ledger(shadow_id).records() if r["kind"] == kind}

    def record_signal(self, shadow_id: str, key: str, data: dict) -> Optional[dict]:
        e = self._entry(shadow_id)
        if e is None or key in self._known_keys(shadow_id, "signal"):
            return None
        return self.ledger(shadow_id).append("signal", {"key": key, **data})

    def record_trade(self, shadow_id: str, trade: dict) -> Optional[dict]:
        """Append one closed paper trade. Needs key, trade_key, day (exit date, ISO), net; refused before the
        shadow's forward_from date (no backfill) and when the key is already in the ledger."""
        e = self._entry(shadow_id)
        if e is None:
            raise KeyError(shadow_id)
        if e["status"] not in self.ACTIVE:
            return None
        for k in ("key", "trade_key", "day", "net"):
            if k not in trade:
                raise ValueError(f"trade lacks {k}")
        if str(trade["day"]) < e["forward_from"]:
            return None
        if trade["key"] in self._known_keys(shadow_id, "trade"):
            return None
        return self.ledger(shadow_id).append("trade", dict(trade))

    def record_paper_trade(self, shadow_id: str, key: str, day: str, symbol: str, side: str, qty: int,
                           entry_quote: Optional[dict] = None, exit_quote: Optional[dict] = None,
                           entry_ref: float = 0.0, exit_ref: float = 0.0, exchange: str = "NSE",
                           product: str = "MIS") -> Optional[dict]:
        """Book a round trip with the paper fill model (touch, else ± half-spread) and the 2026 Zerodha costs."""
        from costs import trade_cost
        exit_side = "SELL" if side == "BUY" else "BUY"
        entry = paper_fill(side, entry_quote, entry_ref)
        exit_ = paper_fill(exit_side, exit_quote, exit_ref)
        gross = (exit_ - entry) * qty * (1 if side == "BUY" else -1)
        cost = trade_cost(entry, exit_, qty, side, exchange, product, symbol)
        return self.record_trade(shadow_id, {
            "key": key, "trade_key": key, "day": day, "symbol": symbol, "side": side, "qty": int(qty),
            "entry_fill": round(entry, 2), "exit_fill": round(exit_, 2), "gross": round(gross, 2),
            "cost": round(float(cost), 2), "net": round(gross - float(cost), 2),
            "notional": round(entry * qty, 2), "fill_model": f"touch else ±{HALF_SPREAD_BPS} bps"})

    def _entry(self, shadow_id: str) -> Optional[dict]:
        return self.entries().get(shadow_id)

    def trades(self, shadow_id: str) -> list[dict]:
        return [r["data"] for r in self.ledger(shadow_id).records() if r["kind"] == "trade"]

    # ── verification ──
    def verify_all(self) -> dict:
        out = {"registry": self.index.verify()}
        for sid, e in self.entries().items():
            led = self.ledger(sid)
            res = led.verify(e.get("checkpoint"))
            if res["ok"]:
                first = (led.records() or [{}])[0]
                if first.get("kind") != "genesis" or first.get("data", {}).get("config_hash") != e["config_hash"]:
                    res = {**res, "ok": False, "error": "genesis record does not match the registered config"}
            out[sid] = res
        out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
        return out

    def checkpoint(self, shadow_id: str) -> None:
        res = self.ledger(shadow_id).verify()
        if res["ok"] and res["head"] is not None:
            e = self._entry(shadow_id)
            if e.get("checkpoint") != tuple(res["head"]):
                self.index.append("checkpoint", {"id": shadow_id, "seq": res["head"][0], "hash": res["head"][1]})

    # ── evaluation (promotion looks + demotion) ──
    def _dd_units(self, e: dict, trades: list[dict]) -> list[float]:
        unit = (e["config"].get("per_trade_notional") or 0)
        days: dict[str, float] = {}
        for t in trades:
            v = float(t["net"])
            if unit and float(t.get("notional") or 0) > 0:
                v *= unit / float(t["notional"])
            days[t["day"]] = days.get(t["day"], 0.0) + v
        return [days[d] for d in sorted(days)]

    def evaluate(self, shadow_id: str, write: bool = True) -> dict:
        e = self._entry(shadow_id)
        if e is None:
            raise KeyError(shadow_id)
        chain = self.ledger(shadow_id).verify(e.get("checkpoint"))
        trades = self.trades(shadow_id)
        out = {"id": shadow_id, "status": e["status"], "trades": len({t["trade_key"] for t in trades}),
               "days": len({t["day"] for t in trades}), "net": round(sum(float(t["net"]) for t in trades), 2),
               "chain_ok": chain["ok"], "looks": dict(e["looks"]), "new_looks": []}
        if not chain["ok"]:
            out["error"] = f"ledger chain broken: {chain['error']} — not evaluated"
            logger.error("[shadow] {}: {}", shadow_id, out["error"])
            return out
        if e["status"] not in self.ACTIVE:
            return out
        # demotion (every evaluation)
        ref = e["config"].get("backtest_max_dd")
        fdd = max_drawdown(self._dd_units(e, trades))
        out["forward_max_dd"] = fdd
        out["demotion_limit"] = round(self.rule["demotion_dd_multiple"] * float(ref), 2) if ref else None
        if ref and fdd > self.rule["demotion_dd_multiple"] * float(ref):
            reason = f"forward max DD {fdd:,.0f} > {self.rule['demotion_dd_multiple']} × backtest {float(ref):,.0f}"
            if write:
                self.index.append("status", {"id": shadow_id, "status": "demoted", "reason": reason})
            out["status"], out["status_reason"] = "demoted", reason
            return out
        if not ref:
            out["demotion_note"] = "no backtest max drawdown in the frozen config: demotion cannot fire"
        # promotion looks (each once, in order)
        if e["status"] == "incubating":
            n_family = max(1, len(self.family_ids()))
            for k, n in enumerate(self.rule["looks_trades"], start=1):
                if str(k) in e["looks"]:
                    continue
                sample = look_sample(trades, n, self.rule["min_forward_days"])
                if sample is None:
                    break
                res = judge_look(shadow_id, k, sample, n_family, self.rule)
                res["id"] = shadow_id
                out["looks"][str(k)] = res
                out["new_looks"].append(k)
                if write:
                    self.index.append("look", res)
                if res["promote"]:
                    reason = (f"look {k}: p={res['p_value']:.5f} <= {res['threshold']:.5f} "
                              f"({res['trades']} trades, {res['days']} days, net {res['net']:,.0f})")
                    if write:
                        self.index.append("status", {"id": shadow_id, "status": "promotion_eligible",
                                                     "reason": reason})
                    out["status"], out["status_reason"] = "promotion_eligible", reason
                    break
        if write:
            self.checkpoint(shadow_id)
        return out

    def evaluate_all(self, write: bool = True) -> dict:
        return {sid: self.evaluate(sid, write) for sid in list(self.entries())}

    def status(self) -> dict:
        ents = self.entries()
        return {"rule": self.rule, "rule_hash": sha256_hex(self.rule), "n_active": len(self.active_ids()),
                "n_family": len(self.family_ids()), "verify": self.verify_all(),
                "shadows": {sid: {"family": e["family"], "strategy_ref": e["strategy_ref"], "source": e["source"],
                                  "config_hash": e["config_hash"], "registered_on": e["registered_on"],
                                  "forward_from": e["forward_from"], "status": e["status"],
                                  "status_reason": e.get("status_reason"), "looks": e["looks"],
                                  "trades": len({t["trade_key"] for t in self.trades(sid)}),
                                  "net": round(sum(float(t["net"]) for t in self.trades(sid)), 2)}
                            for sid, e in ents.items()}}


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Enrolment: the opening-fade forward test and the three validated agents (observed, never altered)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# backtest max drawdowns of DAILY net totals, lab size, lab period 2023-11..2026-11 (DISCOVERY + FINAL):
#   AUC-FADE-3PCT 24,605.28 / AUC-FADE-2PCT 29,535.33 (dd/auction_final trades, 2.1 bps, 2026-11 costs)
#   SWING-ATTN-ON-3X 12,951.31 (dd/r2_swing_out trades, futures costs)
#   VRP-STRANGLE-016-WINGED 129,760 (dd/vrp/final_run DISCOVERY max_dd, 2 lots; FINAL 104,495)
#   AUC-FADE-1TO3PCT: never run in the lab -> None (demotion cannot fire; flagged in the evaluation)
BACKTEST_MAX_DD = {"AUC-FADE-3PCT": 24605.28, "AUC-FADE-2PCT": 29535.33, "AUC-FADE-1TO3PCT": None,
                   "SWING-ATTN-ON-3X": 12951.31, "VRP-STRANGLE-016-WINGED": 129760.0}
AGENT_MAX_HOLD_DAYS = {"auc_fade": 0, "attn_on": 5, "vrp_winged": 10}   # calendar days, weekends/holidays covered


def forward_test_specs() -> list[ShadowSpec]:
    import forward_test as F
    out = []
    uni = hashlib.sha256(" ".join(F.UNIVERSE).encode()).hexdigest()[:16]
    for v, (lo, hi) in F.VARIANTS.items():
        for rule in F.RULES:
            out.append(ShadowSpec(
                id=f"forward:{v}/{rule}", family="opening_auction_fade",
                strategy_ref=f"forward_test:{v}/{rule}", source="forward_test",
                config={"kind": "forward_test", "variant": v, "rule": rule, "min_gap": lo, "max_gap_band": hi,
                        "max_abs_gap": F.MAX_GAP, "max_per_day": F.MAX_PER_DAY, "notional": F.NOTIONAL,
                        "slippage_bps": F.SLIPPAGE_BPS, "universe_sha": uni, "per_trade_notional": F.NOTIONAL,
                        "fill_model": f"auction ± {F.SLIPPAGE_BPS} bps entry, 09:20 LTP ± {F.SLIPPAGE_BPS} bps exit",
                        "max_hold_days": 0, "backtest_max_dd": BACKTEST_MAX_DD.get(v)}))
    return out


def agent_spec(agent) -> ShadowSpec:
    """A validated agent as a shadow: id = name@VERSION, frozen on its lab fingerprint (a lab change without a
    VERSION bump is a config drift: its trades are not copied)."""
    lab_id = str(agent.LAB_ID)
    cfg = {"kind": "validated_agent", "agent": agent.name, "version": str(agent.VERSION), "lab_id": lab_id,
           "lab_fingerprint": str(agent.LAB_FINGERPRINT), "product": getattr(agent, "product", ""),
           "fill_model": "agent paper fills: marketable LIMIT at the quote touch (ask/bid), LTP without a touch",
           "max_hold_days": AGENT_MAX_HOLD_DAYS.get(agent.name, 10), "backtest_max_dd": BACKTEST_MAX_DD.get(lab_id),
           "per_trade_notional": 100000 if agent.name in ("auc_fade", "attn_on") else None}
    return ShadowSpec(id=f"{agent.name}@{agent.VERSION}", family=lab_id, strategy_ref=f"{type(agent).__module__}:"
                      f"{type(agent).__name__}", source="validated_agent", config=cfg)


def ensure_enrolled(reg: ShadowRegistry, agents: Optional[dict] = None) -> dict:
    """Register the forward-test shadows and one shadow per validated agent (idempotent). Returns {id: outcome}."""
    out: dict[str, str] = {}
    specs = forward_test_specs()
    for a in (agents or {}).values():
        try:
            specs.append(agent_spec(a))
        except Exception as exc:
            logger.error("[shadow] cannot build the spec of {}: {}", getattr(a, "name", a), exc)
    known = reg.entries()
    for s in specs:
        try:
            out[s.id] = "known" if s.id in known else "registered"
            reg.register(s)
        except FrozenConfigError as exc:
            out[s.id] = "config_drift"
            logger.error("[shadow] {}", exc)
    return out


def sync_forward_test(reg: ShadowRegistry, state: Optional[dict] = None) -> int:
    """Copy forward_test's built trades (per variant/rule) into the forward shadows. Returns trades added.
    Skipped for a shadow whose config drifted (forward_test constants changed since it was frozen): the trades of
    the changed strategy must not be mixed into the frozen shadow's record."""
    import forward_test as F
    state = state if state is not None else F._load()
    ents, n = reg.entries(), 0
    current = {s.id: s.config_hash for s in forward_test_specs()}
    drifted = {sid for sid, h in current.items() if sid in ents and ents[sid]["config_hash"] != h}
    for sid in sorted(drifted):
        logger.error("[shadow] {}: forward_test config drifted from the frozen config — trades not copied", sid)
    for day, rec in sorted((state.get("days") or {}).items()):
        trades = rec.get("trades") or {}
        for v in F.VARIANTS:
            for rule in F.RULES:
                sid = f"forward:{v}/{rule}"
                if sid not in ents or sid in drifted or sid not in current:
                    continue
                for t in (trades.get(v) or {}).get(rule, []):
                    key = f"{day}:{t['symbol']}"
                    r = reg.record_trade(sid, {
                        "key": key, "trade_key": key, "day": day, "symbol": t["symbol"], "side": t["side"],
                        "qty": t["qty"], "entry_fill": t["entry_fill"], "exit_fill": t["exit_fill"],
                        "gross": t["gross"], "cost": t["cost"], "net": t["net"],
                        "notional": round(float(t["auction_px"]) * int(t["qty"]), 2),
                        "signal_gap_pct": t.get("signal_gap_pct"), "source": "forward_test"})
                    n += r is not None
    return n


def sync_validated_agents(reg: ShadowRegistry, agents: dict, ledger_rows: Optional[list] = None) -> int:
    """Copy each validated agent's closed paper trades (performance ledger, its frozen VERSION) into its shadow.
    Strangle legs share a trade_key (the structure group). Skipped for a shadow whose config drifted."""
    if ledger_rows is None:
        from performance import performance
        ledger_rows = list(performance.trades)
    ents, n = reg.entries(), 0
    for a in agents.values():
        try:
            spec = agent_spec(a)
        except Exception:
            continue
        e = ents.get(spec.id)
        if e is None or e["config_hash"] != spec.config_hash:
            continue
        for row in ledger_rows:
            if row.get("strategy") != a.name or str(row.get("version")) != str(a.VERSION):
                continue
            key = str(row.get("oid") or f"{row.get('t')}:{row.get('symbol')}:{row.get('side')}")
            r = reg.record_trade(spec.id, {
                "key": key, "trade_key": str(row.get("group") or key), "day": str(row.get("t", ""))[:10],
                "symbol": row.get("symbol"), "side": row.get("side"), "qty": row.get("qty"),
                "entry_fill": row.get("entry"), "exit_fill": row.get("exit"), "gross": row.get("gross"),
                "cost": row.get("cost"), "net": float(row.get("net", 0.0)),
                "notional": round(float(row.get("entry") or 0) * float(row.get("qty") or 0), 2),
                "source": "performance_ledger", "exit_kind": row.get("exit_kind")})
            n += r is not None
    return n


def enabled() -> bool:
    """SHADOW_REGISTRY_ENABLED (default false) and PAPER mode."""
    from config import settings
    return bool(getattr(settings, "shadow_registry_enabled", False)) and settings.trading_mode == "PAPER"


def daily_job(reg: Optional[ShadowRegistry] = None, agents: Optional[dict] = None) -> dict:
    """Enrol, copy today's forward-test and validated-agent trades, evaluate. Read-only towards trading."""
    if not enabled():
        return {"skipped": "SHADOW_REGISTRY_ENABLED is off or not PAPER"}
    reg = reg or ShadowRegistry()
    if agents is None:
        from agents.strategy_agents import ALL_AGENTS
        agents = {n: a for n, a in ALL_AGENTS.items() if getattr(a, "scheduled", False)}
    out = {"enrolled": ensure_enrolled(reg, agents)}
    out["forward_trades_added"] = sync_forward_test(reg)
    out["agent_trades_added"] = sync_validated_agents(reg, agents)
    out["evaluation"] = reg.evaluate_all(write=True)
    out["verify_ok"] = reg.verify_all()["ok"]
    logger.info("[shadow] daily: +{} forward, +{} agent trades; chains ok={}", out["forward_trades_added"],
                out["agent_trades_added"], out["verify_ok"])
    return out
