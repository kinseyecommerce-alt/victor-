"""
forward_lab.py — the forward lab: every lab strategy run on each new session of real market data, after the close,
in PAPER, with no orders and no money; tamper-evident ledgers; a PRE-REGISTERED rule that controls luck across the
whole family (forward_lab_rule.json, frozen: its sha256 is pinned here and recorded in the registry's genesis).

Daily job (platform_scheduler, 16:45 IST, only with FORWARD_LAB_ENABLED=true and PAPER; the job re-checks both):
  1. data     forward_lab_store.fetch_session: today's 1-minute bars of the lab universe through kite_client's
              history limiter (waits while the option-bar archiver uses Kite), the corporate-action check / refetch,
              the completeness verdict. First run: 420 sessions of warm-up history (forward_lab_store.bootstrap).
              INCOMPLETE -> retried once at the next session's run; still incomplete -> recorded INCOMPLETE and skipped
              for every strategy (never partial).
  2. run      forward_lab_engine in a SEPARATE PROCESS (python -m forward_lab_engine), nice 10, a time limit
              (FORWARD_LAB_TIME_LIMIT_S) and an address-space limit (FORWARD_LAB_MEM_MB): the API, the price feed and
              the agents are never blocked. Its proposed state is adopted only after a successful run.
  3. ledgers  one hash-chained JSONL ledger per strategy (<root>/ledgers/<id>.jsonl: genesis + one "day" record per
              session with closed trades) and the registry chain (<root>/registry.jsonl: genesis with the rule and
              baseline sha256, registrations, runs (with the sha256 of the day's full trade file trades/<day>.json.gz),
              ledger checkpoints, looks, status changes). shadow_registry.verify_chain checks every file.
  4. rule     looks every 20 forward sessions (max 12): forward group = Bonferroni-qualification over the family
              (never shrinks); proven / candidate group = consistency with the lab; demotion for everyone.
              Each look runs once and is never recomputed.
EV-PEAD-LONG-H60 is copied from the agent's own performance-ledger rows (read-only).
Read-only towards trading: no order entry point is imported, no agent, risk or trading state is read or changed.
At most one Telegram summary per day, only with FORWARD_LAB_TELEGRAM=true.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np
from loguru import logger

import forward_lab_store as FS
from shadow_registry import GENESIS_PREV, _canon, _record_hash, sha256_hex, verify_chain

HERE = Path(__file__).resolve().parent
RULE_FILE = HERE / "forward_lab_rule.json"
BASELINE_FILE = HERE / "forward_lab_baseline.json"
# sha256 of forward_lab_rule.json as committed (the pre-registration). An edited rule file is refused.
RULE_SHA256 = "2370ba69766d50de614cbe9049df9477258fa5b6a8ac364bd12eacf345f73468"
_LOCK = threading.RLock()


# ── rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuleChanged(RuntimeError):
    pass


def load_rule(path=RULE_FILE, expect: Optional[str] = None) -> dict:
    """The frozen rule. Raises RuleChanged when the file's sha256 is not the pinned one."""
    sha = file_sha256(path)
    want = RULE_SHA256 if expect is None else expect
    if want != "PENDING" and sha != want:
        raise RuleChanged(f"forward_lab_rule.json sha256 {sha[:12]} != pinned {want[:12]}: the rule is frozen")
    rule = json.loads(Path(path).read_text())
    rule["_sha256"] = sha
    return rule


def load_baseline(path=BASELINE_FILE) -> dict:
    return json.loads(Path(path).read_text())


def alpha_per_test(rule: dict, family_size: int) -> float:
    q = rule["qualification"]
    return float(q["alpha_family"]) / (max(1, int(family_size)) * int(rule["looks"]["max_looks"]))


def z_required(rule: dict, family_size: int) -> float:
    return NormalDist().inv_cdf(1.0 - alpha_per_test(rule, family_size))


# ── statistics (pure) ─────────────────────────────────────────────────────────────────────────────────────────────
def look_seed(sid: str, look: int, tag: str = "look") -> int:
    return int.from_bytes(hashlib.sha256(f"{sid}|{tag}{look}".encode()).digest()[:8], "big")


def stationary_indices(T: int, n: int, B: int, mean_block: float, rng) -> np.ndarray:
    """B x n indices into a series of length T: Politis-Romano stationary bootstrap (geometric blocks, wrap-around),
    the algorithm of stratlab.robustness.stationary_bootstrap_indices with a sample length n."""
    p = 1.0 / float(mean_block)
    idx = np.empty((B, n), np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new = rng.random((B, n)) < p
    starts = rng.integers(0, T, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def qualification_test(daily, seed: int, B: int, mean_block: float) -> dict:
    """One-sided test of mean daily net > 0: stationary block bootstrap of the CENTRED daily series; z = mean /
    sd(bootstrap means); p = 1 - Phi(z) (the Bonferroni level is far below 1/B, so the bootstrap distribution's
    normal tail is used); p_boot = (1 + #{boot mean >= observed}) / (B + 1) reported alongside."""
    x = np.asarray(daily, float)
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else 0.0, "sd_boot": None, "z": None, "p": 1.0, "p_boot": 1.0}
    mean = float(x.mean())
    rng = np.random.default_rng(int(seed))
    y = x - mean
    idx = stationary_indices(n, n, int(B), mean_block, rng)
    bm = y[idx].mean(axis=1)
    sd = float(bm.std(ddof=1))
    z = (mean / sd) if sd > 0 else (math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0))
    p = 1.0 - NormalDist().cdf(z) if math.isfinite(z) else (0.0 if z > 0 else 1.0)
    p_boot = (1 + int((bm >= mean).sum())) / (B + 1)
    return {"n": n, "mean": mean, "sd_boot": sd, "z": z, "p": p, "p_boot": p_boot}


def consistency_band(lab_daily, n: int, seed: int, B: int, mean_block: float, q: float) -> float:
    """Lower band: the q-quantile of the means of n-session stationary-bootstrap samples of the lab's daily series."""
    L = np.asarray(lab_daily, float)
    rng = np.random.default_rng(int(seed))
    idx = stationary_indices(len(L), int(n), int(B), mean_block, rng)
    return float(np.quantile(L[idx].mean(axis=1), q))


def max_drawdown(values) -> float:
    c = np.cumsum(np.asarray(list(values), float))
    if c.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    return round(float(max(0.0, (peak - c).max())), 2)


# ── hash-chained logs (shadow_registry's format and verify_chain; cached tail, batch append) ─────────────────────
class Chain:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self._tail: Optional[tuple[int, str]] = None

    def _read_tail(self) -> tuple[int, str]:
        if self._tail is not None:
            return self._tail
        seq, prev = -1, GENESIS_PREV
        if self.path.exists():
            size = self.path.stat().st_size
            span = 65536
            with open(self.path, "rb") as fh:
                while True:                      # read back far enough to hold the whole last line
                    start = max(0, size - span)
                    fh.seek(start)
                    buf = fh.read().rstrip(b"\n")
                    if start == 0 or b"\n" in buf:
                        break
                    span *= 4
            if buf:
                last = json.loads(buf.rsplit(b"\n", 1)[-1].decode("utf-8"))
                seq, prev = int(last["seq"]), str(last["hash"])
        self._tail = (seq, prev)
        return self._tail

    def append_many(self, items: list[tuple[str, dict]], ts: Optional[str] = None) -> list[dict]:
        if not items:
            return []
        seq, prev = self._read_tail()
        ts = ts or datetime.now().isoformat(timespec="seconds")
        out = []
        for kind, data in items:
            rec = {"seq": seq + 1, "ts": ts, "kind": kind, "data": data, "prev_hash": prev}
            rec["hash"] = _record_hash(rec)
            out.append(rec)
            seq, prev = rec["seq"], rec["hash"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("".join(_canon(r) + "\n" for r in out))
            fh.flush()
            os.fsync(fh.fileno())
        self._tail = (seq, prev)
        return out

    def append(self, kind: str, data: dict) -> dict:
        return self.append_many([(kind, data)])[0]

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def verify(self, expect_head=None) -> dict:
        return verify_chain(self.path, expect_head)

    def head(self) -> Optional[tuple[int, str]]:
        s, h = self._read_tail()
        return None if s < 0 else (s, h)


def safe_name(sid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in sid) + "-" + hashlib.sha1(sid.encode()).hexdigest()[:6]


# ── the lab ───────────────────────────────────────────────────────────────────────────────────────────────────────
class ForwardLab:
    def __init__(self, root=None, rule: Optional[dict] = None, baseline: Optional[dict] = None) -> None:
        self.root = Path(root) if root is not None else FS.default_root()
        self.rule = rule if rule is not None else load_rule()
        self.baseline = baseline if baseline is not None else load_baseline()
        self.registry = Chain(self.root / "registry.jsonl")
        self.store = FS.Store(self.root / "bars", cap_root=self.root)
        self.state_dir = self.root / "state"
        self._ledgers: dict = {}

    def ledger(self, sid: str) -> Chain:
        c = self._ledgers.get(sid)
        if c is None:
            c = self._ledgers[sid] = Chain(self.root / "ledgers" / f"{safe_name(sid)}.jsonl")
        return c

    # ── folded registry state ──
    def state(self) -> dict:
        st = {"genesis": None, "strategies": {}, "runs": {}, "looks": {}, "status": {}, "checkpoints": {},
              "family": []}
        for rec in self.registry.records():
            k, d = rec["kind"], rec["data"]
            if k == "genesis":
                st["genesis"] = d
            elif k == "register":
                for s in d["strategies"]:
                    st["strategies"][s["id"]] = {**s, "registered_on": d["registered_on"]}
                    if s["group"] == "forward" and s["id"] not in st["family"]:
                        st["family"].append(s["id"])
            elif k == "run":
                st["runs"][int(d["day"])] = d
            elif k == "look":
                st["looks"].setdefault(d["id"], {})[str(d["look"])] = d
            elif k == "status":
                st["status"][d["id"]] = d
            elif k == "checkpoint":
                for sid, h in d["heads"].items():
                    st["checkpoints"][sid] = tuple(h)
        return st

    def family_size(self, st: Optional[dict] = None) -> int:
        st = st or self.state()
        return len(st["family"]) or int(self.rule["groups"]["forward"]["initial_family_size"])

    # ── genesis / registration ──
    def ensure_genesis(self) -> dict:
        st = self.state()
        if st["genesis"] is not None:
            if st["genesis"]["rule_sha256"] != self.rule["_sha256"]:
                raise RuleChanged("the registry was started under a different rule file")
            return st["genesis"]
        g = {"rule_version": self.rule["version"], "rule_sha256": self.rule["_sha256"],
             "baseline_sha256": file_sha256(BASELINE_FILE) if BASELINE_FILE.exists() else None,
             "initial_family_size": self.rule["groups"]["forward"]["initial_family_size"],
             "initial_ids_sha256": self.rule["groups"]["forward"]["initial_ids_sha256"]}
        self.registry.append("genesis", g)
        return g

    def register(self, table: list[dict], first_day: int) -> dict:
        """Register every strategy of `table` not yet registered (idempotent). The forward family only grows."""
        self.ensure_genesis()
        st = self.state()
        new, drift = [], []
        for s in table:
            cur = st["strategies"].get(s["id"])
            ch = sha256_hex({k: s[k] for k in ("id", "group", "kind", "fingerprint")})
            if cur is None:
                new.append({**{k: s[k] for k in ("id", "family", "group", "kind", "fingerprint")}, "config_hash": ch})
            elif cur["config_hash"] != ch:
                drift.append(s["id"])
        if new:
            self.registry.append("register", {"registered_on": int(first_day), "strategies": new,
                                              "family_size_after": len(st["family"]) +
                                              sum(1 for s in new if s["group"] == "forward")})
            items = []
            for s in new:
                self.ledger(s["id"]).append("genesis", {"id": s["id"], "group": s["group"],
                                                        "config_hash": s["config_hash"],
                                                        "rule_sha256": self.rule["_sha256"],
                                                        "registered_on": int(first_day)})
                items.append(s["id"])
        return {"registered": len(new), "config_drift": drift}

    # ── ingest one engine result ──
    def ingest(self, day: int, result: dict, pead_trades: list, table_ids: set, drift: set) -> dict:
        st = self.state()
        if int(day) in st["runs"]:
            raise RuntimeError(f"session {day} was already ingested")
        full: dict = {}
        heads: dict = {}
        strategies = dict(result.get("strategies", {}))
        strategies["EV-PEAD-LONG-H60"] = {"closed": pead_trades, "open": [], "error": None}
        n_closed = 0
        errors = 0
        for sid, r in strategies.items():
            if sid not in st["strategies"] or sid in drift:
                continue
            if r.get("error"):
                errors += 1
            closed = r.get("closed") or []
            if not closed and not r.get("open"):
                continue
            full[sid] = {"closed": closed, "open": r.get("open") or []}
            if not closed and not r.get("error"):
                continue                       # open positions only: in the day's trade file, no ledger record
            nets = [float(t[11]) for t in closed]
            rec = {"date": int(day), "trades": len(closed), "net": round(sum(nets), 2),
                   "gross": round(sum(float(t[9]) for t in closed), 2), "cost": round(sum(float(t[10]) for t in closed), 2),
                   "gp": round(sum(x for x in nets if x > 0), 2), "gl": round(-sum(x for x in nets if x < 0), 2),
                   "open": len(r.get("open") or []), "trades_sha256": sha256_hex(closed)}
            if r.get("error"):
                rec["error"] = str(r["error"])[:200]
            out =self.ledger(sid).append("day", rec)
            heads[sid] = [out["seq"], out["hash"]]
            n_closed += len(closed)
        tdir = self.root / "trades"
        tdir.mkdir(parents=True, exist_ok=True)
        blob = gzip.compress(_canon(full).encode(), mtime=0)
        path = tdir / f"{int(day)}.json.gz"
        path.write_bytes(blob)
        run = {"day": int(day), "status": "complete", "strategies_run": len(strategies), "errors": errors,
               "closed_trades": n_closed, "trades_file": path.name, "trades_sha256": hashlib.sha256(blob).hexdigest(),
               "engine": result.get("engine"), "timing": result.get("timing"), "peak_rss_mb": result.get("peak_rss_mb"),
               "engine_errors": (result.get("errors") or [])[:20]}
        self.registry.append_many([("run", run), ("checkpoint", {"day": int(day), "heads": heads})])
        return run

    def record_skip(self, day: int, status: str, reasons: list) -> dict:
        run = {"day": int(day), "status": status, "strategies_run": 0, "errors": 0, "reasons": reasons[:30]}
        self.registry.append("run", run)
        return run

    # ── series ──
    def complete_days(self, st: dict) -> list[int]:
        return sorted(d for d, r in st["runs"].items() if r["status"] == "complete")

    def day_records(self, sid: str) -> dict:
        return {int(r["data"]["date"]): r["data"] for r in self.ledger(sid).records() if r["kind"] == "day"}

    def series(self, sid: str, st: dict, recs: Optional[dict] = None) -> tuple[list[int], list[float], list[int]]:
        s = st["strategies"][sid]
        days = [d for d in self.complete_days(st) if d >= int(s["registered_on"])]
        recs = self.day_records(sid) if recs is None else recs
        nets = [float(recs.get(d, {}).get("net", 0.0)) for d in days]
        ntr = [int(recs.get(d, {}).get("trades", 0)) for d in days]
        return days, nets, ntr

    def lab_ref(self, sid: str) -> dict:
        return (self.baseline.get("refs") or {}).get(sid) or {}

    # ── the pre-registered rule ──
    def evaluate(self, sid: str, st: dict, write: bool = True) -> dict:
        s = st["strategies"][sid]
        grp = s["group"]
        R = self.rule
        cur = (st["status"].get(sid) or {}).get("status")
        days, nets, ntr = self.series(sid, st)
        out = {"id": sid, "group": grp, "days": len(days), "trades": int(sum(ntr)), "net": round(sum(nets), 2),
               "new": []}
        ref = self.lab_ref(sid)
        lab_dd = float(ref.get("max_dd") or 0.0)
        fdd = max_drawdown(nets)
        mult = float(R["demotion"]["dd_multiple"])
        out["forward_max_dd"], out["lab_max_dd"] = fdd, lab_dd or None
        records = []
        if cur == "demoted":
            out["status"] = "demoted"
            return out
        if lab_dd > 0 and fdd > mult * lab_dd:
            reason = f"forward max DD {fdd:,.0f} > {mult} x lab 2023-26 max DD {lab_dd:,.0f}"
            records.append(("status", {"id": sid, "status": "demoted", "reason": reason, "day": days[-1]}))
            out["status"] = "demoted"
        else:
            looks = dict(st["looks"].get(sid, {}))
            every, max_looks = int(R["looks"]["every_sessions"]), int(R["looks"]["max_looks"])
            status = cur
            for k in range(1, max_looks + 1):
                if str(k) in looks:
                    continue
                n = k * every
                if len(days) < n:
                    break
                sub, subtr = nets[:n], int(sum(ntr[:n]))
                change = None
                res = {"id": sid, "look": k, "day": days[n - 1], "days": n, "trades": subtr,
                       "net": round(sum(sub), 2), "mean_daily": round(float(np.mean(sub)), 4)}
                if grp == "forward":
                    q = R["qualification"]
                    fam = self.family_size(st)
                    res.update({"family_size": fam, "alpha": alpha_per_test(R, fam), "z_required": z_required(R, fam)})
                    if subtr < int(q["min_trades"]) or n < int(q["min_days"]):
                        res.update({"tested": False, "verdict": "insufficient"})
                    else:
                        b = q["bootstrap"]
                        t = qualification_test(sub, look_seed(sid, k), int(b["resamples"]), float(b["mean_block"]))
                        ok = bool(t["mean"] > 0 and t["z"] is not None and t["z"] >= res["z_required"])
                        res.update({"tested": True, "z": t["z"], "p": t["p"], "p_boot": t["p_boot"],
                                    "sd_boot": t["sd_boot"], "verdict": "qualified" if ok else "not_qualified"})
                        if ok and status != "qualified":
                            status = "qualified"
                            change = {"id": sid, "status": "qualified", "day": res["day"],
                                      "reason": f"look {k}: z={t['z']:.2f} >= {res['z_required']:.2f}"}
                else:
                    c = R["consistency"]
                    if subtr < int(c["min_trades"]) or not ref.get("daily"):
                        res.update({"tested": False, "verdict": "insufficient" if ref.get("daily") else "no_lab_series"})
                    else:
                        b = c["bootstrap"]
                        band = consistency_band(ref["daily"], n, look_seed(sid, k, "consistency"),
                                                int(b["resamples"]), float(b["mean_block"]), float(c["quantile"]))
                        bad = float(np.mean(sub)) < band
                        res.update({"tested": True, "lab_mean_daily": ref.get("mean_daily"), "band": round(band, 4),
                                    "verdict": "inconsistent" if bad else "consistent"})
                        if bad and status != "inconsistent":
                            status = "inconsistent"
                            change = {"id": sid, "status": "inconsistent", "day": res["day"],
                                      "reason": f"look {k}: live mean {np.mean(sub):,.1f} < lab 1% band {band:,.1f}"}
                records.append(("look", res))
                if change:
                    records.append(("status", change))
                out["new"].append(k)
            out["status"] = status or ("watching" if grp == "forward" else "consistent")
        if write and records:
            self.registry.append_many(records)
        return out

    def evaluate_all(self, write: bool = True) -> dict:
        st = self.state()
        return {sid: self.evaluate(sid, st, write) for sid in st["strategies"]}

    # ── verification ──
    def verify_all(self) -> dict:
        st = self.state()
        out = {"registry": self.registry.verify()}
        bad = []
        for sid in st["strategies"]:
            res = self.ledger(sid).verify(st["checkpoints"].get(sid))
            if not res["ok"]:
                bad.append({"id": sid, "error": res["error"]})
        out["ledgers_bad"] = bad
        tf_bad = []
        for d, r in st["runs"].items():
            if r.get("trades_file"):
                p = self.root / "trades" / r["trades_file"]
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != r.get("trades_sha256"):
                    tf_bad.append(r["trades_file"])
        out["trade_files_bad"] = tf_bad
        out["ok"] = bool(out["registry"]["ok"] and not bad and not tf_bad)
        return out

    # ── status for the API ──
    def summary_rows(self, st: Optional[dict] = None) -> list[dict]:
        st = st or self.state()
        rows = []
        for sid, s in st["strategies"].items():
            recs = self.day_records(sid)
            days, nets, ntr = self.series(sid, st, recs)
            gp = sum(float(r.get("gp", 0)) for d, r in recs.items() if d in set(days))
            gl = sum(float(r.get("gl", 0)) for d, r in recs.items() if d in set(days))
            looks = st["looks"].get(sid, {})
            z = None
            for k in sorted(looks, key=int):
                if looks[k].get("tested") and looks[k].get("z") is not None:
                    z = looks[k]["z"]
            status = (st["status"].get(sid) or {}).get("status") or \
                     ("watching" if s["group"] == "forward" else "consistent")
            rows.append({"id": sid, "family": s["family"], "group": s["group"], "trades": int(sum(ntr)),
                         "days": len(days), "net": round(sum(nets), 2),
                         "pf": round(gp / gl, 3) if gl > 0 else None,
                         "z": None if z is None or not math.isfinite(z) else round(float(z), 3), "status": status})
        order = {"proven": 0, "candidate": 1, "forward": 2}
        rows.sort(key=lambda r: (order.get(r["group"], 3), r["id"]))
        return rows


# ── settings / paths ──────────────────────────────────────────────────────────────────────────────────────────────
def _settings():
    from config import settings
    return settings


def enabled() -> bool:
    try:
        s = _settings()
        return bool(getattr(s, "forward_lab_enabled", False)) and s.trading_mode == "PAPER"
    except Exception:
        return False


def _status_path(root: Path) -> Path:
    return Path(root) / "status.json"


def _jsave(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, path)


def _jload(path: Path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


# ── engine process ────────────────────────────────────────────────────────────────────────────────────────────────
class EngineFailed(RuntimeError):
    pass


def run_engine_process(store_root: Path, state_dir: Path, day: int, since: int, fwd_start: int, work: Path,
                       jobs: int = 1, time_limit_s: int = 2700, mem_mb: int = 3072, ids=None) -> dict:
    """python -m forward_lab_engine in its own process (never in the API process). Raises EngineFailed on a timeout,
    a crash, the memory limit or an engine error."""
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "forward_lab_engine", "--store", str(store_root), "--state", str(state_dir),
           "--day", str(day), "--since", str(since), "--fwd-start", str(fwd_start), "--work", str(work),
           "--jobs", str(max(1, int(jobs))), "--mem-mb", str(int(mem_mb))]
    if ids:
        cmd += ["--ids", ",".join(ids)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:        # the strategy builder's server NSE archive (last 400 sessions) for the nsedata-based catalogue ids
        import builder_data
        env.update(builder_data.engine_env())
    except Exception:
        pass
    t0 = time.time()
    # own process group: on the time limit the engine AND its worker processes are killed
    p = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        _out, err_txt = p.communicate(timeout=int(time_limit_s))
    except subprocess.TimeoutExpired:
        import signal
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()
        p.communicate()
        raise EngineFailed(f"time limit {time_limit_s}s exceeded")
    if p.returncode != 0:
        err = (work / "error.txt").read_text()[:600] if (work / "error.txt").exists() else (err_txt or "")[-600:]
        raise EngineFailed(f"engine exit {p.returncode}: {err}")
    with gzip.open(work / "result.json.gz", "rt") as fh:
        res = json.load(fh)
    res.setdefault("timing", {})["process_wall"] = round(time.time() - t0, 1)
    return res


def adopt_state(state_dir: Path, work: Path) -> None:
    """The engine's proposed state replaces the current one (only after a successful ingest)."""
    new = work / "state_new"
    if not new.exists():
        return
    old = state_dir.with_name(state_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if state_dir.exists():
        os.replace(state_dir, old)
    shutil.copytree(new, state_dir)
    shutil.rmtree(old, ignore_errors=True)


# ── PEAD (the agent's own ledger, read-only) ─────────────────────────────────────────────────────────────────────
def pead_trades(day: int, since: int, rows: Optional[list] = None) -> list:
    """EV-PEAD-LONG-H60 closed trades from the performance ledger rows of the agent (exit date in (since, day]),
    normalised to the lab's Rs 1 lakh per trade."""
    if rows is None:
        try:
            from performance import performance
            rows = list(performance.trades)
        except Exception:
            rows = []
    out = []
    for r in rows:
        if r.get("strategy") != "pead_h60":
            continue
        try:
            d = int(str(r.get("t", ""))[:10].replace("-", ""))
        except ValueError:
            continue
        if not (int(since) < d <= int(day)):
            continue
        qty, entry, ex = float(r.get("qty") or 0), float(r.get("entry") or 0), float(r.get("exit") or 0)
        notional = qty * entry
        f = 100000.0 / notional if notional > 0 else 1.0
        # the ledger row is written at the exit ("t"); its entry date is not on the row (0 = unknown)
        out.append([str(r.get("symbol")), 0, 1, "", entry, d, str(r.get("t", ""))[11:16], ex, int(qty),
                    round(float(r.get("gross") or 0) * f, 2), round(float(r.get("cost") or 0) * f, 2),
                    round(float(r.get("net") or 0) * f, 2), str(r.get("exit_kind") or r.get("oid") or "ledger")])
    return out


# ── the daily job ─────────────────────────────────────────────────────────────────────────────────────────────────
def _ymd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _d(x: int) -> date:
    return date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100)


def pending_days(last: Optional[int], today: date, is_trading_day, special: set, max_days: int = 5) -> list[int]:
    """Trading days after `last` up to today (special sessions never run), at most `max_days` (oldest first)."""
    out = []
    d = today
    while len(out) < 40:
        if last is not None and _ymd(d) <= int(last):
            break
        if is_trading_day(d) and d not in special:
            out.append(_ymd(d))
        if last is None:
            break
        d -= timedelta(days=1)
    return sorted(out)[-max_days:]


def daily_job(now: Optional[datetime] = None, kite=None, root=None, lab: Optional[ForwardLab] = None,
              fetch=None, engine=None, ledger_rows: Optional[list] = None) -> dict:
    """Scheduler entry point. Off (flag false or LIVE) -> returns at once: no Kite call, nothing read or written."""
    if not enabled():
        return {"skipped": "FORWARD_LAB_ENABLED is off or not PAPER"}
    import ist_clock
    s = _settings()
    now = now or ist_clock.now_ist()
    if now.time() < datetime.strptime("15:45", "%H:%M").time():
        return {"skipped": "before 15:45 IST: today's session is not complete"}
    with _LOCK:
        lab = lab or ForwardLab(root)
        root = lab.root
        status = {"date": None, "status": "never", "strategies_run": 0, "errors": 0}
        try:
            lab.ensure_genesis()
        except RuleChanged as exc:
            status.update({"status": "rule_changed", "error": str(exc)})
            _jsave(_status_path(root), {"last_run": status})
            return status
        st = lab.state()
        vr = lab.registry.verify()
        if not vr["ok"]:
            status.update({"status": "tamper", "error": vr["error"]})
            _jsave(_status_path(root), {"last_run": status})
            logger.error("[forward-lab] registry chain broken: {} — not run", vr["error"])
            return status
        runs = sorted(st["runs"])
        last = runs[-1] if runs else None
        special = set(getattr(ist_clock, "SPECIAL_SESSIONS", {}) or {})
        days = pending_days(last, now.date(), ist_clock.is_trading_day, special)
        js = _jload(root / "job_state.json", {})
        out = {"processed": [], "status": status}
        if not days:
            return {"skipped": "nothing to do", **out}
        if kite is None and fetch is None:
            from kite_client import kite_client as kite
        try:
            import option_bar_archiver as OBA
            arch_lock = OBA._LOCK
        except Exception:
            arch_lock = threading.Lock()
        for day in days:
            # 1. data
            if fetch is not None:
                fr = fetch(lab.store, _d(day))
            else:
                got = arch_lock.acquire(timeout=1800)       # never overlap the option-bar archiver's Kite calls
                try:
                    tokens = FS.instrument_tokens(kite)
                    if not lab.store.sessions():
                        prev = _d(day) - timedelta(days=1)
                        while not ist_clock.is_trading_day(prev):
                            prev -= timedelta(days=1)
                        FS.bootstrap(lab.store, kite, prev, tokens, log=logger.info)
                        m = lab.store.manifest()
                        z = lab.store.raw("_NIFTY50_min")
                        for dd in sorted(set(int(x) for x in z["date"])):
                            m.setdefault("sessions", {}).setdefault(str(dd), {"status": "complete", "source": "warm-up"})
                        lab.store._save_manifest(m)
                    ca = None
                    try:
                        import event_calendar
                        if event_calendar.ca_calendar_loaded():
                            ca = lambda sym, d: bool(event_calendar.price_adjusting_actions(sym, d))
                    except Exception:
                        ca = None
                    fr = FS.fetch_session(lab.store, kite, _d(day), tokens, ca, log=logger.info)
                except FS.DiskCapReached as exc:
                    status.update({"date": str(_d(day)), "status": "disk_cap", "error": str(exc)})
                    break
                finally:
                    if got:
                        arch_lock.release()
            if not fr.get("complete"):
                attempts = int(js.get("attempts", {}).get(str(day), 0)) + 1
                js.setdefault("attempts", {})[str(day)] = attempts
                if day == days[-1] and attempts < 2:
                    status.update({"date": str(_d(day)), "status": "pending", "reasons": fr.get("reasons", [])[:10]})
                    break
                lab.store.mark_session(day, "incomplete", reasons=fr.get("reasons", [])[:20])
                lab.record_skip(day, "incomplete", fr.get("reasons", []))
                status.update({"date": str(_d(day)), "status": "incomplete"})
                out["processed"].append({"day": day, "status": "incomplete"})
                continue
            try:
                lab.store.check_cap()           # ledgers / trade file would be written: stop at the cap
            except FS.DiskCapReached as exc:
                status.update({"date": str(_d(day)), "status": "disk_cap", "error": str(exc)})
                break
            lab.store.mark_session(day, "complete", refetch=fr.get("refetch", []))
            # 2. register (first run: the whole family) and run the engine in its own process
            import forward_lab_engine as E
            table = E.strategy_table()
            st = lab.state()
            fwd_start = min(int(v["registered_on"]) for v in st["strategies"].values()) if st["strategies"] else day
            reg = lab.register(table, day)
            st = lab.state()
            done = sorted(st["runs"])
            since = done[-1] if done else _ymd(_d(day) - timedelta(days=1))
            work = root / "work" / str(day)
            try:
                if engine is not None:
                    res = engine(lab.store.root, lab.state_dir, day, since, fwd_start, work)
                else:
                    res = run_engine_process(lab.store.root, lab.state_dir, day, since, fwd_start, work,
                                             jobs=int(getattr(s, "forward_lab_jobs", 1)),
                                             time_limit_s=int(getattr(s, "forward_lab_time_limit_s", 2700)),
                                             mem_mb=int(getattr(s, "forward_lab_mem_mb", 3072)))
            except Exception as exc:
                attempts = int(js.get("engine_attempts", {}).get(str(day), 0)) + 1
                js.setdefault("engine_attempts", {})[str(day)] = attempts
                logger.error("[forward-lab] engine failed for {} (attempt {}): {}", day, attempts, exc)
                if attempts >= 2:
                    lab.record_skip(day, "error", [str(exc)[:300]])
                    out["processed"].append({"day": day, "status": "error"})
                status.update({"date": str(_d(day)), "status": "error", "error": str(exc)[:300]})
                break
            # 3. ledgers
            ids = {t["id"] for t in table}
            run = lab.ingest(day, res, pead_trades(day, since, ledger_rows), ids, set(reg["config_drift"]))
            adopt_state(lab.state_dir, work)
            shutil.rmtree(work, ignore_errors=True)
            # 4. the rule
            ev = lab.evaluate_all(write=True)
            status.update({"date": str(_d(day)), "status": "complete", "strategies_run": run["strategies_run"],
                           "errors": run["errors"]})
            out["processed"].append({"day": day, "status": "complete", "closed_trades": run["closed_trades"],
                                     "timing": run.get("timing")})
            out["new_status"] = [{"id": k, "status": v["status"]} for k, v in ev.items() if v.get("new")]
        _jsave(root / "job_state.json", js)
        write_status(lab, status)
        out["status"] = status
        out["builder_family"] = builder_family_step()
        return out


def builder_family_step() -> dict:
    """The SECOND forward family, builder/v1 (builder_forward.py, its own frozen rule and ledgers): processed after
    this family's session. A no-op unless BUILDER_ENABLED is on and a builder strategy is registered; never raises."""
    try:
        import builder_forward
        return builder_forward.run_pending()
    except Exception as exc:
        logger.error("[forward-lab] builder/v1 family step failed: {}", exc)
        return {"status": "error", "error": str(exc)[:200]}


def write_status(lab: ForwardLab, last_run: dict) -> dict:
    st = lab.state()
    rows = lab.summary_rows(st)
    ver = lab.verify_all()
    obj = {"last_run": last_run, "rows": rows, "verify": {"ok": ver["ok"], "registry": ver["registry"]["ok"],
                                                         "ledgers_bad": ver["ledgers_bad"][:20],
                                                         "trade_files_bad": ver["trade_files_bad"][:20]},
           "family_size": lab.family_size(st), "disk_mb": round(FS.dir_bytes(lab.root) / 1048576, 2),
           "written": datetime.now().isoformat(timespec="seconds")}
    if last_run.get("status") == "complete":
        obj["as_of"] = last_run.get("date")
    else:
        obj["as_of"] = _jload(_status_path(lab.root), {}).get("as_of")
    _jsave(_status_path(lab.root), obj)
    return obj


def telegram_text(status: dict) -> Optional[str]:
    lr = status.get("last_run") or {}
    if lr.get("status") != "complete":
        return None
    rows = status.get("rows") or []
    q = [r["id"] for r in rows if r["status"] == "qualified"]
    d = sum(1 for r in rows if r["status"] == "demoted")
    inc = [r["id"] for r in rows if r["status"] == "inconsistent"]
    return (f"<b>Forward lab</b> {lr.get('date')}: {lr.get('strategies_run')} strategies run, {lr.get('errors')} "
            f"errors\nqualified: {len(q)}{(' — ' + ', '.join(q[:5])) if q else ''}\ndemoted: {d}\n"
            f"inconsistent: {', '.join(inc) if inc else 'none'}\n(PAPER research only — nothing trades on this)")


def telegram_once(root=None) -> Optional[str]:
    """The day's summary text, at most once per session date (remembered in job_state.json); None otherwise."""
    root = Path(root) if root is not None else FS.default_root()
    st = _jload(_status_path(root), {})
    text = telegram_text(st)
    day = (st.get("last_run") or {}).get("date")
    if not text or not day:
        return None
    js = _jload(root / "job_state.json", {})
    if js.get("telegram_sent") == day:
        return None
    js["telegram_sent"] = day
    _jsave(root / "job_state.json", js)
    return text


# ── API views ─────────────────────────────────────────────────────────────────────────────────────────────────────
def api_status(root=None) -> dict:
    """GET /api/v2/forward-lab (the contract in API_V2.md)."""
    root = Path(root) if root is not None else FS.default_root()
    try:
        rule = load_rule()
        rule_err = None
    except Exception as exc:
        rule, rule_err = json.loads(RULE_FILE.read_text()) if RULE_FILE.exists() else {}, str(exc)
    cached = _jload(_status_path(root), {})
    rows = cached.get("rows") or []
    fam = int(cached.get("family_size") or ((rule.get("groups") or {}).get("forward") or {}).get("initial_family_size")
              or 0)
    lr = cached.get("last_run") or {}
    if rule_err:
        lr = {**lr, "status": "rule_changed"}
    elif cached.get("verify") and not cached["verify"].get("ok"):
        lr = {**lr, "status": "tamper"}
    q = rule.get("qualification") or {}
    as_of = lr.get("date") if lr.get("status") == "complete" else None
    if as_of is None:
        as_of = cached.get("as_of")
    return {
        "enabled": enabled(),
        "as_of": as_of,
        "rule": {"version": str(rule.get("version", "")), "family_size": fam,
                 "alpha": float(q.get("alpha_family", 0.05)),
                 "z_required": round(z_required(rule, fam), 4) if rule and fam else None,
                 "min_trades": int(q.get("min_trades", 50)), "min_days": int(q.get("min_days", 40))},
        "last_run": {"date": lr.get("date"), "status": str(lr.get("status") or "never"),
                     "strategies_run": int(lr.get("strategies_run") or 0), "errors": int(lr.get("errors") or 0)},
        "summary": {"n_strategies": len(rows), "qualified": sum(1 for r in rows if r["status"] == "qualified"),
                    "demoted": sum(1 for r in rows if r["status"] == "demoted"),
                    "consistency_fail": sum(1 for r in rows if r["status"] == "inconsistent")},
        "strategies": rows,
    }


def api_strategy(sid: str, root=None) -> Optional[dict]:
    """GET /api/v2/forward-lab/{id}; None when the id is not registered."""
    root = Path(root) if root is not None else FS.default_root()
    try:
        lab = ForwardLab(root)
    except Exception:
        return None
    st = lab.state()
    if sid not in st["strategies"]:
        return None
    recs = lab.day_records(sid)
    days, nets, ntr = lab.series(sid, st, recs)
    s = st["strategies"][sid]
    chain = lab.ledger(sid).verify(st["checkpoints"].get(sid))
    ref = lab.lab_ref(sid)
    fam = lab.family_size(st)
    looks = [st["looks"][sid][k] for k in sorted(st["looks"].get(sid, {}), key=int)]
    stat = st["status"].get(sid) or {}
    R = lab.rule
    every = int(R["looks"]["every_sessions"])
    return {
        "id": sid,
        "daily": [{"date": str(_d(d)), "trades": int(n), "net": round(float(v), 2)} for d, v, n in zip(days, nets, ntr)],
        "rule_status": {
            "group": s["group"], "family": s["family"], "kind": s.get("kind"), "registered_on": str(_d(s["registered_on"])),
            "status": stat.get("status") or ("watching" if s["group"] == "forward" else "consistent"),
            "status_reason": stat.get("reason"), "looks": looks,
            "next_look_at_days": min(((len(looks) + 1) * every), every * int(R["looks"]["max_looks"]))
            if len(looks) < int(R["looks"]["max_looks"]) else None,
            "days": len(days), "trades": int(sum(ntr)), "net": round(sum(nets), 2),
            "forward_max_dd": max_drawdown(nets),
            "lab": {k: ref.get(k) for k in ("trades", "net", "mean_daily", "max_dd", "pf")},
            "demotion_limit": round(float(R["demotion"]["dd_multiple"]) * float(ref["max_dd"]), 2)
            if ref.get("max_dd") else None,
            "family_size": fam, "z_required": round(z_required(R, fam), 4) if s["group"] == "forward" else None,
            "chain_ok": bool(chain["ok"]), "chain_error": chain.get("error"),
            "config_hash": s.get("config_hash"), "rule_sha256": R.get("_sha256"),
        },
    }
