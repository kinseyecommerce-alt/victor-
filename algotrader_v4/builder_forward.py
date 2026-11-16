"""
builder_forward.py — the forward lab's SECOND family, builder/v1 (frozen rule builder_forward_rule.json, sha256 pinned
below). Every strategy builder.py promotes is registered here and followed on each new session of real data; looks
every 20 sessions with a Bonferroni qualification over every builder strategy ever promoted (never shrinks),
consistency with the strategy's own builder reference and the drawdown demotion — forward_lab.py's statistics
(qualification_test, consistency_band, max_drawdown, the hash-chained Chain) reused unchanged. forward_lab_rule.json
and its 1,276-strategy family are not touched; forward_lab.daily_job only calls run_pending() at its end.

Read-only towards trading: no order entry point, agent, risk or trading state is imported, read or changed. 'ready'
(QUALIFIED, consistent, not demoted) is a flag for a human; nothing reads it to trade.

Data: the builder DATA LAYER's files (stratlab.builder_server_panels.ServerSource, BUILDER_DATA_DIR) cut at the day,
never the forward lab's own 97-name store: its minute store holds every name of the NIFTY 200 extension. A strategy
proven under builder/v2 is registered with its universe's frozen names ("universe_names") and trades exactly those;
one without (builder/v1) trades its implicit universe (the 97 on the minute grid, the top 500 on the daily grid).

Layout <builder root>/forward/: registry.jsonl (genesis, register, run, look, status), ledgers/<id>.jsonl (genesis +
one "day" record per session with closed trades), status.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
RULE_FILE = HERE / "builder_forward_rule.json"
RULE_SHA256 = "80d8bfa640e5ba5a2fe5b2e101c39603de0802ae09fec27361bc775533b621bb"
MAX_DAYS_PER_RUN = 10
import threading  # noqa: E402
_RUN_LOCK = threading.RLock()


def load_rule(path=RULE_FILE, expect: Optional[str] = None) -> dict:
    import forward_lab as FL
    sha = FL.file_sha256(path)
    want = RULE_SHA256 if expect is None else expect
    if want != "PENDING" and sha != want:
        raise FL.RuleChanged(f"builder_forward_rule.json sha256 {sha[:12]} != pinned {want[:12]}: the rule is frozen")
    r = json.loads(Path(path).read_text())
    r["_sha256"] = sha
    return r


class BuilderForward:
    def __init__(self, root, rule: Optional[dict] = None):
        import forward_lab as FL
        self.FL = FL
        self.root = Path(root)
        self.rule = rule if rule is not None else load_rule()
        import builder
        self.registry = builder.fresh_chain(self.root / "registry.jsonl")
        self._ledgers: dict = {}

    def ledger(self, sid: str):
        c = self._ledgers.get(sid)
        if c is None:
            c = self._ledgers[sid] = self.FL.Chain(self.root / "ledgers" / f"{self.FL.safe_name(sid)}.jsonl")
        return c

    def state(self) -> dict:
        st = {"genesis": None, "strategies": {}, "runs": {}, "looks": {}, "status": {}, "family": []}
        for rec in self.registry.records():
            k, d = rec["kind"], rec["data"]
            if k == "genesis":
                st["genesis"] = d
            elif k == "register":
                for s in d["strategies"]:
                    st["strategies"][s["id"]] = {**s, "registered_on": d["registered_on"]}
                    if s["id"] not in st["family"]:
                        st["family"].append(s["id"])
            elif k == "run":
                st["runs"][int(d["day"])] = d
            elif k == "look":
                st["looks"].setdefault(d["id"], {})[str(d["look"])] = d
            elif k == "status":
                st["status"][d["id"]] = d
        return st

    def family_size(self, st=None) -> int:
        st = st or self.state()
        return max(1, len(st["family"]))

    def alpha(self, fam: int) -> float:
        q = self.rule["qualification"]
        return float(q["alpha_family"]) / (max(1, fam) * int(self.rule["looks"]["max_looks"]))

    def z_required(self, fam: int) -> float:
        from statistics import NormalDist
        return NormalDist().inv_cdf(1.0 - self.alpha(fam))

    def ensure_genesis(self):
        st = self.state()
        if st["genesis"] is not None:
            if st["genesis"]["rule_sha256"] != self.rule["_sha256"]:
                raise self.FL.RuleChanged("builder/v1 registry started under a different rule file")
            return st["genesis"]
        g = {"family": self.rule["family"], "rule_version": self.rule["version"], "rule_sha256": self.rule["_sha256"]}
        self.registry.append("genesis", g)
        return g

    def register(self, table: list, first_day: int) -> dict:
        """table: [{"id", "batch", "cand", "reference", "fingerprint"}]; idempotent; the family only grows."""
        self.ensure_genesis()
        st = self.state()
        new = [s for s in table if s["id"] not in st["strategies"]]
        if new:
            rows = [{"id": s["id"], "batch": s["batch"], "cand": s["cand"], "fingerprint": s["fingerprint"],
                     "reference": s["reference"]} for s in new]
            for r, s in zip(rows, new):
                if s.get("universe_names"):            # the universe frozen at the proof (builder/v2 candidates)
                    r["universe_names"] = list(s["universe_names"])
            self.registry.append("register", {"registered_on": int(first_day), "strategies": rows,
                                              "family_size_after": len(st["family"]) + len(rows)})
            for s in rows:
                self.ledger(s["id"]).append("genesis", {"id": s["id"], "rule_sha256": self.rule["_sha256"],
                                                        "registered_on": int(first_day)})
        return {"registered": len(new)}

    # ── one session ──
    def ingest_day(self, day: int, trades_by_id: dict) -> dict:
        st = self.state()
        if int(day) in st["runs"]:
            raise RuntimeError(f"builder forward session {day} already ingested")
        n = 0
        for sid, s in st["strategies"].items():
            if int(day) <= int(s["registered_on"]):
                continue
            closed = [t for t in trades_by_id.get(sid, []) if int(t[6]) == int(day)
                      and int(t[2]) > int(s["registered_on"]) and not str(t[13]).startswith(("window_end", "period_end"))]
            if not closed:
                continue
            nets = [float(t[12]) for t in closed]
            self.ledger(sid).append("day", {"date": int(day), "trades": len(closed), "net": round(sum(nets), 2),
                                            "trades_sha256": hashlib.sha256(json.dumps(closed, default=str)
                                                                            .encode()).hexdigest()})
            n += len(closed)
        run = {"day": int(day), "status": "complete", "closed_trades": n, "strategies": len(st["strategies"])}
        self.registry.append("run", run)
        return run

    def series(self, sid: str, st: dict):
        s = st["strategies"][sid]
        days = sorted(d for d, r in st["runs"].items() if r["status"] == "complete" and d > int(s["registered_on"]))
        recs = {int(r["data"]["date"]): r["data"] for r in self.ledger(sid).records() if r["kind"] == "day"}
        return days, [float(recs.get(d, {}).get("net", 0.0)) for d in days], \
            [int(recs.get(d, {}).get("trades", 0)) for d in days]

    def evaluate(self, sid: str, st: dict, write: bool = True) -> dict:
        FL = self.FL
        R = self.rule
        s = st["strategies"][sid]
        cur = (st["status"].get(sid) or {}).get("status")
        days, nets, ntr = self.series(sid, st)
        ref = s.get("reference") or {}
        out = {"id": sid, "days": len(days), "trades": int(sum(ntr)), "net": round(sum(nets), 2), "new": []}
        if cur == "demoted":
            out["status"] = "demoted"
            return out
        recs = []
        fdd = FL.max_drawdown(nets)
        lab_dd = float(ref.get("max_dd") or 0.0)
        mult = float(R["demotion"]["dd_multiple"])
        if lab_dd > 0 and fdd > mult * lab_dd:
            recs.append(("status", {"id": sid, "status": "demoted", "day": days[-1],
                                    "reason": f"forward max DD {fdd:,.0f} > {mult} x reference {lab_dd:,.0f}"}))
            out["status"] = "demoted"
        else:
            looks = dict(st["looks"].get(sid, {}))
            every, mx = int(R["looks"]["every_sessions"]), int(R["looks"]["max_looks"])
            status = cur
            inconsistent = bool((st["status"].get(sid + "#consistency") or {}).get("status") == "inconsistent")
            for k in range(1, mx + 1):
                if str(k) in looks:
                    continue
                n = k * every
                if len(days) < n:
                    break
                sub, subtr = nets[:n], int(sum(ntr[:n]))
                fam = self.family_size(st)
                res = {"id": sid, "look": k, "day": days[n - 1], "days": n, "trades": subtr,
                       "net": round(sum(sub), 2), "family_size": fam, "alpha": self.alpha(fam),
                       "z_required": self.z_required(fam)}
                q = R["qualification"]
                if subtr < int(q["min_trades"]) or n < int(q["min_days"]):
                    res.update({"tested": False, "verdict": "insufficient"})
                else:
                    b = q["bootstrap"]
                    t = FL.qualification_test(sub, FL.look_seed(sid, k), int(b["resamples"]), float(b["mean_block"]))
                    ok = bool(t["mean"] > 0 and t["z"] is not None and t["z"] >= res["z_required"])
                    res.update({"tested": True, "z": t["z"], "p": t["p"], "verdict": "qualified" if ok
                                else "not_qualified"})
                    if ok and status != "qualified":
                        status = "qualified"
                        recs.append(("status", {"id": sid, "status": "qualified", "day": res["day"],
                                                "reason": f"look {k}: z={t['z']:.2f} >= {res['z_required']:.2f}"}))
                c = R["consistency"]
                if ref.get("daily") and subtr >= int(c["min_trades"]):
                    band = FL.consistency_band(ref["daily"], n, FL.look_seed(sid, k, "consistency"), 10000, 5.0,
                                               float(c["quantile"]))
                    bad = float(np.mean(sub)) < band
                    res.update({"band": round(band, 4), "consistency": "inconsistent" if bad else "consistent"})
                    if bad and not inconsistent:
                        inconsistent = True
                        recs.append(("status", {"id": sid + "#consistency", "status": "inconsistent",
                                                "day": res["day"], "reason": f"look {k}: mean below the 1% band"}))
                recs.append(("look", res))
                out["new"].append(k)
            out["status"] = status or "watching"
            out["inconsistent"] = inconsistent
        if write and recs:
            self.registry.append_many(recs)
        return out

    def evaluate_all(self, write: bool = True) -> dict:
        st = self.state()
        return {sid: self.evaluate(sid, st, write) for sid in st["strategies"]}

    def status_by_id(self) -> dict:
        st = self.state()
        out = {}
        for sid in st["strategies"]:
            stat = (st["status"].get(sid) or {}).get("status") or "watching"
            inc = (st["status"].get(sid + "#consistency") or {}).get("status") == "inconsistent"
            days, nets, ntr = self.series(sid, st)
            out[sid] = {"status": stat, "inconsistent": inc, "days": len(days), "trades": int(sum(ntr)),
                        "net": round(sum(nets), 2), "ready": bool(stat == "qualified" and not inc),
                        "registered_on": st["strategies"][sid]["registered_on"]}
        return out


# ══════════════════════════════════════════════════════════════════════════
# the daily step (called at the end of forward_lab.daily_job)
# ══════════════════════════════════════════════════════════════════════════
def forward_trades(root: str, data_root: Optional[str], day: int, source: str = "server") -> dict:
    """In the forward process: every registered strategy's trades over [registration, day] on data cut at `day`.
    Each strategy trades the size of its own sentence (its candidate JSON's size regime; a builder/v1 / v2 strategy:
    Rs 1 lakh per stock trade, a re-sized builder/v3 one: stratlab.builder_engine.SIZE_TABLES["c10"])."""
    from stratlab import builder_engine as BE
    from stratlab import builder_grammar as G
    from stratlab.builder_source import make_source
    from stratlab.builder_server_panels import ServerSource
    bf = BuilderForward(Path(root))
    st = bf.state()
    src = ServerSource(data_root, upto=int(day)) if source == "server" else make_source(source, data_root)
    E = BE.Engine(src)
    import builder
    E.cache_dir = str(builder.cache_dir())
    jobs = []
    for sid, s in st["strategies"].items():
        names = tuple(s["universe_names"]) if s.get("universe_names") else None
        jobs.append(BE.Job(sid, G.Candidate.from_json(s["cand"]), None, int(s["registered_on"]) + 1, int(day), False,
                           names=names))
    res = BE.run_jobs(E, jobs) if jobs else {}
    cal = sorted(set(int(x) for x in E.dcal) & set(int(x) for x in E.mcal))
    return {"day": int(day), "calendar": [d for d in cal if d <= int(day)],
            "trades": {k: [list(t) for t in v["trades"]] for k, v in res.items()}}


def run_pending(root=None, data_root: Optional[str] = None, runner=None, today: Optional[int] = None) -> dict:
    """Process the pending complete sessions of the builder/v1 family. No-op without a registered strategy or with
    BUILDER_ENABLED off."""
    import builder
    if runner is None and not builder.enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    root = Path(root) if root is not None else builder.default_root() / "forward"
    with _RUN_LOCK:
        return _run_pending(root, data_root, runner, today)


def _run_pending(root: Path, data_root, runner, today) -> dict:
    import builder
    reconciled = []
    if (root.parent / "registry.jsonl").exists():     # promoted strategies whose registration failed earlier
        try:
            reconciled = builder.Builder(root.parent).reconcile_forward()
        except Exception as exc:
            reconciled = [f"error: {exc}"[:200]]
    if not (root / "registry.jsonl").exists():
        return {"skipped": "no builder strategy registered", "reconciled": reconciled}
    bf = BuilderForward(root)
    st = bf.state()
    if not st["strategies"]:
        return {"skipped": "no builder strategy registered"}
    vr = bf.registry.verify()
    if not vr["ok"]:
        return {"status": "tamper", "error": vr["error"]}
    day = int(today) if today else int(datetime.now().strftime("%Y%m%d"))
    if runner is not None:
        res = runner(str(root), data_root, day)
    else:
        work = root / "work"
        work.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "builder_run", "--forward", "--root", str(root), "--day", str(day),
               "--source", "server"] + (["--data-root", str(data_root)] if data_root else [])
        s = builder._settings()
        p = subprocess.run(cmd, cwd=str(HERE), timeout=int(getattr(s, "builder_time_limit_s", 14400)),
                           env=dict(os.environ, BUILDER_CACHE_DIR=str(builder.cache_dir())), capture_output=True,
                           text=True, start_new_session=True)
        if p.returncode != 0:
            return {"status": "error", "error": (p.stderr or "")[-400:]}
        with gzip.open(work / f"forward_{day}.json.gz", "rt") as fh:
            res = json.load(fh)
    done = set(st["runs"])
    reg0 = min(int(s["registered_on"]) for s in st["strategies"].values())
    pending = [d for d in res["calendar"] if d > reg0 and d not in done and d > (max(done) if done else 0)]
    pending = pending[:MAX_DAYS_PER_RUN]
    trades = {k: [tuple(t) for t in v] for k, v in res["trades"].items()}
    out = []
    for d in pending:
        out.append(bf.ingest_day(d, trades))
    ev = bf.evaluate_all(write=True) if pending else {}
    status = {"processed": [r["day"] for r in out], "status": "complete" if out else "nothing to do",
              "reconciled": reconciled,
              "written": datetime.now().isoformat(timespec="seconds"),
              "new_status": [{"id": k, "status": v["status"]} for k, v in ev.items() if v.get("new")]}
    (root / "status.json").write_text(json.dumps(status))
    return status


def run_pending_locked(**kw) -> dict:
    """The scheduler's entry point (platform_scheduler job builder_forward, BUILDER_ENABLED, PAPER): run_pending holding
    the builder data layer's lock, so the data it reads are never half-written. Off -> returns at once."""
    import builder
    if not builder.enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    try:
        import builder_data
        lock = builder_data._LOCK
    except Exception:
        lock = threading.Lock()
    got = lock.acquire(timeout=3600)
    try:
        return run_pending(**kw)
    finally:
        if got:
            lock.release()
