"""
builder.py — the AUTOMATIC STRATEGY BUILDER (builder/v1 .. v3): every night (builder/v3; every week under v1 / v2)
it BUILDS 400 candidate strategies from our own data (5 agents x 4 market types x 20, stratlab/builder_grammar.py), freezes them in a hash-chained registry BEFORE any
return is computed, evaluates them in a separate process under the frozen rule builder_rule.json (sha256 pinned below)
and promotes only what is proven to the forward lab's separate builder/v1 family (builder_forward.py). PAPER research
only: no order entry point, agent, risk or trading state is imported, read or changed.

The job (weekly_job = ONE batch; since builder/v3 platform_scheduler runs nightly_job at 02:30 IST - Saturday 06:00
under v1 / v2 - which runs batch after batch until the grammar is exhausted, BUILDER_NIGHT_BATCHES ran, 05:15 IST
passed or a batch did not complete; only with BUILDER_ENABLED=true and PAPER; the job re-checks both and skips inside
market hours). One batch:
  1. freeze    draw the batch (deterministic in the batch id, never re-drawing a key of an earlier batch), write
               <root>/batches/<id>/candidates.json, append "batch_frozen" (candidate sha256, generator version, grammar
               sha256, data snapshot ids + sha256, N before) to <root>/registry.jsonl
  2. evaluate  python -m builder_run in its own process group, nice 10, <= 2 workers, address-space limit, wall-clock
               limit; it refuses a batch whose candidate file / snapshot does not match the registry; resumable
  3. ingest    "final_open" was appended by the run before any FINAL return; "batch_result" (summary, N_used) is
               appended after a successful run; candidates -> trials ledger (if configured); proven -> promotion
Disk: BUILDER_MAX_MB (default 400) for <root>; at the cap nothing is written (status disk_cap); nothing is deleted.

Rule versions (RULES, each pinned): builder/v1 (grammar bg1) and builder/v2 (grammar bg2 = bg1 x the stock UNIVERSE:
NIFTY100 / NIFTY200 / TOP500 / INDEX). One registry, one N: a registry started under v1 moves to v2 once (record
"rule_upgrade" carrying N, candidates and FINAL looks at that moment); v1 records are never rewritten, a v1 batch still
pending is evaluated under v1, new batches are frozen under v2 and never re-draw a v1 key (nor its bg2 twin). Under v2
a NEW batch waits until the data layer's NIFTY 200 minute universe is complete (data_wait_reason, status
"waiting_data"); a promoted strategy carries the universe's names as resolved on its batch's frozen data.
builder/v3 (grammar bg3 = bg2 under the size regime "c10", Rs 10 lakh capital per agent): MIS stock trades and stock
trades at futures costs are Rs 5 lakh per trade, option lots follow the capital (Rs 1 lakh of premium / of maximum
loss, Rs 5 lakh of naked margin: stratlab/builder_options.option_lots); CNC delivery longs (proportional fees) and
index-futures lots are unchanged - their existing results ARE their Rs 10 lakh results. The numbers are
stratlab/builder_engine.SIZE_TABLES + builder_options.OPTION_SIZES = the rule's costs.size_rs. A re-sized sentence has
a NEW bg3 key (drawn again, counted in N); a sentence of unchanged size is its bg2 sentence and is never re-drawn. A candidate carries its
size regime, so a v1 / v2 batch (pending or seeded) and a v1 / v2 promoted strategy keep Rs 1 lakh. A registry moves
forward along the chain v1 -> v2 -> v3 (one "rule_upgrade" record per step; a registry at v1 takes both steps at once).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
RULE_FILE = HERE / "builder_rule.json"
# The sandbox's batch 0001 (evaluated 2026-11-15 on the lab's data): imported once as the registry's first records so
# every candidate the builder ever made counts in N and is never drawn again (builder_seed_0001.json, sha256 pinned).
SEED_FILE = HERE / "builder_seed_0001.json"
SEED_SHA256 = "15c6c036cd8153ca8be6463be234f55efd3da33e86a4376911d73abd5b3c40d8"
# The sandbox's batches 0002-0006 (evaluated 2026-11-16 under builder/v1 on the lab's data), imported after seed 0001:
# N continues from 4,979, their 2,000 keys are never re-drawn, the server's own first batch is 0007, and the proven
# strategy of batch 0003 is imported as a "promoted" record (registered with the builder/v1 forward family by
# reconcile_forward on the next builder run). sha256 pinned; an edited file is refused (SeedChanged).
SEED2_FILE = HERE / "builder_seed_0002_0006.json"
SEED2_SHA256 = "91dac10b6366c056e5e270f71804ef7da494dac007459bd49ed916ea39c584d9"


def seed_files() -> list:
    """The pinned seed files in import order (read at call time, so a test can patch one)."""
    return [(SEED_FILE, SEED_SHA256), (SEED2_FILE, SEED2_SHA256)]
# sha256 of builder_rule.json as committed (the pre-registration). An edited rule file is refused.
RULE_SHA256 = "a455009021b9e2cfbc55106c8ee1310ac6a7135995a895b6b8cafda06fbc3c06"
# builder/v2 (grammar bg2: the stock UNIVERSE is a dimension of the draw), written next to v1, which stays valid and is
# never rewritten. New batches are frozen under the LAST pinned version; a batch is always evaluated under its own.
RULE_FILE_V2 = HERE / "builder_rule_v2.json"
RULE_SHA256_V2 = "7072377b40de2be492773868b962e8dcc10b7392fc099587fcd1b2ef53d45188"
# builder/v3 (grammar bg3: bg2 under the size regime "c10" = Rs 10 lakh capital per agent; nightly batches), written
# next to v1 / v2, which stay valid and are never rewritten.
RULE_FILE_V3 = HERE / "builder_rule_v3.json"
RULE_SHA256_V3 = "bcf23e8d3098427b097870c98a9d888053d2f50f28e86d83aa5456fa9d79f116"
# builder/v4 (26 Nov 2026): bg3 + trailing-stop exits for intraday / scalping (grammar bg4); gates, costs, DSR as v3
RULE_FILE_V4 = HERE / "builder_rule_v4.json"
RULE_SHA256_V4 = "385b3a15760773cb400f00d3688242a6793c3a66cf7e699ba212dec7b32cefbe"
# the pinned versions in order: a registry may only move forward along this chain (record "rule_upgrade")
RULES = (("builder/v1", RULE_FILE, RULE_SHA256), ("builder/v2", RULE_FILE_V2, RULE_SHA256_V2),
         ("builder/v3", RULE_FILE_V3, RULE_SHA256_V3), ("builder/v4", RULE_FILE_V4, RULE_SHA256_V4))
CURRENT_RULE_FILE, CURRENT_RULE_SHA256 = RULES[-1][1], RULES[-1][2]
DEFAULT_MAX_MB = 400.0
MAX_ATTEMPTS = 3            # failed evaluation runs of one batch before it is recorded `failed` (bookkeeping, not a gate)
NIGHT_STOP_MIN = 5 * 60 + 15   # builder/v3 nightly run: no batch is STARTED at / after 05:15 IST (nor in market hours)
NSE_GATE_LAG_DAYS = 10         # the NSE archive gate ignores the newest days (the data layer re-reads them every run)
_LOCK = threading.RLock()


class RuleChanged(RuntimeError):
    pass


class NotFrozen(RuntimeError):
    pass


class SeedChanged(RuntimeError):
    pass


def load_seed(path=None, expect: Optional[str] = None) -> dict:
    """The committed seed (sandbox batch 0001). Raises SeedChanged when its sha256 is not the pinned one."""
    path = Path(path) if path is not None else SEED_FILE
    sha = file_sha256(path)
    want = SEED_SHA256 if expect is None else expect
    if sha != want:
        raise SeedChanged(f"{path.name} sha256 {sha[:12]} != pinned {want[:12]}: the seed is frozen")
    d = json.loads(path.read_text())
    d["_sha256"] = sha
    return d


def seed_batches(doc: dict) -> list:
    """The batch documents of a seed file: a single-batch seed (0001) is its own batch, a multi-batch seed lists them."""
    if "batches" not in doc:
        return [doc]
    out = []
    for bd in doc["batches"]:
        out.append({"lab_trials_count": doc["lab_trials_count"],
                    "development_evaluations": {"count": int(doc["development_evaluations"])}, **bd})
    return out


def load_seeds() -> list:
    """Every pinned seed file, verified before anything is imported (one edited file refuses them all)."""
    return [(Path(p), load_seed(p, expect=s)) for p, s in seed_files()]


class DiskCapReached(RuntimeError):
    pass


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rule(path=None, expect: Optional[str] = None) -> dict:
    """A pinned rule file. No path: the CURRENT version (the last of RULES). A pinned file must have its own pinned
    sha256; any other path must have one of the pinned sha256s (an edited rule is refused)."""
    path = Path(path) if path is not None else CURRENT_RULE_FILE
    sha = file_sha256(path)
    if expect is None:
        pinned = {Path(f).resolve(): s for _v, f, s in RULES}
        want = pinned.get(path.resolve())
        ok = (sha == want) if want is not None else sha in {s for _v, _f, s in RULES}
    else:
        want = expect
        ok = want == "PENDING" or sha == want
    if not ok:
        raise RuleChanged(f"{path.name} sha256 {sha[:12]} != pinned {str(want)[:12]}: the rule is frozen")
    rule = json.loads(path.read_text())
    rule["_sha256"] = sha
    return rule


def rule_by_sha(sha: str) -> dict:
    """The pinned rule whose sha256 is `sha` (a batch is evaluated under the rule it was frozen with)."""
    for _v, f, s in RULES:
        if s == sha:
            return load_rule(f)
    raise RuleChanged(f"no pinned builder rule has sha256 {str(sha)[:12]}")


def rule_sizes(rule: Optional[dict]) -> dict:
    """Rs per stock trade of a rule's batches by cost kind (+ the size regime): builder/v1 / v2 carry no costs.size_rs
    (Rs 1 lakh everywhere, regime ""); builder/v3's is its costs.size_rs (regime "c10")."""
    rs = ((rule or {}).get("costs") or {}).get("size_rs")
    if rs:
        return {"regime": rs.get("regime"),
                "per_trade_rs": {k: v for k, v in rs.items() if k not in ("regime", "options")},
                "options": rs.get("options"), "unchanged": ((rule or {}).get("costs") or {}).get("unchanged")}
    return {"regime": "", "per_trade_rs": {"equity_mis": 100000, "future": 100000, "equity_cnc": 100000},
            "options": "1 lot", "unchanged": None}


def sizes_by_sha(sha: Optional[str]) -> dict:
    try:
        return rule_sizes(rule_by_sha(sha) if sha else None)
    except Exception:
        return rule_sizes(None)


def rule_index(sha: str) -> int:
    for k, (_v, _f, s) in enumerate(RULES):
        if s == sha:
            return k
    return -1


def _settings():
    from config import settings
    return settings


def enabled() -> bool:
    try:
        s = _settings()
        return bool(getattr(s, "builder_enabled", False)) and s.trading_mode == "PAPER"
    except Exception:
        return False


def default_root() -> Path:
    p = os.getenv("BUILDER_DIR")
    if p:
        return Path(p)
    perf = os.getenv("PERFORMANCE_FILE")
    base = Path(perf).parent if perf else HERE / "data"
    return base / "builder"


def cache_dir() -> Path:
    """Ephemeral caches (compact option tables): never on the volume unless the operator says so."""
    return Path(os.getenv("BUILDER_CACHE_DIR") or "/tmp/builder_cache")


def max_bytes() -> int:
    try:
        mb = float(getattr(_settings(), "builder_max_mb", DEFAULT_MAX_MB))
    except Exception:
        mb = float(os.getenv("BUILDER_MAX_MB", DEFAULT_MAX_MB))
    return int(mb * 1024 * 1024)


def dir_bytes(root) -> int:
    """Bytes this directory costs: a regular file with other hard links (a frozen batch's link to a data-layer minute
    file) costs nothing until it is the last link (the data layer replaced its file), then it counts in full."""
    tot = 0
    for dp, _dn, fn in os.walk(str(root)):
        for f in fn:
            try:
                st = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            if st.st_nlink > 1 and not os.path.islink(os.path.join(dp, f)):
                continue
            tot += st.st_size
    return tot


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_obj(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def _jsave(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, path)


def _jload(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════════════
# the builder (registry + batches)
# ══════════════════════════════════════════════════════════════════════════
def fresh_chain(path):
    """forward_lab.Chain whose tail is re-read on every append under an exclusive file lock: the registry has two
    writers (the app process and the evaluation process), so a cached tail would fork the chain."""
    import fcntl
    from forward_lab import Chain

    class FreshChain(Chain):
        def append_many(self, items, ts=None):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.path) + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                self._tail = None
                try:
                    return super().append_many(items, ts)
                finally:
                    self._tail = None
    return FreshChain(path)


class Builder:
    def __init__(self, root=None, rule: Optional[dict] = None):
        self.root = Path(root) if root is not None else default_root()
        self.rule = rule if rule is not None else load_rule()
        self.registry = fresh_chain(self.root / "registry.jsonl")

    # ── paths ──
    def batch_dir(self, bid: str) -> Path:
        return self.root / "batches" / str(bid)

    def candidates_path(self, bid: str) -> Path:
        return self.batch_dir(bid) / "candidates.json"

    def result_path(self, bid: str) -> Path:
        return self.batch_dir(bid) / "result.json.gz"

    def check_cap(self, extra: int = 0) -> None:
        if dir_bytes(self.root) + extra >= max_bytes():
            raise DiskCapReached(f"builder directory at the cap ({max_bytes() // 1048576} MB)")

    # ── folded registry state ──
    def state(self) -> dict:
        st = {"genesis": None, "batches": {}, "final_open": {}, "results": {}, "promoted": [], "n_used": 0,
              "evaluated": 0, "final_looks": 0, "attempts": {}, "rules": [], "upgrades": []}
        for rec in self.registry.records():
            k, d = rec["kind"], rec["data"]
            if k == "genesis":
                st["genesis"] = d
                st["rules"] = [d["rule_sha256"]]
            elif k == "rule_upgrade":
                st["rules"].append(d["to_sha256"])
                st["upgrades"].append({**d, "ts": rec["ts"], "seq": rec["seq"]})
            elif k == "batch_frozen":
                st["batches"][d["batch"]] = {**d, "frozen_seq": rec["seq"], "frozen_hash": rec["hash"],
                                             "created": rec["ts"]}
            elif k == "final_open":
                st["final_open"].setdefault(d["batch"], []).extend(d["ids"])
            elif k == "batch_result":
                st["results"][d["batch"]] = {**d, "ts": rec["ts"]}
                st["evaluated"] += int(d["candidates"])
                st["n_used"] = max(st["n_used"], int(d["n_used"]))
            elif k == "promoted":
                st["promoted"].append({**d, "ts": rec["ts"], "hash": rec["hash"], "forward_registered": False})
            elif k in ("forward_registered", "forward_register_failed"):
                for pr in st["promoted"]:
                    if pr["id"] == d["id"]:
                        if k == "forward_registered":
                            pr["forward_registered"], pr["forward_error"] = True, None
                        else:
                            pr["forward_error"] = d.get("error")
            elif k == "attempt":
                st["attempts"][d["batch"]] = int(d["n"])
            elif k == "batch_failed":
                st["results"][d["batch"]] = {**d, "failed": True, "ts": rec["ts"], "proven": []}
                st["evaluated"] += int(d["candidates"])
                st["n_used"] = max(st["n_used"], int(d["n_used"]))
            elif k == "erratum":
                st.setdefault("errata", {}).setdefault(d["batch"], []).append({**d, "ts": rec["ts"]})
            elif k == "seed":
                bid = d["batch"]
                sd = {**d, "ts": rec["ts"], "hash": rec["hash"]}
                st.setdefault("seed", sd)                    # the first seed (0001): kept for older readers
                st.setdefault("seeds", {})[bid] = sd
                n_before = d.get("n_before")
                if n_before is None:
                    n_before = d["lab_trials_count"] + d["development_evaluations"]
                st["batches"][bid] = {"batch": bid, "seeded": True, "candidates_sha256": d["candidates_sha256"],
                                      "n_candidates": d["n_candidates"], "snapshot_end": d["snapshot_end"],
                                      "snapshot_sha256": d["snapshot_sha256"], "source": "sandbox seed",
                                      "generator": {"grammar_version": d["grammar_version"],
                                                    "grammar_sha256": d["grammar_sha256"]},
                                      "rule_sha256": d.get("rule_sha256"), "n_before": int(n_before),
                                      "frozen_seq": rec["seq"], "frozen_hash": d["frozen_hash"], "created": rec["ts"]}
                if d.get("cell_universes"):
                    st["batches"][bid]["cell_universes"] = d["cell_universes"]
                st["final_open"][bid] = [f["id"] for f in d["final_looks"]]
                st["results"][bid] = {**{k2: v for k2, v in d["counts"].items() if k2 != "stages"}, "batch": bid,
                                      "proven": list(d.get("proven") or []), "n_used": d["n_used_after"],
                                      "best_discovery_dsr": d.get("best_discovery_dsr"),
                                      "ts": rec["ts"], "seeded": True}
                st["evaluated"] += int(d["n_candidates"])
                st["n_used"] = max(st["n_used"], int(d["n_used_after"]))
                st["seed_keys"] = list(st.get("seed_keys") or []) + list(d["keys"])
                for e in d.get("errata") or []:
                    st.setdefault("errata", {}).setdefault(bid, []).append({**e, "batch": bid})
        st["final_looks"] = sum(len(v) for v in st["final_open"].values())
        return st

    def ensure_genesis(self) -> dict:
        st = self.state()
        if st["genesis"] is not None:
            self.ensure_rule(st)
            self.ensure_seed()
            return st["genesis"]
        from stratlab import builder_grammar as G
        gv = self.rule["grammar"]["version"]
        g = {"rule_version": self.rule["version"], "rule_sha256": self.rule["_sha256"],
             "grammar_version": gv, "grammar_sha256": G.grammar_sha256(gv),
             "lab_trials_count": int(self.rule["dsr"]["lab_trials_count"])}
        self.registry.append("genesis", g)
        self.ensure_seed()
        return g

    def ensure_rule(self, st: Optional[dict] = None) -> None:
        """The registry's rule must be this builder's rule. A registry under an EARLIER pinned version moves forward
        along the chain, one "rule_upgrade" record per step (v1 -> v2 -> v3: a v1 registry opened under v3 gets two
        records), each with N, candidates and FINAL looks at that moment: N continues from it, nothing is recounted
        or rewritten; any other mismatch is refused (RuleChanged)."""
        st = st or self.state()
        chain = st["rules"] or [st["genesis"]["rule_sha256"]]
        cur = self.rule["_sha256"]
        if cur == chain[-1]:
            return
        if cur in chain:
            raise RuleChanged("the builder registry already moved to a later rule: new batches need that rule")
        i_last, i_cur = rule_index(chain[-1]), rule_index(cur)
        if i_last < 0 or i_cur <= i_last:
            raise RuleChanged("the builder registry was started under a different rule file")
        from stratlab import builder_grammar as G
        steps = [self.rule if k == i_cur else load_rule(RULES[k][1]) for k in range(i_last + 1, i_cur + 1)]
        for r in steps:                      # every step is checked before the first record is written
            if int(r["dsr"]["lab_trials_count"]) != int(st["genesis"]["lab_trials_count"]):
                raise RuleChanged("a rule upgrade may not change the lab trials count (N never shrinks)")
            if int(r["dsr"].get("development_evaluations", 0)) < int(self._dev_evals(chain[-1])):
                raise RuleChanged("a rule upgrade may not lower the development evaluations (N never shrinks)")
        n_before = max(st["n_used"], self.n_base(st))
        for k, r in zip(range(i_last + 1, i_cur + 1), steps):
            gv = r["grammar"]["version"]
            self.registry.append("rule_upgrade", {
                "from_version": RULES[k - 1][0], "from_sha256": RULES[k - 1][2], "to_version": r["version"],
                "to_sha256": r["_sha256"], "grammar_version": gv, "grammar_sha256": G.grammar_sha256(gv),
                "batches_before": len(st["batches"]), "evaluated_before": st["evaluated"],
                "final_looks_before": st["final_looks"], "n_used_before": n_before})

    @staticmethod
    def _dev_evals(sha: str) -> int:
        try:
            return int(rule_by_sha(sha)["dsr"].get("development_evaluations", 0))
        except Exception:
            return 0

    def ensure_seed(self, path=None) -> Optional[dict]:
        """Import the sandbox seeds once, in order (hash-chained, idempotent; every file is verified against its pinned
        sha256 BEFORE anything is written). A seed batch is skipped in the registry it was taken from (that registry's
        batch carries the seed's frozen hash). A seed's promoted strategies become "promoted" records (seeded=True):
        reconcile_forward registers them with the builder/v1 forward family. `path` (tests): import only that file,
        verified against the first pin. Returns the first seed record (None in the source registry)."""
        files = [(Path(path), load_seed(path))] if path is not None else load_seeds()
        st = self.state()
        done = {s["frozen_hash"] for s in (st.get("seeds") or {}).values()}
        bid_of = {s["frozen_hash"]: s["batch"] for s in (st.get("seeds") or {}).values()}
        wrote = False
        for fpath, doc in files:
            for seed in seed_batches(doc):
                st = self.state()
                if seed["frozen_hash"] in done:
                    continue
                fb = st["batches"].get(seed["batch"])
                if fb is not None and fb.get("frozen_hash") == seed["frozen_hash"]:
                    bid_of[seed["frozen_hash"]] = seed["batch"]
                    continue                              # the source registry itself
                bid = seed["batch"] if seed["batch"] not in st["batches"] else "sandbox-" + seed["batch"]
                keys = [c["key"] for c in seed["candidates"]]
                if len(keys) != int(seed["n_candidates"]) or len(set(keys)) != len(keys):
                    raise SeedChanged("seed candidate keys are not unique / complete")
                rec = {"batch": bid, "file": fpath.name, "file_sha256": doc["_sha256"],
                       "frozen_hash": seed["frozen_hash"], "candidates_sha256": seed["candidates_sha256"],
                       "snapshot_sha256": seed["snapshot_sha256"], "snapshot_end": seed["snapshot_end"],
                       "result_sha256": seed["result_sha256"], "rule_sha256": seed["rule_sha256"],
                       "grammar_version": seed["grammar_version"], "grammar_sha256": seed["grammar_sha256"],
                       "n_candidates": int(seed["n_candidates"]), "counts": seed["counts"],
                       "final_looks": [{k: f[k] for k in ("id", "confirmed", "trades", "pf", "net")}
                                       for f in seed["final_looks"]],
                       "lab_trials_count": int(seed["lab_trials_count"]),
                       "development_evaluations": int(seed["development_evaluations"]["count"]),
                       "n_used_after": int(seed["n_used_after"]), "errata": seed.get("errata", []),
                       "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(), "keys": keys}
                for k in ("n_before", "proven", "best_discovery_dsr", "cell_universes", "planted_gates"):
                    if seed.get(k) is not None:
                        rec[k] = seed[k]
                self.registry.append("seed", rec)
                done.add(seed["frozen_hash"])
                bid_of[seed["frozen_hash"]] = bid
                wrote = True
            have = {p["id"] for p in self.state()["promoted"]}
            for p in doc.get("promoted") or []:
                if p["id"] in have:
                    continue                              # the source registry (or imported before)
                self.registry.append("promoted", {**p, "batch": bid_of.get(p["frozen_hash"], p["batch"]),
                                                  "seeded": True, "seed_file": fpath.name,
                                                  "seed_file_sha256": doc["_sha256"]})
                wrote = True
        st = self.state()
        if wrote and st["promoted"]:
            write_catalog(self)
        return st.get("seed")

    def drawn_keys(self, st: Optional[dict] = None) -> set:
        st = st or self.state()
        keys = set(st.get("seed_keys") or [])
        for bid in st["batches"]:
            doc = _jload(self.candidates_path(bid), {}) or {}
            keys.update(c["key"] for c in doc.get("candidates", []))
        return keys

    def next_batch_id(self, st: Optional[dict] = None) -> str:
        st = st or self.state()
        return f"{len(st['batches']) + 1:04d}"

    def lab_count(self) -> int:
        # lab_retest_trials (builder/v3): the lab's own re-test of its catalogue at the rule's size regime
        base = int(self.rule["dsr"]["lab_trials_count"]) + int(self.rule["dsr"].get("lab_retest_trials", 0) or 0)
        path = os.getenv("BUILDER_TRIALS_LEDGER") or os.getenv("STRATLAB_TRIALS")
        if not path or not os.path.exists(path):
            return base
        try:
            from stratlab import trials as TR
            return max(base, TR.count_trials(TR.TrialsLedger(path).read(), "2023-26"))
        except Exception:
            return base

    def n_base(self, st: Optional[dict] = None) -> int:
        """N before a new batch: lab count + every builder candidate evaluated + every FINAL look so far."""
        st = st or self.state()
        return max(self.lab_count() + int(self.rule["dsr"].get("development_evaluations", 0)) + st["evaluated"]
                   + st["final_looks"], 0)

    # ── 1. freeze (the automatic pre-registration) ──
    def freeze(self, snapshot: dict, batch_id: Optional[str] = None, per_cell: Optional[int] = None,
               inputs: Optional[dict] = None) -> dict:
        from stratlab import builder_grammar as G
        from stratlab.builder_source import snapshot_sha
        with _LOCK:
            self.ensure_genesis()
            st = self.state()
            bid = batch_id or self.next_batch_id(st)
            if bid in st["batches"]:
                return st["batches"][bid]
            self.check_cap()
            gv = self.rule["grammar"]["version"]
            cands = G.draw_batch(bid, self.drawn_keys(st), per_cell or int(self.rule["batch"]["per_cell"]), version=gv)
            rows = []
            per: dict = {}
            unis: dict = {}
            for c in cands:
                k = per.get(c.cell, 0)
                per[c.cell] = k + 1
                cu = unis.setdefault(c.cell, {})
                cu[c.stock_universe] = cu.get(c.stock_universe, 0) + 1
                rows.append({"id": G.candidate_id(bid, c, k), "key": c.key, "cand": c.to_json(), "text": G.describe(c)})
            doc = {"batch": bid, "generator": {"grammar_version": gv, "grammar_sha256": G.grammar_sha256(gv)},
                   "rule_sha256": self.rule["_sha256"], "snapshot": snapshot, "candidates": rows}
            p = self.candidates_path(bid)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(canon(doc))
            rec = {"batch": bid, "candidates_sha256": file_sha256(p), "n_candidates": len(rows),
                   "cells": per, "generator": doc["generator"], "rule_sha256": self.rule["_sha256"],
                   "rule_version": self.rule["version"], "cell_universes": unis,
                   "snapshot_sha256": snapshot_sha(snapshot), "snapshot_end": snapshot.get("end"),
                   "source": snapshot.get("source"), "n_before": self.n_base(st)}
            if inputs:
                rec["inputs"] = inputs                   # the immutable inputs the evaluation runs on (builder_bundle)
            self.registry.append("batch_frozen", rec)
            return self.state()["batches"][bid]

    def verify_frozen(self, bid: str, snapshot: Optional[dict] = None) -> dict:
        """Raises NotFrozen unless the batch's candidate file (and, if given, the data snapshot) equals its registry
        record. Returns the candidate document."""
        from stratlab.builder_source import snapshot_sha
        st = self.state()
        b = st["batches"].get(bid)
        if b is None:
            raise NotFrozen(f"batch {bid} is not in the registry: freeze it before any evaluation")
        p = self.candidates_path(bid)
        if not p.exists() or file_sha256(p) != b["candidates_sha256"]:
            raise NotFrozen(f"batch {bid}: candidate file differs from its registry record")
        vr = self.registry.verify()
        if not vr["ok"]:
            raise NotFrozen(f"registry chain broken: {vr['error']}")
        if snapshot is not None and snapshot_sha(snapshot) != b["snapshot_sha256"]:
            raise NotFrozen(f"batch {bid}: data snapshot changed since the freeze")
        return json.loads(p.read_text())

    # ── frozen inputs, attempts, terminal failure ──
    def inputs_dir(self, bid: str, st: Optional[dict] = None) -> Optional[Path]:
        st = st or self.state()
        inp = (st["batches"].get(bid) or {}).get("inputs")
        return (self.root / "inputs" / inp["dir"]) if inp else None

    def inputs_problem(self, bid: str, st: Optional[dict] = None) -> Optional[str]:
        """None when the batch's frozen inputs are present and unchanged (or it has none); else what is wrong."""
        st = st or self.state()
        inp = (st["batches"].get(bid) or {}).get("inputs")
        if not inp:
            return None
        d = self.root / "inputs" / inp["dir"]
        mp = d / "manifest.json"
        if not mp.exists():
            return "frozen inputs missing"
        if file_sha256(mp) != inp["manifest_sha256"]:
            return "frozen inputs changed"
        return None

    def record_attempt(self, bid: str, error: str) -> int:
        n = int(self.state()["attempts"].get(bid, 0)) + 1
        self.registry.append("attempt", {"batch": bid, "n": n, "error": str(error)[:400]})
        return n

    def fail(self, bid: str, reason: str) -> dict:
        """Terminal failure: the batch is closed `failed`; every candidate stays counted in N and in the never-redraw
        set (its candidate file is kept), every FINAL look already opened stays counted, the queue moves on."""
        with _LOCK:
            st = self.state()
            if bid in st["results"]:
                return st["results"][bid]
            fb = st["batches"][bid]
            n = int(fb["n_candidates"])
            n_used = max(st["n_used"], self.n_base(st) + n)
            rec = {"batch": bid, "reason": str(reason)[:400], "attempts": int(st["attempts"].get(bid, 0)),
                   "candidates": n, "final_looks": list(st["final_open"].get(bid, [])), "n_used": n_used,
                   "discovery_passers": None, "shuffle_passers": None, "validation_tested": None,
                   "validation_passers": None, "final_confirmed": None, "best_discovery_dsr": None}
            self.registry.append("batch_failed", rec)
            d = self.inputs_dir(bid, st)
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)
            return self.state()["results"][bid]

    def _register_forward(self, p: dict) -> bool:
        try:
            import builder_forward
            row = {"id": p["id"], "batch": p["batch"], "cand": p["cand"], "reference": p["reference"],
                   "fingerprint": p.get("fingerprint") or sha256_obj(p["cand"])[:16]}
            if p.get("universe_names"):
                row["universe_names"] = list(p["universe_names"])
            builder_forward.BuilderForward(self.root / "forward").register([row], first_day=int(p.get("first_day") or 0))
        except Exception as exc:
            self.registry.append("forward_register_failed", {"id": p["id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False
        self.registry.append("forward_registered", {"id": p["id"]})
        return True

    def reconcile_forward(self) -> list:
        """Register every promoted strategy the builder/v1 family does not hold yet (a failed or interrupted
        registration); idempotent. Returns the ids registered now."""
        st = self.state()
        todo = [p for p in st["promoted"] if not p.get("forward_registered")]
        if not todo:
            return []
        done = []
        import builder_forward
        fw_root = self.root / "forward"
        have = set()
        if (fw_root / "registry.jsonl").exists():
            have = set(builder_forward.BuilderForward(fw_root).state()["strategies"])
        for p in todo:
            if p["id"] in have:
                self.registry.append("forward_registered", {"id": p["id"], "found": True})
                done.append(p["id"])
            elif self._register_forward(p):
                done.append(p["id"])
        return done

    def erratum(self, bid: str, note: dict) -> dict:
        """A recorded defect of an evaluated batch. Results are never recomputed (that would be a second look):
        the batch's candidates keep counting in N and are never re-drawn; the note says what the defect affected."""
        if bid not in self.state()["batches"]:
            raise NotFrozen(bid)
        return self.registry.append("erratum", {"batch": bid, **note})

    def open_final(self, bid: str, ids: list) -> None:
        st = self.state()
        have = set(st["final_open"].get(bid, []))
        new = [i for i in ids if i not in have]
        if new:
            self.registry.append("final_open", {"batch": bid, "ids": new,
                                                "final_looks_after": st["final_looks"] + len(new)})

    # ── 3. ingest ──
    def ingest(self, bid: str, summary: dict, trials_path: Optional[str] = None) -> dict:
        with _LOCK:
            st = self.state()
            if bid in st["results"]:
                return st["results"][bid]
            if bid not in st["batches"]:
                raise NotFrozen(bid)
            self.check_cap(200_000)
            blob = gzip.compress(canon(summary).encode(), mtime=0)
            self.result_path(bid).write_bytes(blob)
            n_used = max(st["n_used"], int(summary["n_proof"]))
            rec = {"batch": bid, "candidates": int(summary["candidates"]),
                   "discovery_passers": summary["discovery_passers"], "shuffle_passers": summary["shuffle_passers"],
                   "validation_tested": summary["validation_tested"], "validation_passers": summary["validation_passers"],
                   "final_looks": summary["final_looks"], "final_confirmed": summary["final_confirmed"],
                   "proven": summary["proven"], "n_disc": summary["n_disc"], "n_proof": summary["n_proof"],
                   "n_used": n_used, "v": summary["v"], "best_discovery_dsr": summary["best_discovery_dsr"],
                   "planted_gates": {k: v for k, v in summary["planted"].items() if k.startswith("gate_")},
                   "result_sha256": hashlib.sha256(blob).hexdigest(), "seconds": summary.get("seconds"),
                   "peak_rss_mb": summary.get("peak_rss_mb")}
            self.registry.append("batch_result", rec)
            d = self.inputs_dir(bid, st)
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)        # the batch is evaluated: its frozen inputs are done
            out = {"result": rec}
            if trials_path:
                out["trials_appended"] = append_trials(trials_path, bid, summary)
            for cid in summary["proven"]:
                out.setdefault("promoted", []).append(self.promote(bid, cid, summary))
            return out

    # ── 4. promotion ──
    def promote(self, bid: str, cid: str, summary: dict) -> dict:
        doc = json.loads(self.candidates_path(bid).read_text())
        row = next(r for r in doc["candidates"] if r["id"] == cid)
        res = next(r for r in summary["rows"] if r["id"] == cid)
        if res["stage"] != "proven":
            raise ValueError(f"{cid} is not proven: never promoted")
        st = self.state()
        if any(p["id"] == cid for p in st["promoted"]):
            return {"id": cid, "already": True}
        ref = reference_series(self, bid, cid, res)
        rec = {"id": cid, "batch": bid, "key": row["key"], "cand": row["cand"], "text": row["text"],
               "frozen_hash": st["batches"][bid]["frozen_hash"], "dsr": res["proof"]["dsr"]["dsr"],
               "reference": ref, "first_day": int(summary["windows"]["T"]),
               "fingerprint": sha256_obj(row["cand"])[:16],
               "rule_sha256": st["batches"][bid].get("rule_sha256")}
        from stratlab import builder_grammar as G
        rec["universe"] = G.Candidate.from_json(row["cand"]).stock_universe
        names = (res.get("proof") or {}).get("universe_names")
        if names:                      # the universe as resolved on the batch's frozen data: the forward step trades it
            rec["universe_names"] = list(names)
        # persisted as NOT yet registered with builder/v1: reconcile_forward() retries on every batch and daily
        # forward run until a "forward_registered" record exists
        self.registry.append("promoted", rec)
        write_catalog(self)
        rec["forward_registered"] = self._register_forward(rec)
        return rec


def reference_series(b: Builder, bid: str, cid: str, row: Optional[dict] = None) -> dict:
    """The builder's own reference for the forward family: VALIDATION + FINAL daily nets (zeros on sessions without
    an exit, from the proof; else the exit days of the run's trade files)."""
    import forward_lab as FL
    daily = ((row or {}).get("proof") or {}).get("reference_daily")
    if daily:
        return {"daily": daily, "net": round(sum(daily), 2), "max_dd": FL.max_drawdown(daily), "sessions": len(daily)}
    from stratlab import builder_eval as EV
    work = b.batch_dir(bid) / "work"
    tr = EV.read_trades(str(work / "trades" / f"val_{cid}.csv.gz")) + \
        EV.read_trades(str(work / "trades" / f"fin_{cid}.csv.gz"))
    by: dict = {}
    for t in tr:
        by[t[6]] = by.get(t[6], 0.0) + t[12]
    days = sorted(by)
    nets = [round(by[d], 2) for d in days]
    import forward_lab as FL
    return {"days": days, "daily": nets, "trades": len(tr), "net": round(sum(nets), 2),
            "max_dd": FL.max_drawdown(nets)}


def write_catalog(b: Builder) -> Path:
    """<root>/catalog/builder_gen.py — generated by code, one spec per promoted strategy (never hand-edited)."""
    st = b.state()
    specs = []
    for p in st["promoted"]:
        sp = {"id": p["id"], "batch": p["batch"], "key": p["key"], "cand": p["cand"], "text": p["text"],
              "frozen_hash": p["frozen_hash"], "registry_hash": p["hash"]}
        if p.get("universe_names"):
            sp["universe_names"] = list(p["universe_names"])
        if p.get("rule_sha256"):
            sp["rule_sha256"] = p["rule_sha256"]
        specs.append(sp)
    body = ('"""GENERATED by builder.write_catalog — do not edit. Promoted builder strategies (catalogue-compatible\n'
            'specs: stratlab.builder_catalog.load() turns each into a runnable spec)."""\n'
            f"RULE_SHA256 = {b.rule['_sha256']!r}\n"
            f"SPECS = {json.dumps(specs, indent=1, sort_keys=True)}\n")
    p = b.root / "catalog" / "builder_gen.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(body)
    os.replace(tmp, p)
    return p


def append_trials(path: str, bid: str, summary: dict) -> int:
    """Every candidate's DISCOVERY row, every VALIDATION screen and FINAL look -> the lab trials ledger (era
    builder-v1 / -v2 / -v3 by the batch's rule; the cost model is builder/v1's for v1 / v2 batches and builder/v3's
    (the re-sized trades) for v3 batches). Backed up first."""
    from stratlab import trials as TR
    led = TR.TrialsLedger(path)
    if os.path.exists(path):
        shutil.copyfile(path, path + f".bak_builder_{bid}_" + time.strftime("%Y%m%d_%H%M%S"))
    W = summary["windows"]
    vtag = str(summary.get("rule_version") or "builder/v1").split("/")[-1]          # v1 | v2 | v3
    cost_model = "builder/v3" if vtag == "v3" else "builder/v1"
    rows = []
    keep = ("trades", "net", "pf", "mean_net", "std_net", "sr_trade", "mean_gross", "mean_cost", "n_days",
            "sr_daily", "skew_daily", "kurt_daily")
    for r in summary["rows"]:
        base = {"strategy_id": r["id"], "family": "builder", "group": f"{r['agent']}|{r['mtype']}",
                "params_fp": "k:" + hashlib.sha1(r["key"].encode()).hexdigest()[:12], "era": f"builder-{vtag}",
                "dataset": f"builder_{bid}", "cost_model": cost_model}
        d = r["discovery"]
        rows.append({**base, "protocol": f"builder {vtag} discovery", "stage": "builder_discovery",
                     "period": f"0-{W['discovery'][1]}", "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                     "sr_daily": d["sr_daily"], "verdict": "PASSED GATES" if d["passed"] else "FAILED GATES",
                     "source": f"builder/{bid}#{r['id']}", "note": "; ".join(d["why"])})
        if r.get("validation"):
            s = r["validation"]["series"]
            rows.append({**base, **{k: s.get(k) for k in keep}, "protocol": f"builder {vtag} validation (protocol.screen)",
                         "stage": "builder_validation", "period": "%d-%d" % tuple(W["validation"]),
                         "verdict": "PASSED SCREEN" if r["validation"]["passed"] else "FAILED SCREEN",
                         "source": f"builder/{bid}#{r['id']}",
                         "note": "; ".join(r["validation"]["screen"]["reasons"])})
        if r.get("final"):
            s = r["final"]["series"]
            rows.append({**base, **{k: s.get(k) for k in keep}, "protocol": f"builder {vtag} final",
                         "stage": "builder_final", "period": "%d-%d" % tuple(W["final"]),
                         "verdict": "CONFIRMED" if r["final"]["confirmed"] else "REJECTED ON FINAL",
                         "source": f"builder/{bid}#{r['id']}", "note": ""})
    return led.append(rows)


# ══════════════════════════════════════════════════════════════════════════
# 2. the evaluation process
# ══════════════════════════════════════════════════════════════════════════
class RunFailed(RuntimeError):
    pass


# One thread per numeric library and two malloc arenas in the builder's child processes: the address-space limit
# (RLIMIT_AS) counts every thread's stack and every arena's reservation, which on an 8-core box is hundreds of MB
# that hold no data.
PROCESS_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"}


def run_process(root: Path, bid: str, source: str = "server", jobs: int = 1, time_limit_s: int = 14400,
                mem_mb: int = 3584, data_root: Optional[str] = None) -> dict:
    """python -m builder_run in its own process group (never inside the API process). Raises RunFailed on a timeout,
    a crash, the memory limit or an error. Returns the run's summary (also kept in the work directory)."""
    work = Path(root) / "batches" / bid / "work"
    work.mkdir(parents=True, exist_ok=True)
    d = Builder(root).inputs_dir(bid)
    if d is not None:                              # always evaluate on the inputs frozen with the batch
        source, data_root = "bundle", str(d)
    cmd = [sys.executable, "-m", "builder_run", "--root", str(root), "--batch", bid, "--source", source,
           "--jobs", str(max(1, min(2, int(jobs)))), "--mem-mb", str(int(mem_mb))]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    env = dict(os.environ, BUILDER_CACHE_DIR=str(cache_dir()), **PROCESS_ENV)
    t0 = time.time()
    with open(work / "run.log", "a") as logf:
        p = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=logf, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        try:
            p.wait(timeout=int(time_limit_s))
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            p.wait()
            raise RunFailed(f"time limit {time_limit_s}s exceeded (resumable: finished units are kept)")
    if p.returncode != 0:
        err = (work / "error.txt").read_text()[:600] if (work / "error.txt").exists() else f"exit {p.returncode}"
        raise RunFailed(f"builder_run exit {p.returncode}: {err}")
    with gzip.open(work / "summary.json.gz", "rt") as fh:
        res = json.load(fh)
    res["process_wall"] = round(time.time() - t0, 1)
    return res


# ══════════════════════════════════════════════════════════════════════════
# the batch job (nightly since builder/v3; the name weekly_job is kept for its callers)
# ══════════════════════════════════════════════════════════════════════════
def _status_path(root: Path) -> Path:
    return Path(root) / "status.json"


class _Done(Exception):
    """weekly_job finished early with its status already set (a terminal batch failure)."""


def weekly_job(now: Optional[datetime] = None, root=None, source: str = "server", runner=None,
               snapshot_fn=None, time_limit_s: Optional[int] = None) -> dict:
    """ONE batch (resume the pending one, else freeze + evaluate + ingest a new one). Off (flag false or LIVE) ->
    returns at once: nothing read or written. A new batch is not frozen while the data layer is incomplete (status
    "waiting_data") or when the grammar has no undrawn sentence left (status "grammar_exhausted")."""
    if not enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    import ist_clock
    now = now or ist_clock.now_ist()
    if ist_clock.is_trading_day(now.date()) and 9 * 60 <= now.hour * 60 + now.minute <= 15 * 60 + 45:
        return {"skipped": "market hours"}
    root = Path(root) if root is not None else default_root()
    status = {"started": now.isoformat(timespec="seconds"), "status": "running"}
    try:
        b = Builder(root)
        b.ensure_genesis()
        vr = b.registry.verify()
        if not vr["ok"]:
            status.update({"status": "tamper", "error": vr["error"]})
            return status
        status["forward_reconciled"] = b.reconcile_forward()
        st = b.state()
        pending = [bid for bid in st["batches"] if bid not in st["results"]]
        from stratlab.builder_source import snapshot_sha

        def live_differs(bid_):
            live = snapshot_fn() if snapshot_fn else snapshot_process(root, source, bundle=False)[0]
            return snapshot_sha(live) != st["batches"][bid_]["snapshot_sha256"]
        if pending:
            bid = sorted(pending)[0]                      # resume an unfinished batch before a new one
            problem = b.inputs_problem(bid, st)
            if problem and live_differs(bid):
                b.fail(bid, f"{problem} and the live data changed since the freeze: it can never be evaluated "
                            f"on its frozen inputs")
                status.update({"status": "failed", "batch": bid, "reason": problem})
                raise _Done()
        else:
            if snapshot_fn:
                snap, inputs = snapshot_fn(), None
            else:
                wait = data_wait_reason(b, source)
                if wait:                                  # rule v2 schedule.data: no NIFTY200 on a half-filled store
                    status.update({"status": "waiting_data", "reason": wait})
                    raise _Done()
            left = grammar_progress(b, st)
            if left["undrawn"] == 0:                      # every sentence of the grammar was drawn: nothing to freeze
                status.update({"status": "grammar_exhausted", "undrawn": left})
                raise _Done()
            if snapshot_fn is None:
                snap, inputs = snapshot_process(root, source)
            bid = b.freeze(snap, inputs=inputs)["batch"]
        status["batch"] = bid
        s = _settings()
        try:
            if runner is not None:
                summary = runner(root, bid)
            else:
                limit = int(getattr(s, "builder_time_limit_s", 14400))
                summary = run_process(root, bid, source, jobs=int(getattr(s, "builder_jobs", 2)),
                                      time_limit_s=min(limit, int(time_limit_s)) if time_limit_s else limit,
                                      mem_mb=int(getattr(s, "builder_mem_mb", 3584)))
        except Exception as exc:
            n = b.record_attempt(bid, f"{type(exc).__name__}: {exc}")
            err = str(exc)
            unrecoverable = ("NotFrozen" in err or "BundleCorrupt" in err) and live_differs(bid)
            if n >= MAX_ATTEMPTS or unrecoverable:
                why = "inputs unusable and the live data changed" if unrecoverable else f"{n} failed attempts"
                b.fail(bid, f"{why}; last error: {err[:300]}")
                status.update({"status": "failed", "batch": bid, "attempts": n, "error": err[:400]})
                raise _Done()
            raise
        out = b.ingest(bid, summary, os.getenv("BUILDER_TRIALS_LEDGER") or None)
        status.update({"status": "complete", "proven": summary["proven"], "seconds": summary.get("seconds")})
        status["ingest"] = {k: v for k, v in out.items() if k != "result"}
    except _Done:
        pass
    except DiskCapReached as exc:
        status.update({"status": "disk_cap", "error": str(exc)})
    except RuleChanged as exc:
        status.update({"status": "rule_changed", "error": str(exc)})
    except SeedChanged as exc:
        status.update({"status": "seed_changed", "error": str(exc)})
    except Exception as exc:
        status.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]})
    status["finished"] = datetime.now().isoformat(timespec="seconds")
    try:
        _jsave(_status_path(root), status)
    except Exception:
        pass
    return status


def data_wait_reason(b: "Builder", source: str = "server", root=None) -> Optional[str]:
    """Why a NEW batch must not be frozen yet (None = go): under a rule whose grammar has the NIFTY200 universe, the
    builder data layer's minute universe must be complete (every added name Kite lists fully backfilled); under
    builder/v3 (schedule.data_gates) the data layer's per-session NSE archive backfill (2012 ->) must be complete too
    (nse_archive_status)."""
    if source != "server" or b.rule["grammar"]["version"] == "bg1":
        return None
    try:
        import builder_universe as BU
        ms = BU.minute_status(root)
    except Exception as exc:
        return f"minute universe status unavailable: {type(exc).__name__}: {str(exc)[:160]}"
    if ms.get("mode") != "nifty200":
        return ("BUILDER_MINUTE_UNIVERSE is nifty100: the NIFTY200 universe of builder/v2 would be the 97 names again "
                "(new batches wait; a pending batch is still resumed)")
    if ms.get("unresolved"):
        return (f"{len(ms['unresolved'])} added names unresolved: the Kite instrument lookup failed "
                f"({(ms.get('token_lookup') or {}).get('error') or 'no successful dump yet'}); retried next minute run")
    if not ms.get("ready"):
        return (f"minute backfill of the added NIFTY 200 names at {ms.get('progress_pct')} % "
                f"({ms.get('names_ready')}/{ms.get('names_fetchable')} names complete)")
    gates = (b.rule.get("schedule") or {}).get("data_gates") or ["minute_universe"]
    if "nse_archives" in gates:                           # builder/v3: the NSE archive backfill (2012 ->) too
        try:
            ns = nse_archive_status(root=root)
        except Exception as exc:
            return f"NSE archive status unavailable: {type(exc).__name__}: {str(exc)[:160]}"
        if not ns["complete"]:
            return ns["reason"]
    return None


def nse_archive_status(now: Optional[datetime] = None, root=None) -> dict:
    """Is the builder data layer's NSE archive backfill (2012-01-01 ->) complete? Read-only: builder_data's own state
    file and its own finality rule (builder_data.Builder.final / process_day's done-condition), nothing is fetched or
    written.
      days     every calendar day from NSE_START to today - NSE_GATE_LAG_DAYS must be FINAL: its cash archive status is
               final (ok / no session / missing, as the data layer rules it) and, on a session, so is every per-session
               set (index closes, F&O parts, participant OI / volume, ban list, FII stats). A day in an error status
               (retried by the data layer on every run) is NOT final: the gate waits and names it.
      windows  every quarterly / monthly feed window (deals, corporate actions, index membership, filing feeds) that
               ended more than NSE_GATE_LAG_DAYS ago must have been ATTEMPTED after its end; windows whose last answer
               was an error are counted in `windows_error` (reported, not blocking: NSE does not serve some old
               months, and the data layer keeps retrying them)."""
    import builder_data as BD
    from datetime import timedelta
    if now is None:
        import ist_clock
        now = ist_clock.now_ist()
    today = now.date()
    bd = BD.Builder(root)
    out = {"complete": False, "from": BD.ymd(BD.NSE_START), "days_pending": None, "windows_pending": None,
           "windows_error": None, "reason": None}
    if not bd.state_path.exists():
        out["reason"] = "the data layer's NSE archive backfill has not started (no state file)"
        return out
    st = bd.state()
    days = st.get("days") or {}
    last = today - timedelta(days=NSE_GATE_LAG_DAYS)
    out["to"] = BD.ymd(last)
    pending, d = [], BD.NSE_START
    while d <= last:
        k = BD.ymd(d)
        ds = days.get(str(k)) or {}
        cash = ds.get("cash")
        done = bd.final(cash, k, today) and (cash != "ok" or all(bd.final(ds.get(x), k, today) for x in BD.DAY_SETS))
        if not done:
            pending.append(k)
        d += timedelta(days=1)
    wins = st.get("windows") or {}
    expected = [(f"deals:{kind}:{tag}", e) for _s, e, tag in bd.quarters(BD.NSE_START, today)
                for kind in ("bulk", "block")]
    expected += [(f"ca:{tag}", e) for _s, e, tag in bd.quarters(BD.NSE_START, today + timedelta(days=92))]
    expected += [(f"mcwb:{tag}", e) for _s, e, tag in bd.months(BD.NSE_START,
                                                                   today.replace(day=1) - timedelta(days=1))]
    expected += [(f"ev:{feed}:{tag}", e) for feed in BD.FEEDS for _s, e, tag in bd.months(BD.EVENTS_FROM, today)
                 if feed != "ifr" or e >= BD.date(2025, 4, 1)]
    wp, we = [], []
    for key, e in expected:
        if e > last:
            continue
        rec = wins.get(key) or {}
        if rec.get("status") == "ok" or bd._wdone(st, key, e, today):
            continue
        if rec and int(rec.get("at", 0) or 0) > BD.ymd(e) + 3:
            we.append(key)                               # attempted after its end, last answer an error
        else:
            wp.append(key)
    out.update({"days_pending": len(pending), "windows_pending": len(wp), "windows_error": len(we),
                "complete": not pending and not wp})
    if pending:
        k = pending[-1]
        stt = {x: (days.get(str(k)) or {}).get(x) for x in BD.DAY_SETS if (days.get(str(k)) or {}).get(x)}
        out["reason"] = (f"NSE archive backfill (2012 ->) incomplete: {len(pending)} calendar days from {pending[0]} "
                         f"to {pending[-1]} not final yet (newest: {k} {stt or 'not fetched'})")
        if len(pending) <= 12:                           # a few stuck days: name every one with what is not final
            def _bad(kk):
                ds_ = days.get(str(kk)) or {}
                return {x: ds_.get(x) for x in BD.DAY_SETS if not bd.final(ds_.get(x), kk, today)} or "not fetched"
            out["pending_days"] = {str(kk): _bad(kk) for kk in pending}
            out["reason"] += "; all: " + "; ".join(f"{kk} {v}" for kk, v in out["pending_days"].items())
    elif wp:
        out["reason"] = (f"NSE feed backfill incomplete: {len(wp)} windows not fetched yet (e.g. {wp[0]})")
    return out


def night_window_open(now: datetime) -> bool:
    """May a batch START now? Not between NIGHT_STOP (05:15 IST) and the end of the market-hours guard (15:45), and
    never inside market hours: the 02:30 job's last batch starts before 05:15."""
    m = now.hour * 60 + now.minute
    return not (NIGHT_STOP_MIN <= m <= 15 * 60 + 45)


def nightly_job(now: Optional[datetime] = None, root=None, source: str = "server", runner=None, snapshot_fn=None,
                clock=None, max_batches: Optional[int] = None, locked: bool = False) -> dict:
    """builder/v3 schedule: batch after batch in one night. Each batch is ONE weekly_job call (frozen, evaluated and
    ingested in full before the next is frozen; 400 candidates each; N grows batch by batch exactly as with one batch a
    night). Stops when (a) the grammar has no undrawn sentence left, (b) BUILDER_NIGHT_BATCHES batches ran or the night
    window closed (no batch is started after 05:15 IST or inside market hours), or (c) a batch did not complete
    (failed / error / disk cap / waiting for data). `clock` (tests): the time before each batch."""
    if not enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    import ist_clock
    clock = clock or ((lambda: now) if now is not None else ist_clock.now_ist)
    budget = int(max_batches if max_batches is not None else getattr(_settings(), "builder_night_batches", 12) or 0)
    root = Path(root) if root is not None else default_root()
    t = clock()
    night = {"started": t.isoformat(timespec="seconds"), "budget": budget, "batches": [], "stopped": None}
    last: dict = {}
    while True:
        if len(night["batches"]) >= budget:
            night["stopped"] = f"night budget ({budget} batches)"
            break
        if not night_window_open(t):
            night["stopped"] = "night window closed (no batch is started after 05:15 IST or inside market hours)"
            break
        # an evaluation must be over before the 09:00 market-hours guard: its wall-clock limit ends at 08:30
        end = t.replace(hour=8, minute=30, second=0, microsecond=0)
        limit = int((end - t).total_seconds()) if t < end else None
        kw = dict(now=t, root=root, source=source, runner=runner, snapshot_fn=snapshot_fn,
                  time_limit_s=max(600, limit) if limit is not None else None)
        last = weekly_job_locked(**kw) if locked else weekly_job(**kw)
        if last.get("status") != "complete":
            night["stopped"] = last.get("status") or last.get("skipped") or "not complete"
            break
        night["batches"].append(last.get("batch"))
        t = clock()
    night["batches_run"] = len(night["batches"])
    night["finished"] = datetime.now().isoformat(timespec="seconds")
    try:
        b = Builder(root)
        if (root / "registry.jsonl").exists():
            night["undrawn"] = grammar_progress(b)
            st_ = b.state()
            res = st_["results"]
            # what the night found, in one place (the scheduler logs it): per batch the gate counts, and N after
            night["results"] = {bid: {k: res[bid].get(k) for k in ("discovery_passers", "validation_passers",
                                                                     "final_confirmed", "proven", "n_used")}
                                for bid in night["batches"] if bid in res}
            night["proven"] = [c for bid in night["batches"] if bid in res for c in (res[bid].get("proven") or [])]
            night["n_used"] = max(st_["n_used"], b.n_base(st_))           # N as GET /api/v2/builder reports it
    except Exception as exc:
        night["undrawn_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out = {**last, "night": night}
    try:
        if (root / "registry.jsonl").exists() or last.get("status"):
            _jsave(_status_path(root), out)
    except Exception:
        pass
    return out


def nightly_job_locked(**kw) -> dict:
    """nightly_job; every batch holds the builder data layer's lock on its own (weekly_job_locked), so the data
    layer's jobs can run between two batches but never while one freezes or is evaluated."""
    if not enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    return nightly_job(locked=True, **kw)


_GP_CACHE: dict = {}


def grammar_progress(b: "Builder", st: Optional[dict] = None) -> dict:
    """How much of the CURRENT grammar is still undrawn: sentences (x market type) whose key - and whose bg1 twin -
    was never drawn in any batch (the seeds included). {"grammar", "total", "undrawn", "drawn", "per_agent":
    {agent: {"total", "undrawn"}}}. Zero undrawn = every sentence of the grammar was tested (or is frozen)."""
    from stratlab import builder_grammar as G
    st = st or b.state()
    gv = b.rule["grammar"]["version"]
    ck = (str(b.root), gv, len(st["batches"]), len(st.get("seed_keys") or []))
    if ck in _GP_CACHE:
        return _GP_CACHE[ck]
    out = undrawn_counts(gv, b.drawn_keys(st))
    _GP_CACHE.clear()
    _GP_CACHE[ck] = out
    return out


def undrawn_counts(gv: str, drawn) -> dict:
    """grammar_progress for a given set of drawn keys (pure)."""
    from stratlab import builder_grammar as G
    drawn = set(drawn)
    per = {}
    for a in G.AGENTS:
        tot = und = 0
        for m in G.MARKET_TYPES:
            for c in G.cell_grammar(a, m, gv):
                tot += 1
                und += bool(G.never_drawn(c, drawn))
        per[a] = {"total": tot, "undrawn": und}
    out = {"grammar": gv, "total": sum(v["total"] for v in per.values()),
           "undrawn": sum(v["undrawn"] for v in per.values()), "per_agent": per}
    out["drawn"] = out["total"] - out["undrawn"]
    return out


def weekly_job_locked(**kw) -> dict:
    """weekly_job holding the builder data layer's lock (its NSE / Kite jobs never write while a batch freezes or is
    evaluated). Off -> returns at once without waiting for the lock."""
    if not enabled():
        return {"skipped": "BUILDER_ENABLED is off or not PAPER"}
    try:
        import builder_data
        lock = builder_data._LOCK
    except Exception:
        lock = threading.Lock()
    got = lock.acquire(timeout=7200)
    try:
        return weekly_job(**kw)
    finally:
        if got:
            lock.release()


def data_snapshot(source: str = "server", data_root: Optional[str] = None) -> dict:
    """The data snapshot ids (in THIS process: building the panels takes a minute and ~1 GB)."""
    from stratlab.builder_source import make_source
    return make_source(source, data_root).snapshot()


def snapshot_mem_mb() -> int:
    """Address-space limit of the snapshot process (BUILDER_SNAPSHOT_MEM_MB): it builds the full daily panel from
    the whole cash table, so it gets more room than one evaluation process."""
    return int(getattr(_settings(), "builder_snapshot_mem_mb", 6144))


def snapshot_process(root: Path, source: str = "server", data_root: Optional[str] = None, time_limit_s: int = 3600,
                     mem_mb: Optional[int] = None, bundle: bool = True) -> tuple:
    """data_snapshot in a separate process (python -m builder_run --snapshot): the API process never builds panels.
    With bundle=True the process also writes the batch's IMMUTABLE inputs (stratlab/builder_bundle.py) under
    <root>/inputs/ (server sources; dropped again when the builder directory would pass BUILDER_MAX_MB).
    Returns (snapshot, inputs | None)."""
    out = Path(root) / "snapshot.json"
    if out.exists():
        out.unlink()
    cmd = [sys.executable, "-m", "builder_run", "--snapshot", "--root", str(root), "--source", source,
           "--mem-mb", str(int(mem_mb if mem_mb is not None else snapshot_mem_mb()))] + \
        (["--data-root", str(data_root)] if data_root else []) + \
        (["--bundle"] if bundle else [])
    env = dict(os.environ, BUILDER_CACHE_DIR=str(cache_dir()), **PROCESS_ENV)
    p = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True, timeout=int(time_limit_s),
                       start_new_session=True)
    if p.returncode != 0 or not out.exists():
        raise RunFailed(f"snapshot process exit {p.returncode}: {(p.stderr or '')[-1200:]}")
    doc = json.loads(out.read_text())
    inputs = doc.get("inputs")
    if inputs:
        d = Path(root) / "inputs" / inputs["dir"]
        if dir_bytes(root) >= max_bytes():
            shutil.rmtree(d, ignore_errors=True)
            inputs = None
            doc["inputs_error"] = "frozen inputs dropped: BUILDER_MAX_MB reached"
    return doc["snapshot"], inputs


def write_inputs(root, src) -> Optional[dict]:
    """In the snapshot process: the frozen inputs of the next batch (server-layout sources only)."""
    from stratlab import builder_bundle as BB
    from stratlab.builder_source import snapshot_sha
    if not getattr(src, "minute_root", None):
        return None
    snap = src.snapshot()
    name = snapshot_sha(snap)[:16] + "-" + datetime.now().strftime("%Y%m%d%H%M%S")
    d = Path(root) / "inputs" / name
    d.parent.mkdir(parents=True, exist_ok=True)
    man = BB.write_bundle(src, str(d))
    return {"dir": name, "manifest_sha256": BB.manifest_sha(str(d)), "snapshot_sha256": snapshot_sha(man["snapshot"]),
            "bytes": dir_bytes(d)}


# ══════════════════════════════════════════════════════════════════════════
# API views
# ══════════════════════════════════════════════════════════════════════════
def _result(b: Builder, bid: str) -> Optional[dict]:
    p = b.result_path(bid)
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


_CU_CACHE: dict = {}


def cell_universes(b: Builder, bid: str, fb: dict, st: dict) -> dict:
    """{cell: {universe: candidates}} of a batch (bg1 batches and the seed: their implicit universes)."""
    if fb.get("cell_universes"):
        return fb["cell_universes"]
    from stratlab import builder_grammar as G
    ck = (str(b.root), bid, fb.get("candidates_sha256"))
    if ck in _CU_CACHE:
        return _CU_CACHE[ck]
    out: dict = {}
    if fb.get("seeded"):
        for k in ((st.get("seeds") or {}).get(bid) or {}).get("keys") or []:
            try:
                v = json.loads(k)
                cell, u = f"{v[1]}|{v[2]}", (v[10] if len(v) > 10 else G.implied_universe(v[1], v[3], v[7]))
            except Exception:
                continue
            out.setdefault(cell, {}).setdefault(u, 0)
            out[cell][u] += 1
    else:
        for r in (_jload(b.candidates_path(bid), {}) or {}).get("candidates", []):
            c = G.Candidate.from_json(r["cand"])
            out.setdefault(c.cell, {}).setdefault(c.stock_universe, 0)
            out[c.cell][c.stock_universe] += 1
    _CU_CACHE[ck] = out
    return out


def api_status(root=None) -> dict:
    """GET /api/v2/builder (API_V2.md)."""
    from stratlab import builder_grammar as G
    root = Path(root) if root is not None else default_root()
    try:
        rule = load_rule()
        rule_err = None
    except Exception as exc:
        rule, rule_err = (json.loads(CURRENT_RULE_FILE.read_text()) if CURRENT_RULE_FILE.exists() else {}), str(exc)
        rule["_sha256"] = file_sha256(CURRENT_RULE_FILE) if CURRENT_RULE_FILE.exists() else None
    gv = (rule.get("grammar") or {}).get("version") or G.GRAMMAR_VERSION
    out = {"enabled": enabled(), "rule_version": rule.get("version"), "rule_sha256": rule.get("_sha256"),
           "rule_error": rule_err, "agents": list(G.AGENTS), "market_types": list(G.MARKET_TYPES),
           "grammar_version": gv, "grammar_sizes": G.grammar_sizes(gv), "batches": [], "grid": [], "n_used": None,
           "lab_trials_count": (rule.get("dsr") or {}).get("lab_trials_count"),
           "lab_retest_trials": int((rule.get("dsr") or {}).get("lab_retest_trials", 0) or 0),
           "evaluated": 0, "final_looks": 0,
           "promoted": 0, "last_run": _jload(_status_path(root), {}), "chain_ok": True,
           "rules": [{"version": v, "sha256": s, "file": Path(f).name} for v, f, s in RULES],
           "size": {"text": (rule.get("costs") or {}).get("size"), "capital": (rule.get("costs") or {}).get("capital"),
                    **rule_sizes(rule)},
           "schedule": (rule.get("schedule") or {}).get("when"),
           "universes": {"choices": list(G.UNIVERSES), "text": dict(G.UNIVERSE_TEXT)}}
    if not (root / "registry.jsonl").exists():
        out["grid"] = [{"agent": a, "mtype": m, "proven": 0, "in_forward_lab": 0, "qualified": 0, "ready": 0,
                        "universes": {}} for a in G.AGENTS for m in G.MARKET_TYPES]
        try:
            sds = [sb for _p, doc in load_seeds() for sb in seed_batches(doc)]
            out["n_used"] = (max(int(sb["n_used_after"]) for sb in sds)         # imported on first use
                             + int((rule.get("dsr") or {}).get("lab_retest_trials", 0) or 0))
            out["seed"] = {"batch": sds[0]["batch"], "frozen_hash": sds[0]["frozen_hash"], "imported": False,
                           "candidates": sds[0]["n_candidates"]}
            out["seeds"] = [{"batch": sb["batch"], "frozen_hash": sb["frozen_hash"], "imported": False,
                             "candidates": sb["n_candidates"]} for sb in sds]
            ck = ("no-registry", gv, len(sds))
            if ck not in _GP_CACHE:                       # (the seeds' keys count as drawn before the first run too)
                _GP_CACHE[ck] = undrawn_counts(gv, [c["key"] for sb in sds for c in sb["candidates"]])
            out["undrawn"] = _GP_CACHE[ck]
        except Exception as exc:
            out["n_used"] = out["lab_trials_count"]
            out["seed_error"] = str(exc)[:200]
        out["night_batches"] = int(getattr(_settings(), "builder_night_batches", 12) or 0)
        return out
    b = Builder(root, rule)
    st = b.state()
    out["chain_ok"] = bool(b.registry.verify()["ok"])
    out["evaluated"], out["final_looks"] = st["evaluated"], st["final_looks"]
    if st.get("seed"):
        out["seed"] = {"batch": st["seed"]["batch"], "frozen_hash": st["seed"]["frozen_hash"], "imported": True,
                       "candidates": st["seed"]["n_candidates"], "file_sha256": st["seed"]["file_sha256"]}
        out["seeds"] = [{"batch": sd["batch"], "frozen_hash": sd["frozen_hash"], "imported": True,
                         "candidates": sd["n_candidates"], "file": sd.get("file"), "file_sha256": sd["file_sha256"]}
                        for sd in (st.get("seeds") or {}).values()]
    out["n_used"] = max(st["n_used"], b.n_base(st))
    try:
        out["undrawn"] = grammar_progress(b, st)
    except Exception as exc:
        out["undrawn"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    out["night_batches"] = int(getattr(_settings(), "builder_night_batches", 12) or 0)
    out["disk_mb"] = round(dir_bytes(root) / 1048576, 2)
    out["rule_chain"] = [{"sha256": s, "version": (RULES[rule_index(s)][0] if rule_index(s) >= 0 else None)}
                         for s in st["rules"]]
    out["upgrades"] = st["upgrades"]
    cu_all = {}
    last_universes = None
    for bid in sorted(st["batches"], reverse=True):
        fb = st["batches"][bid]
        r = st["results"].get(bid)
        cu = cell_universes(b, bid, fb, st)
        for cell, d in cu.items():
            for u, n in d.items():
                cu_all.setdefault(cell, {}).setdefault(u, 0)
                cu_all[cell][u] += n
        row = {"id": bid, "created": fb["created"], "hash": fb["frozen_hash"],
               "candidates_sha256": fb["candidates_sha256"], "snapshot_end": fb.get("snapshot_end"),
               "status": "seeded" if fb.get("seeded") else ("failed" if (r or {}).get("failed") else
               ("evaluated" if r else ("final_open" if bid in st["final_open"] else "frozen"))),
               "candidates": fb["n_candidates"],
               "rule_version": fb.get("rule_version") or (RULES[rule_index(fb.get("rule_sha256", ""))][0]
                                                          if rule_index(fb.get("rule_sha256", "")) >= 0 else
                                                          "builder/v1"),
               "grammar_version": (fb.get("generator") or {}).get("grammar_version"),
               "size": sizes_by_sha(fb.get("rule_sha256")),
               "universes": {u: sum(d.get(u, 0) for d in cu.values()) for u in G.UNIVERSES
                             if any(d.get(u) for d in cu.values())}}
        if r and last_universes is None and not r.get("failed") and not fb.get("seeded"):
            res = _result(b, bid)
            if res and res.get("universes"):
                last_universes = {"batch": bid, **{k: v for k, v in res["universes"].items()
                                                   if k in ("split", "meta", "counts")},
                                  **{u: {k: v for k, v in (res["universes"].get(u) or {}).items() if k != "names"}
                                     for u in ("NIFTY100", "NIFTY200", "TOP500")}}
        if r:
            row.update({k: r.get(k) for k in ("discovery_passers", "shuffle_passers", "validation_tested",
                                              "validation_passers", "final_confirmed", "n_used", "best_discovery_dsr")})
            if r.get("failed"):
                row["failure"] = {"reason": r.get("reason"), "attempts": r.get("attempts")}
            row["proven"] = len(r["proven"])
            row["errata"] = len((st.get("errata") or {}).get(bid, []))
            row["shuffle_expected"] = r["shuffle_passers"]
        out["batches"].append(row)
    fw = {}
    try:
        import builder_forward
        fw = builder_forward.BuilderForward(root / "forward").status_by_id()
    except Exception:
        fw = {}
    grid = {(a, m): {"agent": a, "mtype": m, "proven": 0, "in_forward_lab": 0, "qualified": 0, "ready": 0,
                     "universes": {u: {"candidates": n, "proven": 0} for u, n in
                                   sorted(cu_all.get(f"{a}|{m}", {}).items())}}
            for a in G.AGENTS for m in G.MARKET_TYPES}
    for p in st["promoted"]:
        g = grid[(p["cand"]["agent"], p["cand"]["mtype"])]
        g["proven"] += 1
        u = p.get("universe") or G.Candidate.from_json(p["cand"]).stock_universe
        g["universes"].setdefault(u, {"candidates": 0, "proven": 0})["proven"] += 1
        f = fw.get(p["id"])
        if f:
            g["in_forward_lab"] += 1
            g["qualified"] += int(f.get("status") == "qualified")
            g["ready"] += int(bool(f.get("ready")))
    out["grid"] = list(grid.values())
    if last_universes:
        out["universes"]["last_batch"] = last_universes
    out["promoted"] = len(st["promoted"])
    out["promoted_unregistered"] = sum(1 for p in st["promoted"] if not p.get("forward_registered"))
    return out


def api_batch(bid: str, root=None) -> Optional[dict]:
    """GET /api/v2/builder/batches/{id}."""
    root = Path(root) if root is not None else default_root()
    if not (root / "registry.jsonl").exists():
        return None
    from stratlab import builder_grammar as G
    try:
        rule = load_rule()
    except Exception:
        rule = json.loads(CURRENT_RULE_FILE.read_text())
        rule["_sha256"] = None
    b = Builder(root, rule)
    st = b.state()
    fb = st["batches"].get(str(bid))
    if fb is None:
        return None
    doc = _jload(b.candidates_path(bid), {}) or {}
    res = _result(b, bid)
    by = {r["id"]: r for r in (res or {}).get("rows", [])}
    rows = []
    for c in doc.get("candidates", []):
        r = by.get(c["id"], {})
        d = r.get("discovery") or {}
        v = (r.get("validation") or {}).get("screen") or {}
        f = (r.get("final") or {}).get("final") or {}
        pr = r.get("proof") or {}
        cobj = G.Candidate.from_json(c["cand"])
        rows.append({"id": c["id"], "agent": c["cand"]["agent"], "mtype": c["cand"]["mtype"],
                     "block": c["cand"]["block"], "universe": cobj.stock_universe,
                     "universe_explicit": bool(cobj.universe), "size": cobj.size, "text": c["text"],
                     "stage": r.get("stage", "frozen"),
                     "discovery": {k: d.get(k) for k in ("trades", "pf", "net", "h1", "h2", "dsr", "passed",
                                                         "shuffle_passed", "why")} if d else None,
                     "validation": {"passed": (r.get("validation") or {}).get("passed"),
                                    "trades": (v.get("disc") or {}).get("trades"), "pf": (v.get("disc") or {}).get("pf"),
                                    "net": (v.get("disc") or {}).get("net"), "reasons": v.get("reasons")}
                     if r.get("validation") else None,
                     "final": {"confirmed": (r.get("final") or {}).get("confirmed"), "trades": f.get("trades"),
                               "pf": f.get("pf"), "net": f.get("net")} if r.get("final") else None,
                     "proof": {"dsr": (pr.get("dsr") or {}).get("dsr"), "gates": pr.get("gates"),
                               "proven": pr.get("proven")} if pr else None})
    out = {"id": bid, "created": fb["created"], "hash": fb["frozen_hash"],
           "candidates_sha256": fb["candidates_sha256"], "snapshot_sha256": fb["snapshot_sha256"],
           "snapshot_end": fb.get("snapshot_end"), "generator": fb["generator"], "n_before": fb.get("n_before"),
           "status": "failed" if (st["results"].get(str(bid)) or {}).get("failed") else
           ("evaluated" if res else "frozen"), "candidates": rows,
           "errata": (st.get("errata") or {}).get(str(bid), []), "seeded": bool(fb.get("seeded")),
           "rule_sha256": fb.get("rule_sha256"), "size": sizes_by_sha(fb.get("rule_sha256")),
           "rule_version": fb.get("rule_version") or (RULES[rule_index(fb.get("rule_sha256", ""))][0]
                                                      if rule_index(fb.get("rule_sha256", "")) >= 0 else "builder/v1"),
           "cell_universes": cell_universes(b, str(bid), fb, st)}
    if res and res.get("universes"):
        u = res["universes"]
        out["universes"] = {k: v for k, v in u.items() if k in ("split", "complete_share", "meta", "counts")}
        for k in ("NIFTY100", "NIFTY200", "TOP500"):
            if k in u:
                out["universes"][k] = dict(u[k])
    if fb.get("seeded"):
        sd = (st.get("seeds") or {}).get(str(bid)) or {}
        out.update({"status": "seeded", "seed": {k: sd.get(k) for k in ("file", "file_sha256", "counts", "final_looks",
                                                                         "n_used_after", "keys_sha256", "proven")}})
    if res:
        out.update({k: res.get(k) for k in ("windows", "discovery_passers", "shuffle_passers", "validation_tested",
                                            "validation_passers", "final_confirmed", "proven", "n_disc", "n_proof",
                                            "v", "best_discovery_dsr", "cells", "seconds", "peak_rss_mb")})
        out["planted"] = {k: v for k, v in (res.get("planted") or {}).items()}
    return out
