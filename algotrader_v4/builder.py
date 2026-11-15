"""
builder.py — the AUTOMATIC STRATEGY BUILDER (builder/v1): every week it BUILDS 400 candidate strategies from our own
data (5 agents x 4 market types x 20, stratlab/builder_grammar.py), freezes them in a hash-chained registry BEFORE any
return is computed, evaluates them in a separate process under the frozen rule builder_rule.json (sha256 pinned below)
and promotes only what is proven to the forward lab's separate builder/v1 family (builder_forward.py). PAPER research
only: no order entry point, agent, risk or trading state is imported, read or changed.

Weekly job (platform_scheduler, Saturday 06:00 IST, only with BUILDER_ENABLED=true and PAPER; the job re-checks both):
  1. freeze    draw the batch (deterministic in the batch id, never re-drawing a key of an earlier batch), write
               <root>/batches/<id>/candidates.json, append "batch_frozen" (candidate sha256, generator version, grammar
               sha256, data snapshot ids + sha256, N before) to <root>/registry.jsonl
  2. evaluate  python -m builder_run in its own process group, nice 10, <= 2 workers, address-space limit, wall-clock
               limit; it refuses a batch whose candidate file / snapshot does not match the registry; resumable
  3. ingest    "final_open" was appended by the run before any FINAL return; "batch_result" (summary, N_used) is
               appended after a successful run; candidates -> trials ledger (if configured); proven -> promotion
Disk: BUILDER_MAX_MB (default 400) for <root>; at the cap nothing is written (status disk_cap); nothing is deleted.
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
# sha256 of builder_rule.json as committed (the pre-registration). An edited rule file is refused.
RULE_SHA256 = "a455009021b9e2cfbc55106c8ee1310ac6a7135995a895b6b8cafda06fbc3c06"
DEFAULT_MAX_MB = 400.0
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


class DiskCapReached(RuntimeError):
    pass


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rule(path=RULE_FILE, expect: Optional[str] = None) -> dict:
    sha = file_sha256(path)
    want = RULE_SHA256 if expect is None else expect
    if want != "PENDING" and sha != want:
        raise RuleChanged(f"builder_rule.json sha256 {sha[:12]} != pinned {want[:12]}: the rule is frozen")
    rule = json.loads(Path(path).read_text())
    rule["_sha256"] = sha
    return rule


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
    tot = 0
    for dp, _dn, fn in os.walk(str(root)):
        for f in fn:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
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
              "evaluated": 0, "final_looks": 0}
        for rec in self.registry.records():
            k, d = rec["kind"], rec["data"]
            if k == "genesis":
                st["genesis"] = d
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
                st["promoted"].append({**d, "ts": rec["ts"], "hash": rec["hash"]})
            elif k == "erratum":
                st.setdefault("errata", {}).setdefault(d["batch"], []).append({**d, "ts": rec["ts"]})
            elif k == "seed":
                bid = d["batch"]
                st["seed"] = {**d, "ts": rec["ts"], "hash": rec["hash"]}
                st["batches"][bid] = {"batch": bid, "seeded": True, "candidates_sha256": d["candidates_sha256"],
                                      "n_candidates": d["n_candidates"], "snapshot_end": d["snapshot_end"],
                                      "snapshot_sha256": d["snapshot_sha256"], "source": "sandbox seed",
                                      "generator": {"grammar_version": d["grammar_version"],
                                                    "grammar_sha256": d["grammar_sha256"]},
                                      "n_before": d["lab_trials_count"] + d["development_evaluations"],
                                      "frozen_seq": rec["seq"], "frozen_hash": d["frozen_hash"], "created": rec["ts"]}
                st["final_open"][bid] = [f["id"] for f in d["final_looks"]]
                st["results"][bid] = {**{k2: v for k2, v in d["counts"].items() if k2 != "stages"}, "batch": bid,
                                      "proven": [], "n_used": d["n_used_after"], "best_discovery_dsr": None,
                                      "ts": rec["ts"], "seeded": True}
                st["evaluated"] += int(d["n_candidates"])
                st["n_used"] = max(st["n_used"], int(d["n_used_after"]))
                st["seed_keys"] = list(d["keys"])
                for e in d.get("errata") or []:
                    st.setdefault("errata", {}).setdefault(bid, []).append({**e, "batch": bid})
        st["final_looks"] = sum(len(v) for v in st["final_open"].values())
        return st

    def ensure_genesis(self) -> dict:
        st = self.state()
        if st["genesis"] is not None:
            if st["genesis"]["rule_sha256"] != self.rule["_sha256"]:
                raise RuleChanged("the builder registry was started under a different rule file")
            self.ensure_seed()
            return st["genesis"]
        from stratlab import builder_grammar as G
        g = {"rule_version": self.rule["version"], "rule_sha256": self.rule["_sha256"],
             "grammar_version": G.GRAMMAR_VERSION, "grammar_sha256": G.grammar_sha256(),
             "lab_trials_count": int(self.rule["dsr"]["lab_trials_count"])}
        self.registry.append("genesis", g)
        self.ensure_seed()
        return g

    def ensure_seed(self, path=None) -> Optional[dict]:
        """Import the sandbox seed once (hash-chained, idempotent; refused if the file is not the pinned one). Skipped
        in the registry the seed was taken from (its batch 0001 carries the seed's frozen hash)."""
        st = self.state()
        if st.get("seed") is not None:
            return st["seed"]
        seed = load_seed(path)
        fb = st["batches"].get(seed["batch"])
        if fb is not None and fb.get("frozen_hash") == seed["frozen_hash"]:
            return None                                   # the source registry itself
        bid = seed["batch"] if seed["batch"] not in st["batches"] else "sandbox-" + seed["batch"]
        keys = [c["key"] for c in seed["candidates"]]
        if len(keys) != int(seed["n_candidates"]) or len(set(keys)) != len(keys):
            raise SeedChanged("seed candidate keys are not unique / complete")
        rec = {"batch": bid, "file": Path(path or SEED_FILE).name, "file_sha256": seed["_sha256"],
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
        self.registry.append("seed", rec)
        return self.state()["seed"]

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
        base = int(self.rule["dsr"]["lab_trials_count"])
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
    def freeze(self, snapshot: dict, batch_id: Optional[str] = None, per_cell: Optional[int] = None) -> dict:
        from stratlab import builder_grammar as G
        from stratlab.builder_source import snapshot_sha
        with _LOCK:
            self.ensure_genesis()
            st = self.state()
            bid = batch_id or self.next_batch_id(st)
            if bid in st["batches"]:
                return st["batches"][bid]
            self.check_cap()
            cands = G.draw_batch(bid, self.drawn_keys(st), per_cell or int(self.rule["batch"]["per_cell"]))
            rows = []
            per: dict = {}
            for c in cands:
                k = per.get(c.cell, 0)
                per[c.cell] = k + 1
                rows.append({"id": G.candidate_id(bid, c, k), "key": c.key, "cand": c.to_json(), "text": G.describe(c)})
            doc = {"batch": bid, "generator": {"grammar_version": G.GRAMMAR_VERSION,
                                               "grammar_sha256": G.grammar_sha256()},
                   "rule_sha256": self.rule["_sha256"], "snapshot": snapshot, "candidates": rows}
            p = self.candidates_path(bid)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(canon(doc))
            rec = {"batch": bid, "candidates_sha256": file_sha256(p), "n_candidates": len(rows),
                   "cells": per, "generator": doc["generator"], "rule_sha256": self.rule["_sha256"],
                   "snapshot_sha256": snapshot_sha(snapshot), "snapshot_end": snapshot.get("end"),
                   "source": snapshot.get("source"), "n_before": self.n_base(st)}
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
               "reference": ref}
        self.registry.append("promoted", rec)
        write_catalog(self)
        try:
            import builder_forward
            builder_forward.BuilderForward(self.root / "forward").register(
                [{"id": cid, "batch": bid, "cand": row["cand"], "reference": ref,
                  "fingerprint": sha256_obj(row["cand"])[:16]}], first_day=int(summary["windows"]["T"]))
        except Exception as exc:                     # pragma: no cover - registration is retried by the weekly job
            rec["forward_error"] = str(exc)[:200]
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
    specs = [{"id": p["id"], "batch": p["batch"], "key": p["key"], "cand": p["cand"], "text": p["text"],
              "frozen_hash": p["frozen_hash"], "registry_hash": p["hash"]} for p in st["promoted"]]
    body = ('"""GENERATED by builder.write_catalog — do not edit. Promoted builder/v1 strategies (catalogue-compatible\n'
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
    builder-v1). Backed up first."""
    from stratlab import trials as TR
    led = TR.TrialsLedger(path)
    if os.path.exists(path):
        shutil.copyfile(path, path + f".bak_builder_{bid}_" + time.strftime("%Y%m%d_%H%M%S"))
    W = summary["windows"]
    rows = []
    keep = ("trades", "net", "pf", "mean_net", "std_net", "sr_trade", "mean_gross", "mean_cost", "n_days",
            "sr_daily", "skew_daily", "kurt_daily")
    for r in summary["rows"]:
        base = {"strategy_id": r["id"], "family": "builder", "group": f"{r['agent']}|{r['mtype']}",
                "params_fp": "k:" + hashlib.sha1(r["key"].encode()).hexdigest()[:12], "era": "builder-v1",
                "dataset": f"builder_{bid}", "cost_model": "builder/v1"}
        d = r["discovery"]
        rows.append({**base, "protocol": "builder v1 discovery", "stage": "builder_discovery",
                     "period": f"0-{W['discovery'][1]}", "trades": d["trades"], "net": d["net"], "pf": d["pf"],
                     "sr_daily": d["sr_daily"], "verdict": "PASSED GATES" if d["passed"] else "FAILED GATES",
                     "source": f"builder/{bid}#{r['id']}", "note": "; ".join(d["why"])})
        if r.get("validation"):
            s = r["validation"]["series"]
            rows.append({**base, **{k: s.get(k) for k in keep}, "protocol": "builder v1 validation (protocol.screen)",
                         "stage": "builder_validation", "period": "%d-%d" % tuple(W["validation"]),
                         "verdict": "PASSED SCREEN" if r["validation"]["passed"] else "FAILED SCREEN",
                         "source": f"builder/{bid}#{r['id']}",
                         "note": "; ".join(r["validation"]["screen"]["reasons"])})
        if r.get("final"):
            s = r["final"]["series"]
            rows.append({**base, **{k: s.get(k) for k in keep}, "protocol": "builder v1 final",
                         "stage": "builder_final", "period": "%d-%d" % tuple(W["final"]),
                         "verdict": "CONFIRMED" if r["final"]["confirmed"] else "REJECTED ON FINAL",
                         "source": f"builder/{bid}#{r['id']}", "note": ""})
    return led.append(rows)


# ══════════════════════════════════════════════════════════════════════════
# 2. the evaluation process
# ══════════════════════════════════════════════════════════════════════════
class RunFailed(RuntimeError):
    pass


def run_process(root: Path, bid: str, source: str = "server", jobs: int = 1, time_limit_s: int = 14400,
                mem_mb: int = 3584, data_root: Optional[str] = None) -> dict:
    """python -m builder_run in its own process group (never inside the API process). Raises RunFailed on a timeout,
    a crash, the memory limit or an error. Returns the run's summary (also kept in the work directory)."""
    work = Path(root) / "batches" / bid / "work"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "builder_run", "--root", str(root), "--batch", bid, "--source", source,
           "--jobs", str(max(1, min(2, int(jobs)))), "--mem-mb", str(int(mem_mb))]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               BUILDER_CACHE_DIR=str(cache_dir()))
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
# the weekly job
# ══════════════════════════════════════════════════════════════════════════
def _status_path(root: Path) -> Path:
    return Path(root) / "status.json"


def weekly_job(now: Optional[datetime] = None, root=None, source: str = "server", runner=None,
               snapshot_fn=None) -> dict:
    """Scheduler entry point. Off (flag false or LIVE) -> returns at once: nothing read or written."""
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
        st = b.state()
        pending = [bid for bid in st["batches"] if bid not in st["results"]]
        if pending:
            bid = sorted(pending)[0]                      # resume an unfinished batch before a new one
        else:
            snap = snapshot_fn() if snapshot_fn else snapshot_process(root, source)
            bid = b.freeze(snap)["batch"]
        status["batch"] = bid
        s = _settings()
        if runner is not None:
            summary = runner(root, bid)
        else:
            summary = run_process(root, bid, source, jobs=int(getattr(s, "builder_jobs", 2)),
                                  time_limit_s=int(getattr(s, "builder_time_limit_s", 14400)),
                                  mem_mb=int(getattr(s, "builder_mem_mb", 3584)))
        out = b.ingest(bid, summary, os.getenv("BUILDER_TRIALS_LEDGER") or None)
        status.update({"status": "complete", "proven": summary["proven"], "seconds": summary.get("seconds")})
        status["ingest"] = {k: v for k, v in out.items() if k != "result"}
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


def snapshot_process(root: Path, source: str = "server", data_root: Optional[str] = None, time_limit_s: int = 3600,
                     mem_mb: int = 4096) -> dict:
    """data_snapshot in a separate process (python -m builder_run --snapshot): the API process never builds panels."""
    out = Path(root) / "snapshot.json"
    if out.exists():
        out.unlink()
    cmd = [sys.executable, "-m", "builder_run", "--snapshot", "--root", str(root), "--source", source,
           "--mem-mb", str(int(mem_mb))] + (["--data-root", str(data_root)] if data_root else [])
    env = dict(os.environ, BUILDER_CACHE_DIR=str(cache_dir()))
    p = subprocess.run(cmd, cwd=str(HERE), env=env, capture_output=True, text=True, timeout=int(time_limit_s),
                       start_new_session=True)
    if p.returncode != 0 or not out.exists():
        raise RunFailed(f"snapshot process exit {p.returncode}: {(p.stderr or '')[-400:]}")
    return json.loads(out.read_text())


# ══════════════════════════════════════════════════════════════════════════
# API views
# ══════════════════════════════════════════════════════════════════════════
def _result(b: Builder, bid: str) -> Optional[dict]:
    p = b.result_path(bid)
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def api_status(root=None) -> dict:
    """GET /api/v2/builder (API_V2.md)."""
    from stratlab import builder_grammar as G
    root = Path(root) if root is not None else default_root()
    try:
        rule = load_rule()
        rule_err = None
    except Exception as exc:
        rule, rule_err = (json.loads(RULE_FILE.read_text()) if RULE_FILE.exists() else {}), str(exc)
        rule["_sha256"] = file_sha256(RULE_FILE) if RULE_FILE.exists() else None
    out = {"enabled": enabled(), "rule_version": rule.get("version"), "rule_sha256": rule.get("_sha256"),
           "rule_error": rule_err, "agents": list(G.AGENTS), "market_types": list(G.MARKET_TYPES),
           "grammar_sizes": G.grammar_sizes(), "batches": [], "grid": [], "n_used": None,
           "lab_trials_count": (rule.get("dsr") or {}).get("lab_trials_count"), "evaluated": 0, "final_looks": 0,
           "promoted": 0, "last_run": _jload(_status_path(root), {}), "chain_ok": True}
    if not (root / "registry.jsonl").exists():
        out["grid"] = [{"agent": a, "mtype": m, "proven": 0, "in_forward_lab": 0, "qualified": 0, "ready": 0}
                       for a in G.AGENTS for m in G.MARKET_TYPES]
        try:
            sd = load_seed()
            out["n_used"] = int(sd["n_used_after"])          # imported on first use
            out["seed"] = {"batch": sd["batch"], "frozen_hash": sd["frozen_hash"], "imported": False,
                           "candidates": sd["n_candidates"]}
        except Exception as exc:
            out["n_used"] = out["lab_trials_count"]
            out["seed_error"] = str(exc)[:200]
        return out
    b = Builder(root, rule)
    st = b.state()
    out["chain_ok"] = bool(b.registry.verify()["ok"])
    out["evaluated"], out["final_looks"] = st["evaluated"], st["final_looks"]
    if st.get("seed"):
        out["seed"] = {"batch": st["seed"]["batch"], "frozen_hash": st["seed"]["frozen_hash"], "imported": True,
                       "candidates": st["seed"]["n_candidates"], "file_sha256": st["seed"]["file_sha256"]}
    out["n_used"] = max(st["n_used"], b.n_base(st))
    out["disk_mb"] = round(dir_bytes(root) / 1048576, 2)
    for bid in sorted(st["batches"], reverse=True):
        fb = st["batches"][bid]
        r = st["results"].get(bid)
        row = {"id": bid, "created": fb["created"], "hash": fb["frozen_hash"],
               "candidates_sha256": fb["candidates_sha256"], "snapshot_end": fb.get("snapshot_end"),
               "status": "seeded" if fb.get("seeded") else
               ("evaluated" if r else ("final_open" if bid in st["final_open"] else "frozen")),
               "candidates": fb["n_candidates"]}
        if r:
            row.update({k: r[k] for k in ("discovery_passers", "shuffle_passers", "validation_tested",
                                          "validation_passers", "final_confirmed", "n_used", "best_discovery_dsr")})
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
    grid = {(a, m): {"agent": a, "mtype": m, "proven": 0, "in_forward_lab": 0, "qualified": 0, "ready": 0}
            for a in G.AGENTS for m in G.MARKET_TYPES}
    for p in st["promoted"]:
        g = grid[(p["cand"]["agent"], p["cand"]["mtype"])]
        g["proven"] += 1
        f = fw.get(p["id"])
        if f:
            g["in_forward_lab"] += 1
            g["qualified"] += int(f.get("status") == "qualified")
            g["ready"] += int(bool(f.get("ready")))
    out["grid"] = list(grid.values())
    out["promoted"] = len(st["promoted"])
    return out


def api_batch(bid: str, root=None) -> Optional[dict]:
    """GET /api/v2/builder/batches/{id}."""
    root = Path(root) if root is not None else default_root()
    if not (root / "registry.jsonl").exists():
        return None
    try:
        rule = load_rule()
    except Exception:
        rule = json.loads(RULE_FILE.read_text())
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
        rows.append({"id": c["id"], "agent": c["cand"]["agent"], "mtype": c["cand"]["mtype"],
                     "block": c["cand"]["block"], "text": c["text"], "stage": r.get("stage", "frozen"),
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
           "status": "evaluated" if res else "frozen", "candidates": rows,
           "errata": (st.get("errata") or {}).get(str(bid), []), "seeded": bool(fb.get("seeded"))}
    if fb.get("seeded"):
        sd = st.get("seed") or {}
        out.update({"status": "seeded", "seed": {k: sd.get(k) for k in ("file", "file_sha256", "counts", "final_looks",
                                                                         "n_used_after", "keys_sha256")}})
    if res:
        out.update({k: res.get(k) for k in ("windows", "discovery_passers", "shuffle_passers", "validation_tested",
                                            "validation_passers", "final_confirmed", "proven", "n_disc", "n_proof",
                                            "v", "best_discovery_dsr", "cells", "seconds", "peak_rss_mb")})
        out["planted"] = {k: v for k, v in (res.get("planted") or {}).items()}
    return out
