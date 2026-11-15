"""
builder_run.py — the builder's evaluation PROCESS (started by builder.run_process; never imported by the API).

    python -m builder_run --root <builder dir> --batch 0001 --source server|lab [--data-root <builder data dir>]
                          [--jobs 2] [--mem-mb 3584]

Refuses to evaluate a batch that is not frozen in the registry, whose candidate file changed, or whose data snapshot
differs from the frozen one (builder.Builder.verify_frozen). nice 10, an address-space limit (RLIMIT_AS) per process,
<= 2 forked workers (the DISCOVERY units are split by agent). Every finished unit is kept in <batch>/work/ and reused
on a re-run (resumable). Writes <batch>/work/summary.json.gz; on an error <batch>/work/error.txt and exit 1.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _limit(mem_mb: int) -> None:
    try:
        os.nice(10)
    except Exception:
        pass
    if mem_mb and mem_mb > 0:
        b = int(mem_mb) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception:
            pass


def _peak_mb() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return round(max(a, b), 1)


_W: dict = {}


def _disc_worker(agent: str) -> str:
    _W["batch"].discovery_agent(agent)
    return agent


def evaluate(root: str, bid: str, source: str = "lab", data_root: str | None = None, jobs: int = 1,
             log=print) -> dict:
    import builder
    from stratlab import builder_engine as BE
    from stratlab import builder_eval as EV
    from stratlab import builder_grammar as G
    from stratlab.builder_source import make_source
    b = builder.Builder(root)
    src = make_source(source, data_root)
    inp = (b.state()["batches"].get(bid) or {}).get("inputs")
    if source == "bundle" and (not inp or src.man is None or
                               __import__("stratlab.builder_bundle", fromlist=["x"]).manifest_sha(data_root)
                               != inp["manifest_sha256"]):
        from stratlab.builder_bundle import BundleCorrupt
        raise BundleCorrupt(f"batch {bid}: the frozen inputs at {data_root} are not the ones in its registry record")
    doc = b.verify_frozen(bid, src.snapshot())
    rec = b.state()["batches"][bid]
    cands = [(r["id"], G.Candidate.from_json(r["cand"])) for r in doc["candidates"]]
    t0 = time.time()
    E = BE.Engine(src)
    E.cache_dir = str(builder.cache_dir())
    E.log = log
    batch = EV.Batch(E, bid, cands, b.rule, str(b.batch_dir(bid) / "work"), int(rec["n_before"]), log,
                     open_final=lambda ids: b.open_final(bid, ids))
    jobs = max(1, min(2, int(jobs)))
    if jobs > 1:
        import multiprocessing as mp
        todo = [a for a in G.AGENTS if not os.path.exists(batch.p(f"discovery_{a}.json"))]
        if any(c.agent == "options" for _i, c in cands):
            from stratlab import builder_options as BO
            BO.ensure_loaded(E)                         # build the compact option tables once, before forking
        _W["batch"] = batch
        with mp.get_context("fork").Pool(jobs) as pool:
            for a in pool.imap_unordered(_disc_worker, todo):
                log(f"[builder {bid}] worker finished {a}")
    summary = batch.run()
    summary["seconds"] = round(time.time() - t0, 1)
    summary["peak_rss_mb"] = _peak_mb()
    summary["source"] = source
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--forward", action="store_true", help="builder/v1 forward family: trades up to --day")
    ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--snapshot", action="store_true", help="write the data snapshot ids to <root>/snapshot.json")
    ap.add_argument("--bundle", action="store_true", help="with --snapshot: also write the frozen inputs")
    ap.add_argument("--source", default="server")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mem-mb", type=int, default=3584)
    a = ap.parse_args(argv)
    _limit(a.mem_mb)
    if a.snapshot:
        import builder
        from stratlab.builder_source import make_source
        src = make_source(a.source, a.data_root)
        snap = src.snapshot()
        inputs = builder.write_inputs(a.root, src) if a.bundle else None
        os.makedirs(a.root, exist_ok=True)
        tmp = os.path.join(a.root, "snapshot.json.tmp")
        with open(tmp, "w") as fh:
            json.dump({"snapshot": snap, "inputs": inputs}, fh, default=str)
        os.replace(tmp, os.path.join(a.root, "snapshot.json"))
        return 0
    if a.forward:
        import builder_forward
        work = os.path.join(a.root, "work")
        os.makedirs(work, exist_ok=True)
        try:
            res = builder_forward.forward_trades(a.root, a.data_root, int(a.day), a.source)
            tmp = os.path.join(work, f"forward_{a.day}.json.gz.tmp")
            with gzip.open(tmp, "wt") as fh:
                json.dump(res, fh, default=str)
            os.replace(tmp, os.path.join(work, f"forward_{a.day}.json.gz"))
            return 0
        except Exception as exc:
            with open(os.path.join(work, "error.txt"), "w") as fh:
                fh.write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if not a.batch:
        ap.error("--batch is required")
    work = os.path.join(a.root, "batches", a.batch, "work")
    os.makedirs(work, exist_ok=True)

    def log(msg):
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)
    try:
        s = evaluate(a.root, a.batch, a.source, a.data_root, a.jobs, log)
        tmp = os.path.join(work, "summary.json.gz.tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(s, fh, default=str)
        os.replace(tmp, os.path.join(work, "summary.json.gz"))
        log(f"[builder {a.batch}] done in {s['seconds']}s, peak RSS {s['peak_rss_mb']} MB, proven {s['proven']}")
        return 0
    except Exception as exc:
        with open(os.path.join(work, "error.txt"), "w") as fh:
            fh.write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        log(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
