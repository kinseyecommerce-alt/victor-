"""
stratlab/builder_source.py — where the strategy builder's data come from. Read-only; nothing here computes a return.

Two sources with one interface (ctx(), minute_parts(sym), option_dirs(), nifty_daily(), end, snapshot()):
  LabSource     the research sandbox: datagen_features' panels (events_daily point-in-time top-500 daily cache 2012 ->,
                the old-era + 3-year minute caches 2016 -> through corrected.py, NSE EOD files of both eras) and, for
                index options, nsedata-layout directories of both eras (nse_old_view 2012-2023 + nse 2023 ->).
  ServerSource  the server's builder data layer (built by builder_data.py, which this module never imports or
                edits): <root>/minute (forward-lab store format, read with forward_lab_store.Store.load_symbol),
                <root>/nse (STRATLAB_NSE_DIR layout), <root>/flow (deals, corporate actions), <root>/events (results).
                Panels are built from those files by stratlab.builder_server_panels with the same feature code.
snapshot() = the data snapshot ids frozen into a batch before evaluation: per input, its path, size and sha256 (files)
or file count / newest name / total size (directories), plus the snapshot end T.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def _sha_file(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
            n += len(b)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def file_id(path: str) -> dict:
    if not os.path.exists(path):
        return {"path": path, "missing": True}
    if os.path.isdir(path):
        names, size = [], 0
        for root, _d, files in os.walk(path):
            for f in files:
                if f.startswith(".") or f.endswith(".tmp"):
                    continue
                p = os.path.join(root, f)
                names.append(os.path.relpath(p, path))
                try:
                    size += os.path.getsize(p)
                except OSError:
                    pass
        names.sort()
        h = hashlib.sha256("\n".join(names).encode()).hexdigest()
        return {"path": path, "files": len(names), "newest": names[-1] if names else None, "bytes": size,
                "names_sha256": h}
    return {"path": path, "bytes": os.path.getsize(path), "sha256": _sha_file(path)}


class LabSource:
    kind = "lab"

    def __init__(self):
        from . import datagen_features as F
        self.F = F
        self._ctx = None
        self._parts: dict = {}

    def ctx(self):
        if self._ctx is None:
            from . import datagen_run as R
            self._ctx = R.context()
        return self._ctx

    @property
    def end(self) -> int:
        c = self.ctx()
        return int(min(c.daily.cal[-1], c.minute.cal[-1]))

    def minute_parts(self, sym: str) -> list:
        """SymbolData of each era (corrected overlay installed by context())."""
        from . import data as D
        self.ctx()
        out = []
        for era in ("old", "3y"):
            sd = D.load_symbol(self.F.MIN_CACHES[era], sym)
            if sd is not None:
                out.append(sd)
        return out

    def option_dirs(self) -> list:
        """[(nse dir, bars dir, first day, last day)] in time order (nsedata layout)."""
        S = self.F.SCRATCH
        return [(os.path.join(S, "nse_old_view"), os.path.join(S, "nse_old_view", "bars"), 0, self.F.OLD_END),
                (os.path.join(S, "nse"), os.path.join(S, "bars3y"), self.F.OLD_END + 1, 99999999)]

    def nifty_daily(self) -> dict | None:
        from . import data_daily as DD
        from . import events_daily as ED
        return DD.load_daily_aux(ED.CACHE, "_NIFTY50_day")

    def snapshot(self) -> dict:
        F = self.F
        from . import events_daily as ED
        ins = {
            "minute_summary": os.path.join(F.CACHE, "minsum.npz"),
            "delivery_panel": os.path.join(F.CACHE, "delivery_panel.npz"),
            "oi_panel": os.path.join(F.CACHE, "oi_panel.npz"),
            "expiries": os.path.join(F.CACHE, "expiries.json"),
            "events_daily_manifest": os.path.join(ED.CACHE, "_manifest.json"),
            "minute_old_manifest": os.path.join(F.MIN_CACHES["old"], "_manifest.json"),
            "minute_3y_manifest": os.path.join(F.MIN_CACHES["3y"], "_manifest.json"),
            "ban_new": os.path.join(F.NSE_NEW, "ban.csv.gz"),
            "ban_old": os.path.join(F.OLD_VIEW, "ban.csv.gz"),
            "participant_new": os.path.join(F.NSE_NEW, "participant_oi.csv.gz"),
            "participant_old": os.path.join(F.OLD_VIEW, "participant_oi.csv.gz"),
            "deals": os.path.join(F.FLOWS, "deals", "deals.csv.gz"),
            "corpactions": os.path.join(F.FLOWS, "corpactions", "corpactions.csv.gz"),
            "results": os.path.join(F.EVENTS, "results.csv.gz"),
            "fo_status_old": os.path.join(F.OLD_VIEW, "fo_parts", "status.csv"),
            "fo_status_new": os.path.join(F.NSE_NEW, "fo_parts", "status.csv"),
        }
        out = {k: file_id(v) for k, v in ins.items()}
        for k, (d, _b, _lo, _hi) in zip(("optidx_old", "optidx_new"), self.option_dirs()):
            out[k] = file_id(os.path.join(d, "fo_parts", "optidx"))
        c = self.ctx()
        return {"source": "lab", "end": self.end, "daily_calendar": [int(c.daily.cal[0]), int(c.daily.cal[-1])],
                "minute_calendar": [int(c.minute.cal[0]), int(c.minute.cal[-1])], "inputs": out}


def snapshot_sha(snap: dict) -> str:
    return hashlib.sha256(json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_source(kind: str = "lab", root: str | None = None):
    if kind == "lab":
        return LabSource()
    if kind == "server":
        from .builder_server_panels import ServerSource
        return ServerSource(root)
    if kind == "bundle":
        from .builder_bundle import BundleSource
        return BundleSource(root)
    if kind == "synthetic":
        from .builder_server_panels import SyntheticSource
        return SyntheticSource(root)
    raise KeyError(kind)


__all__ = ["LabSource", "make_source", "snapshot_sha", "file_id", "np"]
