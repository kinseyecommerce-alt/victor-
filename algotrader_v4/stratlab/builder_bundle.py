"""
stratlab/builder_bundle.py — the IMMUTABLE INPUTS of a frozen batch (so an evaluation that died can resume on exactly
the data it was frozen on, whatever the data layer has written since).

At freeze time (in the snapshot process) write_bundle(src, dest) stores, under <builder root>/inputs/<id>/:
  arrays.npz     every panel and feature the engine reads (datagen_features.Ctx: the daily panel, the minute summary
                 panel, delivery %, OI change, ban list, deals, participant OI, results timing, corporate actions,
                 results-before-the-open, index expiries) and the NIFTY daily bars - compressed
  options.npz    the compact real-option tables (stratlab/builder_options.build_tables)
  minute/        the 1-minute bars the walks read: HARD LINKS to the data layer's forward_lab_store month files (the
                 store replaces a file atomically, so a link keeps the frozen bytes; it costs disk only once the data
                 layer replaces that file - builder.dir_bytes counts a linked file only when it is the last link);
                 copied when a link is impossible (another filesystem)
  manifest.json  the frozen data snapshot (exactly the dict hashed into the batch's registry record), T, and the
                 sha256 + size of every file above
BundleSource(dest) is a data source (stratlab/builder_source.py interface) that refuses to load (BundleCorrupt) unless
every file still has its frozen sha256. The FIRST evaluation of a batch already runs from its bundle, so a retry sees
identical inputs by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np

from . import datagen_features as F

D_FEATURES = ("dp", "doi", "ban", "react")
DAILY_KEYS = ("o", "h", "l", "c", "v", "member", "fo")
MIN_FIELDS = F.MIN_FIELDS


class BundleCorrupt(RuntimeError):
    pass


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _link_or_copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_bundle(src, dest: str, log=None) -> dict:
    """Materialise the source's inputs in `dest` (must not exist). Returns the manifest."""
    if os.path.exists(dest):
        raise FileExistsError(dest)
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    snap = src.snapshot()
    ctx = src.ctx()
    P, M = ctx.daily, ctx.minute
    arr = {"syms": np.array(P.syms, dtype=object), "cal": P.cal, "nifty_o": P.nifty["o"], "nifty_c": P.nifty["c"],
           "vix_c": P.vix["c"], "excl": json.dumps({str(k): v.tolist() for k, v in P.excl.items()}),
           "msyms": np.array(M.syms, dtype=object), "mcal": M.cal, "special": np.array(sorted(M.special), np.int64),
           "dirty": json.dumps({str(k): v.tolist() for k, v in M.dirty.items()})}
    ul = getattr(src, "universe_lists", None)
    if ul is not None:                     # the universe lists the engine resolves NIFTY100 / NIFTY200 from
        arr["universe"] = json.dumps(ul(), sort_keys=True, default=str)
    for k in DAILY_KEYS:
        arr["d_" + k] = getattr(P, k)
    for k in MIN_FIELDS:
        arr["m_" + k] = M.f[k]
    for k in D_FEATURES:
        arr["f_" + k] = ctx.d(k)
    deals = ctx.d("deals")
    arr["f_deals_inst"], arr["f_deals_other"] = deals["inst"], deals["other"]
    part = ctx.d("part")
    arr["f_part"] = json.dumps({k: [None if x != x else float(x) for x in v.tolist()] for k, v in part.items()})
    arr["f_ca"] = json.dumps(ctx.d("ca"), default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    arr["f_pre"] = ctx.m("pre")
    arr["f_exp"] = json.dumps(ctx.m("exp"))
    nd = src.nifty_daily()
    if nd is not None:
        for k in ("date", "o", "h", "l", "c"):
            arr["nd_" + k] = np.asarray(nd[k])
    np.savez_compressed(os.path.join(tmp, "arrays.npz"), **arr)
    # compact option tables
    from . import builder_options as BO
    tabs = BO.build_tables(src.option_dirs(), log)
    BO._save(os.path.join(tmp, "options.npz"), tabs)
    # minute bars: links to the store's month files
    how = {"link": 0, "copy": 0}
    mroot = getattr(src, "minute_root", None)
    if mroot and os.path.isdir(os.path.join(mroot, "bars")):
        for dp, _dn, fn in os.walk(os.path.join(mroot, "bars")):
            for f in fn:
                if f.endswith(".tmp"):
                    continue
                s = os.path.join(dp, f)
                rel = os.path.relpath(s, mroot)
                how[_link_or_copy(s, os.path.join(tmp, "minute", rel))] += 1
        man = os.path.join(mroot, "manifest.json")
        if os.path.exists(man):
            shutil.copy2(man, os.path.join(tmp, "minute", "manifest.json"))
    files = {}
    for dp, _dn, fn in os.walk(tmp):
        for f in fn:
            p = os.path.join(dp, f)
            files[os.path.relpath(p, tmp)] = {"sha256": _sha(p), "bytes": os.path.getsize(p)}
    manifest = {"version": "bb1", "snapshot": snap, "T": int(src.end), "upto": getattr(src, "upto", None),
                "files": files, "minute_files": how}
    with open(os.path.join(tmp, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, sort_keys=True, default=str)
    os.replace(tmp, dest)
    return manifest


def manifest_sha(dest: str) -> str:
    return _sha(os.path.join(dest, "manifest.json"))


def verify(dest: str, expect_manifest_sha: str | None = None) -> dict:
    """Raise BundleCorrupt unless the bundle exists and every file has its frozen sha256."""
    mp = os.path.join(dest, "manifest.json")
    if not os.path.exists(mp):
        raise BundleCorrupt(f"frozen inputs missing: {dest}")
    if expect_manifest_sha and _sha(mp) != expect_manifest_sha:
        raise BundleCorrupt("frozen inputs: manifest changed")
    with open(mp) as fh:
        man = json.load(fh)
    for rel, meta in man["files"].items():
        p = os.path.join(dest, rel)
        if not os.path.exists(p) or os.path.getsize(p) != meta["bytes"] or _sha(p) != meta["sha256"]:
            raise BundleCorrupt(f"frozen inputs: {rel} missing or changed")
    return man


class BundleSource:
    kind = "bundle"

    def __init__(self, dest: str, expect_manifest_sha: str | None = None):
        self.dest = str(dest)
        self.man = verify(self.dest, expect_manifest_sha)
        self.minute_root = os.path.join(self.dest, "minute")
        self.upto = self.man.get("upto")
        self._ctx = None
        self._store = None
        self.option_tables_path = os.path.join(self.dest, "options.npz")

    @property
    def end(self) -> int:
        return int(self.man["T"])

    def snapshot(self) -> dict:
        return self.man["snapshot"]

    def option_dirs(self) -> list:
        return []

    def ctx(self):
        if self._ctx is None:
            z = np.load(os.path.join(self.dest, "arrays.npz"), allow_pickle=True)
            excl = {int(k): np.asarray(v, np.int64) for k, v in json.loads(str(z["excl"])).items()}
            P = F.DailyPanel(list(z["syms"]), z["cal"], *(z["d_" + k] for k in ("o", "h", "l", "c", "v")),
                             z["d_member"], z["d_fo"], {"o": z["nifty_o"], "c": z["nifty_c"]}, {"c": z["vix_c"]}, excl)
            dirty = {int(k): np.asarray(v, np.int64) for k, v in json.loads(str(z["dirty"])).items()}
            M = F.MinutePanel(list(z["msyms"]), z["mcal"], {k: z["m_" + k] for k in MIN_FIELDS},
                              set(int(x) for x in z["special"]), dirty)
            ctx = F.Ctx(P, M)
            for k in D_FEATURES:
                ctx._c["d:" + k] = z["f_" + k]
            ctx._c["d:deals"] = {"inst": z["f_deals_inst"], "other": z["f_deals_other"]}
            ctx._c["d:part"] = {k: np.array([np.nan if x is None else x for x in v], float)
                                for k, v in json.loads(str(z["f_part"])).items()}
            ca = json.loads(str(z["f_ca"]))
            ctx._c["d:ca"] = {k: [tuple(x) for x in v] for k, v in ca.items()}
            ctx._c["m:pre"] = z["f_pre"]
            ctx._c["m:exp"] = json.loads(str(z["f_exp"]))
            self._nd = ({k: z["nd_" + k] for k in ("date", "o", "h", "l", "c")} if "nd_date" in z.files else None)
            self._ul = json.loads(str(z["universe"])) if "universe" in z.files else None
            self._ctx = ctx
        return self._ctx

    def universe_lists(self):
        """As frozen with the batch (a bundle written before the bg2 data layer has none: every stock is base)."""
        self.ctx()
        return self._ul

    def nifty_daily(self):
        self.ctx()
        return self._nd

    def minute_parts(self, sym: str) -> list:
        if self._store is None:
            import forward_lab_store as FS
            self._store = FS.Store(self.minute_root)
        sd = self._store.load_symbol(sym, upto=self.upto)
        return [sd] if sd is not None else []
