"""
stratlab/builder_catalog.py — loads the GENERATED catalogue of promoted builder strategies (<builder root>/catalog/
builder_gen.py, written only by builder.write_catalog for proven candidates) into runnable specs. It is deliberately
NOT in stratlab/catalog/ (whose every module is imported into stratlab.api.REGISTRY and so into the forward lab's frozen
1,276-strategy family): builder strategies live in their own forward family, builder/v1.
"""
from __future__ import annotations

import dataclasses
import os

from . import builder_grammar as G


@dataclasses.dataclass(frozen=True)
class BuilderSpec:
    id: str
    batch: str
    key: str
    cand: G.Candidate
    text: str
    frozen_hash: str
    registry_hash: str

    @property
    def kind(self) -> str:
        return self.cand.agent

    @property
    def family(self) -> str:
        return f"builder/{self.cand.agent}/{self.cand.block}"

    @property
    def fingerprint(self) -> str:
        import hashlib
        return hashlib.sha1(self.key.encode()).hexdigest()[:12]

    def run(self, engine, lo: int, hi: int) -> list:
        """Trade tuples (sim.TRADE_FIELDS) with entry in [lo, hi] through the builder engine."""
        from . import builder_engine as BE
        return BE.run_jobs(engine, [BE.Job(self.id, self.cand, None, lo, hi, False)])[self.id]["trades"]


def load(path: str) -> list:
    if not os.path.exists(path):
        return []
    ns: dict = {}
    with open(path) as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    return [BuilderSpec(s["id"], s["batch"], s["key"], G.Candidate.from_json(s["cand"]), s["text"], s["frozen_hash"],
                        s["registry_hash"]) for s in ns.get("SPECS", [])]
