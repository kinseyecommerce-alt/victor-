"""
bg_compute.py — run heavy backtests in a separate process.

A 45-day replay of ten symbols pushes 100k minute bars through the agents' Python code and
takes a minute of CPU; the walk-forward optimiser runs dozens of them. Done on the server's
event loop, that freezes everything: ticks, the dashboard, even /health. So anything that
replays history runs here, in a fresh interpreter started as `python bg_worker.py`, which reads
one JSON job from stdin and writes one JSON result to stdout. The child inherits the env (and
so the vault), can load Kite history itself, and never touches the server's event loop. One job
runs at a time: the container has one or two cores.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_HERE = Path(__file__).resolve().parent
_sem = asyncio.Semaphore(1)
JOB_TIMEOUT_S = 3600


async def run_in_process(fn: str, *args: Any, timeout: float = JOB_TIMEOUT_S) -> Any:
    """Run the job function `fn` (a name in this module) with JSON-serialisable args in a worker
    process and return its JSON result."""
    if fn not in JOBS:
        raise ValueError(f"unknown background job {fn}")
    async with _sem:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(_HERE / "bg_worker.py"), cwd=str(_HERE),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        payload = json.dumps({"fn": fn, "args": list(args)}, default=str).encode()
        try:
            out, err = await asyncio.wait_for(proc.communicate(payload), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"background job {fn} exceeded {timeout:.0f}s and was killed")
        if proc.returncode != 0:
            tail = err.decode(errors="replace").strip().splitlines()[-5:]
            logger.error("[bg] job {} failed (rc={}): {}", fn, proc.returncode, " | ".join(tail))
            raise RuntimeError(f"background job {fn} failed: {' | '.join(tail)[-300:]}")
        try:
            return json.loads(out.decode())
        except Exception as exc:
            raise RuntimeError(f"background job {fn} returned no JSON: {out[:200]!r}") from exc


# ── Jobs: run in the child; results must be JSON-serialisable ────────────

def replay_job(agent: str, symbols: list[str], days: int, slippage_bps: float, params: Optional[dict],
               include_trades: bool) -> dict:
    import replay_backtest as rb
    res = rb.replay(agent, symbols, days, slippage_bps, params)
    return {**res.to_dict(include_trades=include_trades), "benchmark": rb.benchmark_for(res)}


def optimise_job(agent: str, symbols: list[str], days: int) -> dict:
    """Walk-forward search without adopting: the parent saves/applies the winner."""
    import optimizer
    return optimizer.optimise(agent, symbols, days, apply=False)


def ab_test_job(agent: str, param: str, value: float, symbols: list[str], days: int) -> dict:
    import research_loop
    return asyncio.run(research_loop.ab_test(agent, param, value, symbols, days, bars_by_symbol=None, _in_worker=True))


def pid_job() -> int:
    return os.getpid()


def sleep_job(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


JOBS = {"replay_job": replay_job, "optimise_job": optimise_job, "ab_test_job": ab_test_job,
        "pid_job": pid_job, "sleep_job": sleep_job}
