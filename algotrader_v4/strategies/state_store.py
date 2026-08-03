"""
strategies/state_store.py
SQLite persistence for the positional system — the spec's non-negotiable
for unattended operation: open positions / strategy state survive
restarts, intended orders are idempotent (unique client tag, marked
"pending" before placement and "placed"/"failed" after), and a trade log
feeds the kill criteria.

Layout:
  kv            engine + strategy state snapshot (single JSON row)
  pending_plans intended next-open orders with lifecycle status
  trade_log     closed-trade P&L per strategy (kill-criteria input)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from strategies.positional_engine import OrderPlan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_plans (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created    TEXT NOT NULL,              -- signal date (ISO)
    client_tag TEXT NOT NULL UNIQUE,       -- idempotency key
    plan       TEXT NOT NULL,              -- OrderPlan as JSON
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|placed|failed|cancelled
    order_id   TEXT DEFAULT '',
    gtt_id     TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS trade_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    closed   TEXT NOT NULL,
    strategy TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    pnl      REAL NOT NULL
);
"""


class PositionalStateStore:

    def __init__(self, path: str = "logs/positional_state.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Engine state snapshot ──────────────────────────────────────────────

    def save_engine_state(self, state: dict) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES ('engine', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (json.dumps(state),))
        self._conn.commit()

    def load_engine_state(self) -> dict | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = 'engine'").fetchone()
        return json.loads(row[0]) if row else None

    # ── Pending order plans (idempotent placement) ─────────────────────────

    def queue_plans(self, signal_date: str, plans: list[OrderPlan]) -> int:
        """Store intended orders; duplicate client tags are ignored so a
        re-run of the EOD job never double-queues."""
        from dataclasses import asdict
        n = 0
        for i, p in enumerate(plans):
            tag = f"{signal_date}:{p.strategy}:{p.symbol}:{p.action}:{i}"
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO pending_plans (created, client_tag, plan) "
                "VALUES (?, ?, ?)", (signal_date, tag, json.dumps(asdict(p))))
            n += cur.rowcount
        self._conn.commit()
        return n

    def pending_plans(self) -> list[tuple[int, OrderPlan]]:
        rows = self._conn.execute(
            "SELECT id, plan FROM pending_plans WHERE status = 'pending' "
            "ORDER BY id").fetchall()
        return [(rid, OrderPlan(**json.loads(blob))) for rid, blob in rows]

    def mark_plan(self, plan_id: int, status: str,
                  order_id: str = "", gtt_id: str = "") -> None:
        self._conn.execute(
            "UPDATE pending_plans SET status = ?, order_id = ?, gtt_id = ? "
            "WHERE id = ?", (status, order_id, gtt_id, plan_id))
        self._conn.commit()

    def cancel_stale_pending(self, before_date: str) -> int:
        """Plans not executed by their next session are stale — the EOD run
        will regenerate anything still valid from fresh bars."""
        cur = self._conn.execute(
            "UPDATE pending_plans SET status = 'cancelled' "
            "WHERE status = 'pending' AND created < ?", (before_date,))
        self._conn.commit()
        return cur.rowcount

    # ── Trade log (kill-criteria input) ────────────────────────────────────

    def record_trade(self, closed: str, strategy: str,
                     symbol: str, pnl: float) -> None:
        self._conn.execute(
            "INSERT INTO trade_log (closed, strategy, symbol, pnl) "
            "VALUES (?, ?, ?, ?)", (closed, strategy, symbol, pnl))
        self._conn.commit()

    def trade_pnls(self, strategy: str) -> list[float]:
        rows = self._conn.execute(
            "SELECT pnl FROM trade_log WHERE strategy = ? ORDER BY id",
            (strategy,)).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
