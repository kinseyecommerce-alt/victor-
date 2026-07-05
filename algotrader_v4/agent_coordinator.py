"""
agent_coordinator.py — Central arbiter over the agent bus.

The four MCX agents run independently, but they share one account and a set of
correlated contracts. The coordinator sits on top of `agent_bus` and decides,
for every intended entry, whether it may proceed — preventing the agents from
working against each other:

  1. Conflict block   — no two agents may hold the same contract in opposite
                        directions (they'd just pay spread fighting each other).
  2. Duplicate block  — one agent owns a contract at a time; a second agent
                        can't stack the same contract.
  3. Group exposure   — correlated contracts (all bullion, all energy, …) share
                        one margin budget; entries that would blow the group cap
                        are rejected.
  4. Global cap       — a hard ceiling on total concurrent coordinated positions.
  5. Peer conviction  — if a peer agent independently signals the SAME direction
                        on the contract, size is boosted (higher conviction);
                        if a peer signals the OPPOSITE direction, size is damped.

`request()` is the transactional entry point agents call before ordering: it
evaluates and, on approval, atomically reserves the slot (no await in between,
so concurrent agent tasks can't both win the same contract). `release()` frees
it on exit/rollback. `evaluate()` is the pure, side-effect-free preview used by
tests and the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from agent_bus import agent_bus, TOPIC_SIGNAL, TOPIC_REGIME
import mcx_universe


@dataclass
class Reservation:
    agent:  str
    symbol: str
    side:   str      # BUY / SELL
    lots:   int
    group:  str
    margin: float


@dataclass
class Decision:
    allowed:      bool
    size_factor:  float = 1.0
    reason:       str = "OK"
    peers_agree:  list[str] = field(default_factory=list)
    peers_oppose: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed, "size_factor": self.size_factor,
            "reason": self.reason, "peers_agree": self.peers_agree,
            "peers_oppose": self.peers_oppose,
        }


class AgentCoordinator:
    # Defaults; overridden from settings when available
    DEFAULT_MAX_CONCURRENT   = 6
    DEFAULT_GROUP_MARGIN_CAP = 250000.0   # ₹ per correlation group
    BOOST_PER_PEER           = 0.25
    BOOST_CAP                = 0.5        # max +50% size from agreeing peers
    DAMP_PER_PEER            = 0.25
    DAMP_FLOOR               = 0.5        # never damp below 50%

    def __init__(self) -> None:
        self._book: dict[str, Reservation] = {}   # symbol → reservation

    # ── Config (read live so runtime tweaks apply) ───────────────────────────────
    def _max_concurrent(self) -> int:
        try:
            from config import settings
            return int(getattr(settings, "coord_max_concurrent_positions",
                               self.DEFAULT_MAX_CONCURRENT))
        except Exception:
            return self.DEFAULT_MAX_CONCURRENT

    def _group_cap(self) -> float:
        try:
            from config import settings
            return float(getattr(settings, "coord_group_margin_cap",
                                 self.DEFAULT_GROUP_MARGIN_CAP))
        except Exception:
            return self.DEFAULT_GROUP_MARGIN_CAP

    def _enabled(self) -> bool:
        try:
            from config import settings
            return bool(getattr(settings, "use_agent_coordinator", True))
        except Exception:
            return True

    # ── Exposure helpers ─────────────────────────────────────────────────────────
    def _group_margin(self, group: str, exclude_symbol: str | None = None) -> float:
        return sum(
            r.margin for s, r in self._book.items()
            if r.group == group and s != exclude_symbol
        )

    def _margin_for(self, symbol: str, lots: int) -> float:
        c = mcx_universe.contract(symbol)
        per_lot = c.margin_per_lot if c else 0.0
        return per_lot * max(lots, 1)

    # ── Peer conviction (read from the bus) ──────────────────────────────────────
    def _peer_alignment(self, symbol: str, side: str, agent: str) -> tuple[list[str], list[str]]:
        agree, oppose = [], []
        for peer_agent, msg in agent_bus.peer_signal(symbol, agent).items():
            peer_side = msg.payload.get("action") or msg.payload.get("side")
            if peer_side in ("BUY", "SELL"):
                (agree if peer_side == side else oppose).append(peer_agent)
        return agree, oppose

    def _regime_factor(self) -> float:
        msg = None
        latest = agent_bus.latest_on_topic(TOPIC_REGIME)
        if latest:
            msg = max(latest, key=lambda m: m.seq)
        if msg:
            try:
                return float(msg.payload.get("size_factor", 1.0))
            except Exception:
                return 1.0
        return 1.0

    # ── Core decision (pure) ─────────────────────────────────────────────────────
    def evaluate(self, agent: str, symbol: str, side: str, lots: int) -> Decision:
        if not self._enabled():
            return Decision(allowed=True, reason="coordinator disabled")

        if side not in ("BUY", "SELL"):
            return Decision(allowed=False, reason=f"invalid side {side}")

        group  = mcx_universe.group_of(symbol)
        margin = self._margin_for(symbol, lots)

        # 1 & 2 — contract already spoken for
        held = self._book.get(symbol)
        if held is not None:
            if held.side != side:
                return Decision(allowed=False,
                                reason=f"conflict: {held.agent} holds {symbol} {held.side}")
            return Decision(allowed=False,
                            reason=f"duplicate: {held.agent} already holds {symbol}")

        # 4 — global concurrency cap
        if len(self._book) >= self._max_concurrent():
            return Decision(allowed=False,
                            reason=f"global cap {self._max_concurrent()} positions reached")

        # 3 — correlated-group margin cap
        group_cap = self._group_cap()
        projected = self._group_margin(group) + margin
        if projected > group_cap:
            return Decision(allowed=False,
                            reason=(f"group '{group}' margin ₹{projected:.0f} "
                                    f"exceeds cap ₹{group_cap:.0f}"))

        # 5 — peer conviction → size factor
        agree, oppose = self._peer_alignment(symbol, side, agent)
        sf = 1.0
        if agree:
            sf += min(len(agree) * self.BOOST_PER_PEER, self.BOOST_CAP)
        if oppose:
            sf = max(self.DAMP_FLOOR, sf - len(oppose) * self.DAMP_PER_PEER)
        sf = round(sf * self._regime_factor(), 3)

        return Decision(allowed=True, size_factor=sf, reason="OK",
                        peers_agree=agree, peers_oppose=oppose)

    # ── Transactional request (evaluate + reserve) ───────────────────────────────
    def request(self, agent: str, symbol: str, side: str, lots: int) -> Decision:
        decision = self.evaluate(agent, symbol, side, lots)
        if decision.allowed:
            self._book[symbol] = Reservation(
                agent=agent, symbol=symbol, side=side, lots=lots,
                group=mcx_universe.group_of(symbol),
                margin=self._margin_for(symbol, lots),
            )
            self._persist()
        # Broadcast the arbitration outcome so the dashboard/peers can see it
        agent_bus.publish("coordinator", "decision",
                          {**decision.as_dict(), "symbol": symbol,
                           "requesting_agent": agent, "side": side, "lots": lots},
                          key=symbol)
        if not decision.allowed:
            logger.debug("[coordinator] veto {} {} {}: {}",
                         agent, side, symbol, decision.reason)
        return decision

    def release(self, symbol: str, agent: str | None = None) -> None:
        held = self._book.get(symbol)
        if held is None:
            return
        if agent is not None and held.agent != agent:
            return
        self._book.pop(symbol, None)
        self._persist()

    # ── Durable state + reconciliation ───────────────────────────────────────────
    _STATE_FILE = "coordinator_state.json"

    def _persist(self) -> None:
        try:
            from state_store import save_json
            save_json(self._STATE_FILE, {"book": self.book()})
        except Exception:
            pass

    def load(self) -> None:
        """Restore the reserved book from disk (call on startup, before reconcile)."""
        try:
            from state_store import load_json
            data = load_json(self._STATE_FILE, {}) or {}
        except Exception:
            data = {}
        for r in data.get("book", []):
            sym = r.get("symbol")
            if not sym:
                continue
            self._book[sym] = Reservation(
                agent=r.get("agent", ""), symbol=sym, side=r.get("side", "BUY"),
                lots=int(r.get("lots", 1)), group=r.get("group", mcx_universe.group_of(sym)),
                margin=float(r.get("margin", 0.0)),
            )
        if self._book:
            logger.info("[coordinator] restored {} reservation(s) from disk", len(self._book))

    def reconcile(self, positions: list[dict]) -> dict:
        """Reconcile the reserved book against the broker's ACTUAL open positions.

        `positions` = broker net positions [{tradingsymbol, quantity, ...}]. Any
        reservation with no matching live position is dropped (stale after a
        restart/disconnect); any live position not reserved is adopted so the
        coordinator won't hand its contract to another agent.
        """
        live: dict[str, int] = {}
        for p in positions or []:
            q = p.get("quantity", 0)
            if q:
                # positions may carry the futures tradingsymbol; match on base prefix
                sym = p.get("tradingsymbol", "")
                base = next((s for s in mcx_universe.MCX_SYMBOLS if sym.startswith(s)), sym)
                live[base] = q
        dropped, adopted = [], []
        for sym in list(self._book):
            if sym not in live:
                self._book.pop(sym, None)
                dropped.append(sym)
        for sym, q in live.items():
            if sym not in self._book:
                self._book[sym] = Reservation(
                    agent="reconciled", symbol=sym, side="BUY" if q > 0 else "SELL",
                    lots=max(1, abs(q) // mcx_universe.lot_size(sym)),
                    group=mcx_universe.group_of(sym), margin=self._margin_for(sym, 1))
                adopted.append(sym)
        if dropped or adopted:
            logger.info("[coordinator] reconciled: dropped {} stale, adopted {} live",
                        dropped, adopted)
        self._persist()
        return {"dropped": dropped, "adopted": adopted, "open_positions": len(self._book)}

    # ── Introspection ────────────────────────────────────────────────────────────
    def book(self) -> list[dict]:
        return [
            {"symbol": r.symbol, "agent": r.agent, "side": r.side,
             "lots": r.lots, "group": r.group, "margin": r.margin}
            for r in self._book.values()
        ]

    def group_exposure(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._book.values():
            out[r.group] = out.get(r.group, 0.0) + r.margin
        return out

    def status(self) -> dict:
        return {
            "enabled":            self._enabled(),
            "open_positions":     len(self._book),
            "max_concurrent":     self._max_concurrent(),
            "group_margin_cap":   self._group_cap(),
            "group_exposure":     self.group_exposure(),
            "book":               self.book(),
        }

    def reset(self) -> None:
        self._book.clear()


# Singleton
agent_coordinator = AgentCoordinator()
