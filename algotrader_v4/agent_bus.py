"""
agent_bus.py — Inter-agent message bus (shared blackboard).

Every MCX trading agent publishes to this bus and can read what its peers have
published. It is the substrate for agent-to-agent communication: agents
broadcast their live signals, their intended entries, their fills and their
current exposure; the coordinator (agent_coordinator.py) reads the bus to
arbitrate between agents before any order is placed.

Design: a single in-process blackboard. Runs inside the FastAPI event loop
(single-threaded asyncio), so plain dict operations are safe without locks.
Two views are maintained:

  • _latest — the most recent message per (topic, key), for O(1) "what does
    agent X think about symbol Y right now?" lookups.
  • _log    — a capped rolling history for the dashboard / debugging.

Topics are strings (see the TOPIC_* constants). `key` is usually the symbol,
falling back to the publishing agent when a message is not symbol-scoped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ist_clock import now_ist


# ── Topics ──────────────────────────────────────────────────────────────────────
TOPIC_SIGNAL    = "signal"     # an agent's evaluate_tick verdict for a symbol
TOPIC_INTENT    = "intent"     # an agent intends to enter (pre-order broadcast)
TOPIC_FILL      = "fill"       # an agent entered a position
TOPIC_EXIT      = "exit"       # an agent closed a position
TOPIC_EXPOSURE  = "exposure"   # an agent's current open exposure snapshot
TOPIC_REGIME    = "regime"     # market-regime broadcast (from master agent)
TOPIC_HEARTBEAT = "heartbeat"  # agent liveness / stats


@dataclass
class AgentMessage:
    seq:     int
    ts:      str            # ISO timestamp (IST)
    agent:   str
    topic:   str
    key:     str            # symbol, or agent name when not symbol-scoped
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "ts": self.ts, "agent": self.agent,
            "topic": self.topic, "key": self.key, "payload": self.payload,
        }


class AgentBus:
    def __init__(self, log_capacity: int = 500) -> None:
        self._seq = 0
        self._log: list[AgentMessage] = []
        self._latest: dict[tuple[str, str], AgentMessage] = {}
        self._subscribers: list[Callable[[AgentMessage], None]] = []
        self._log_capacity = log_capacity

    # ── Publish ────────────────────────────────────────────────────────────────
    def publish(
        self, agent: str, topic: str, payload: dict[str, Any],
        key: Optional[str] = None,
    ) -> AgentMessage:
        self._seq += 1
        msg = AgentMessage(
            seq=self._seq, ts=now_ist().isoformat(), agent=agent,
            topic=topic, key=key or agent, payload=dict(payload),
        )
        self._latest[(topic, msg.key)] = msg
        self._log.append(msg)
        if len(self._log) > self._log_capacity:
            self._log = self._log[-self._log_capacity:]
        for cb in self._subscribers:
            try:
                cb(msg)
            except Exception:
                pass
        return msg

    def subscribe(self, callback: Callable[[AgentMessage], None]) -> None:
        """Register a synchronous callback invoked on every published message."""
        self._subscribers.append(callback)

    # ── Read ─────────────────────────────────────────────────────────────────────
    def latest(self, topic: str, key: str) -> Optional[AgentMessage]:
        """Most recent message on a topic for a specific key (symbol/agent)."""
        return self._latest.get((topic, key))

    def latest_on_topic(self, topic: str) -> list[AgentMessage]:
        """All current latest-per-key messages on a topic."""
        return [m for (t, _k), m in self._latest.items() if t == topic]

    def signals_for(self, symbol: str, exclude_agent: str | None = None) -> list[AgentMessage]:
        """Every agent's latest SIGNAL for a symbol (optionally excluding one agent)."""
        out = []
        for (t, _k), m in self._latest.items():
            if t != TOPIC_SIGNAL or m.key != symbol:
                continue
            if exclude_agent and m.agent == exclude_agent:
                continue
            out.append(m)
        return out

    def peer_signal(
        self, symbol: str, agent: str,
    ) -> dict[str, AgentMessage]:
        """Map of peer-agent → their latest signal on `symbol` (excludes `agent`)."""
        return {m.agent: m for m in self.signals_for(symbol, exclude_agent=agent)}

    def exposures(self) -> list[AgentMessage]:
        return self.latest_on_topic(TOPIC_EXPOSURE)

    def recent(self, n: int = 50, topic: str | None = None) -> list[dict]:
        msgs = self._log if topic is None else [m for m in self._log if m.topic == topic]
        return [m.as_dict() for m in msgs[-n:]]

    def stats(self) -> dict[str, Any]:
        by_topic: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        for m in self._log:
            by_topic[m.topic] = by_topic.get(m.topic, 0) + 1
            by_agent[m.agent] = by_agent.get(m.agent, 0) + 1
        return {
            "messages_total": self._seq,
            "log_size":       len(self._log),
            "by_topic":       by_topic,
            "by_agent":       by_agent,
            "subscribers":    len(self._subscribers),
        }

    def clear(self) -> None:
        self._seq = 0
        self._log.clear()
        self._latest.clear()


# Singleton — imported everywhere agents need to talk
agent_bus = AgentBus()
