"""
master_agent.py  — AlgoTrader Pro v5
Fully autonomous master agent — see master_agent_full.py for implementation.
This file re-exports the singleton for backward compatibility.
"""
from master_agent_v5 import MasterAgent, master_agent, MASTER_PROMPT
__all__ = ["MasterAgent", "master_agent", "MASTER_PROMPT"]
