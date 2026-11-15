"""
agents/strategy_agents.py — the registry of the kept strategies (ALL_AGENTS).

Every kept strategy is a lab-validated, PAPER-only agent driven by platform_scheduler jobs (scheduled = True);
the legacy tick agents were removed in the core rebuild (REBUILD.md). The name of this module and of ALL_AGENTS
are kept: the paper book, the shadow registry, the scheduler and the tests address the agents through it.

    auc_fade    AUC-FADE-3PCT / -2PCT (VALIDATED_AUCTION_VARIANT)   proven
    attn_on     SWING-ATTN-ON-3X                                    proven
    vrp_winged  VRP-STRANGLE-016-WINGED                             candidate
    pead_h60    EV-PEAD-LONG-H60 (VALIDATED_PEAD_ENABLED, default off) candidate
"""
from __future__ import annotations

from ist_clock import now_ist  # noqa: F401  (module-level clock some tests patch)
from agents.base_agent import BaseAgent
from agents.validated_auction import AuctionFadeAgent
from agents.validated_attention import AttentionOvernightAgent
from agents.validated_strangle import WingedStrangleAgent
from agents.validated_pead import PeadCandidateAgent

ALL_AGENTS: dict[str, BaseAgent] = {
    "auc_fade": AuctionFadeAgent(),
    "attn_on": AttentionOvernightAgent(),
    "vrp_winged": WingedStrangleAgent(),
    "pead_h60": PeadCandidateAgent(),
}

# Proven = passed the pre-registered protocol in both eras (proven_status in the v2 API); candidate = paper only.
PROVEN = {"auc_fade": True, "attn_on": True, "vrp_winged": False, "pead_h60": False}
