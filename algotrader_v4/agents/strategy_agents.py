"""
agents/strategy_agents.py — the registry of the kept strategies (ALL_AGENTS).

Every kept strategy is a lab-validated, PAPER-only agent driven by platform_scheduler jobs (scheduled = True);
the legacy tick agents were removed in the core rebuild (REBUILD.md). The name of this module and of ALL_AGENTS
are kept: the paper book, the shadow registry, the scheduler and the tests address the agents through it.

    auc_fade    AUC-FADE-3PCT / -2PCT (VALIDATED_AUCTION_VARIANT)   candidate (not executable as tested)
    attn_on     SWING-ATTN-ON-3X                                    candidate (negative at CNC costs)
    vrp_winged  VRP-STRANGLE-016-WINGED                             candidate (model-priced)
    pead_h60    EV-PEAD-LONG-H60 (VALIDATED_PEAD_ENABLED, default off) candidate (DSR 0.51)
    vrp_naked   VRP-STRANGLE-016-NAKED (VALIDATED_STRANGLE_NAKED_ENABLED, default off) candidate (unlimited risk)
"""
from __future__ import annotations

from ist_clock import now_ist  # noqa: F401  (module-level clock some tests patch)
from agents.base_agent import BaseAgent
from agents.validated_auction import AuctionFadeAgent
from agents.validated_attention import AttentionOvernightAgent
from agents.validated_strangle import WingedStrangleAgent, NakedStrangleAgent
from agents.validated_pead import PeadCandidateAgent

ALL_AGENTS: dict[str, BaseAgent] = {
    "auc_fade": AuctionFadeAgent(),
    "attn_on": AttentionOvernightAgent(),
    "vrp_winged": WingedStrangleAgent(),
    "pead_h60": PeadCandidateAgent(),
    "vrp_naked": NakedStrangleAgent(),       # VALIDATED_STRANGLE_NAKED_ENABLED (default off)
}

# ONE definition of "proven" (22 Nov 2026), used by the app, the API and the dashboard. A strategy is proven only if
# ALL three hold:
#   1. luck test: Deflated Sharpe >= 0.95 at the CURRENT trial count N (the builder's N, which never shrinks);
#   2. executable as tested: the tested entry and exit can be placed with real orders at the tested moments;
#   3. positive after the costs of the instrument actually held (and at the 5 bps stress).
# Everything else is a candidate: paper only. Before 22 Nov "proven" meant "passed the pre-registered protocol in
# both eras", which marked auc_fade and attn_on proven; the lab README and the pinned forward-lab rule keep their own
# older wording (pinned files are never rewritten).
PROVEN = {"auc_fade": False, "attn_on": False, "vrp_winged": False, "pead_h60": False, "vrp_naked": False}
WHY_NOT_PROVEN = {
    "auc_fade": "signal and entry use the final 09:15 auction price, which is not known while pre-open orders can "
                "be placed (not executable as tested); DSR below 0.95 at the current N",
    "attn_on": "held as a delivery (CNC) position, and at CNC costs the lab result is negative on FINAL "
               "(-7.5 bps/trade); DSR below 0.95 at the current N",
    "vrp_winged": "passed only on modelled option prices; re-priced on real NSE closes (stratlab/STRANGLE_REALPRICE.md) "
                  "the real credit is 0.755 of the model (wings cost 1.73x) and the lab record turns from +Rs 1.95 L to "
                  "about -Rs 0.97 L (PF 0.88); fails its own 80 % credit bar",
    "pead_h60": "DSR 0.51 at the fair trial count N = 9 (needs 0.95); excess over NIFTY since 2018 not significant",
    "vrp_naked": "positive on real NSE closes (credit 0.898 of the model, PF 1.22) but the luck test at the current N "
                 "is not met (2023-26 t of the model result 1.49), and it carries unlimited risk (no wings)",
}
