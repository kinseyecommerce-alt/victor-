---
name: research
description: >
  RETIRED (core rebuild, Nov 2026) — do not use. The weekly AI research loop, the walk-forward optimiser and
  their endpoints (/research/*, /optimize/*, /backtest/replay) were removed together with the LLM
  integrations; the lab rules may not be tuned from live results. Kept for history only.
---

> **RETIRED in the core rebuild (Nov 2026, `algotrader_v4/REBUILD.md`).** Do not use. The weekly AI research loop, the walk-forward optimiser and their endpoints (/research/*, /optimize/*, /backtest/replay) were removed together with the LLM integrations; the lab rules may not be tuned from live results. Kept for history only.
>
> Instead: live-vs-lab monitoring is `GET /api/v2/drift` (warnings only) and the shadow registry (`GET /api/v2/shadow-registry`, fixed pre-registered promotion rule); new ideas go through stratlab's pre-registered protocol, not through parameter changes on the running app.
>
> Everything below is the original text, kept for history; its commands no longer work.

# Research loop — AlgoTrader Pro

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j()    { curl -sS -m 30 -H "$H" "$ALGO_URL$1"; }
post() { curl -sS -m 900 -H "$H" -H "Content-Type: application/json" -X POST "$ALGO_URL$1" -d "${2:-{}}"; }
```

| Request | Call |
|---|---|
| What would the analyst see? | `j /research/evidence` |
| Run a research cycle now (Claude proposal → replay A/B → adopt/reject; costs one API call + ~1 min) | `post /research/run '["RELIANCE","TCS","INFY"]'` — ask before running, it can change live parameters |
| What happened in past cycles | `j "/research/log?n=10"` |
| Walk-forward optimiser (background, ~5 min) | `post /optimize/run '{"days":45}'` then `j /optimize/status` |
| Current / saved parameters | `j /optimize/status` (`saved`) |
| Undo a change | `post /optimize/params/intraday/reset` |

## How to report

- **Adopted:** name the agent, parameter, old → new value, and the replay numbers (net ₹ current vs
  proposed, trades, drawdown). Say it came from a replay of N days on which symbols.
- **Rejected / no change:** quote the verdict and the proposal's rationale. A rejection is a good
  outcome — it means a plausible-sounding idea didn't survive the data.
- Always mention the data source line (`real` vs `synthetic (paper simulator)`): synthetic results
  prove the machinery, not the strategy.
- Never adopt anything by hand that the A/B test rejected; if the user insists, say what the
  numbers showed and that they are overriding evidence.
