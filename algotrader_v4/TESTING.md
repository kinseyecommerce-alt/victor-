# Testing AlgoTrader Pro (core rebuild)

The app now runs four lab-validated strategies in PAPER only (REBUILD.md). The demo simulator, the AI trader and the
legacy tick agents were removed, so there is no "demo mode" any more: without Kite data the server starts and serves
the API, and the agents take no entries (`no_kite_data`).

| Level | What it proves | Needs |
|---|---|---|
| 1. Automated tests | Logic, lab parity, persistence, engine overlays, API | Python 3.11+ |
| 2. Local smoke start | The server starts without credentials and every endpoint answers | same |
| 3. Real-price paper | The strategies on real Kite prices, simulated orders | Kite Connect app + daily login |
| 4. Live | not possible for the kept strategies (they refuse LIVE) | — |

## 1. Automated tests

```bash
cd algotrader_v4
python3 test_pipeline.py            # full suite: lab parity of the agents, stratlab, paper book, SEBI, vault,
                                    # Kite login, auth, PEAD (test_validated_pead.py) and test_rebuild.py
python3 test_shadow_registry.py
python3 test_validated_pead.py
python3 test_option_bar_archiver.py
python3 test_rebuild.py             # old state files load; trading states, approvals, risk budget, drift, v2 API
```
Every run must end with zero failures. The agents' lab-parity tests are the acceptance gate for any change to an
agent, `validated_base.py`, `validated_data.py`, `pead_data.py` or the paper broker.

## 2. Local smoke start (no credentials)

```bash
cd algotrader_v4
TRADING_MODE=PAPER PERFORMANCE_FILE=/tmp/at/performance.json uvicorn main:app --port 8010
curl -s localhost:8010/health
curl -s localhost:8010/validated | head -c 300
curl -s localhost:8010/api/v2/overview
curl -s localhost:8010/api/v2/strategies | head -c 300
```
Without `API_KEY` the API is open (local only). With `API_KEY` set, `/api/v2/*` (except `/api/v2/health`) and every
write need `X-API-Key` or a Bearer JWT (`POST /auth/login`).

## 3. Real-price paper trading

`.env` / server environment:
```
TRADING_MODE=PAPER
PAPER_DATA_SOURCE=KITE
VALIDATED_AUCTION_VARIANT=2PCT
API_KEY=<long random>
JWT_SECRET_KEY=<long random>
```
Kite keys and the daily auto-login fields (user id, password, TOTP seed) go into the encrypted vault through
`POST /settings/credentials` (never into chat or files you share). The server logs in at 08:45 / 08:55 / 09:05 IST.

Each trading day the scheduler does everything (times in `platform_scheduler.py`). Watch:
* `GET /api/v2/overview` — Kite connected, market open, trading states, portfolio P&L and drawdown brake;
* `GET /api/v2/strategies` — per strategy: entry block reason, today's plan / positions / trades, P&L vs the lab,
  readiness progress, drift warnings;
* `GET /api/v2/system` — scheduled jobs and the price feed.

Operator controls: `POST /api/v2/trading-state` (REDUCE_ONLY / HALTED, global or per strategy),
`POST /bot/stop` (persists until `POST /bot/start`), `POST /api/v2/strategies/{name}/enable`, the SEBI kill switch
(`POST /sebi/kill-switch`).

## 4. Live

Not possible for the kept strategies: every validated agent refuses LIVE. `GET /api/v2/readiness` lists the LIVE
prerequisites (static outbound IP registered with Kite in `KITE_STATIC_IP` — required for API orders since
1 April 2026 —, API key, JWT secret, kill-switch reset secret, Kite keys); it reports `ready_for_live: false` while
no strategy is live-capable.

## Deploying

Railway builds `deploy/Dockerfile` (root directory `algotrader_v4`); persistent state lives on the volume under
`/data` (ledger, paper book, agent enables, trading states, approvals, shadow ledgers, PEAD store, option-bar
archive, vault). `deploy/AWS.md` describes a Mumbai EC2 host with a fixed IP (the static-IP route for a future LIVE).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Every agent shows `no_kite_data` | `PAPER_DATA_SOURCE=KITE` and a Kite session for today (`GET /auth/kite/status`) |
| `stopped` | the operator Stop is persisted: `POST /bot/start` |
| `trading_state:…` | `GET /api/v2/trading-state`; a brake level needs `POST /api/v2/risk/brake/reset` |
| `awaiting_approval` | semi-auto mode is on: approve today's ticket (`GET/POST /api/v2/approvals…`) or turn the mode off |
| `no_data:store_stale` | the session store missed a day; it refreshes at 08:55, at restart and at 15:40 |
