# Deploying AlgoTrader Pro on AWS (Mumbai)

One small server in Mumbai is all it needs. Expect about **$18–20 / month** (t3.small, 30 GB
disk, Elastic IP). Everything below is done once; updates afterwards are one command.

## What you get

```
Internet ──443──▶ nginx (HTTPS) ──▶ uvicorn :8000 (FastAPI + React dashboard)
                                        ├── tick engine, agents, AI trader
                                        ├── .secrets/vault.enc  (your keys, encrypted)
                                        └── data/               (ledger, swing positions)
systemd restarts the app if it crashes or the server reboots.
```

## 1. Create the server

In the AWS console (region **Asia Pacific (Mumbai) `ap-south-1`**) → EC2 → Launch instance:

| Setting | Value |
|---|---|
| AMI | Ubuntu Server 24.04 LTS (64-bit x86) |
| Instance type | `t3.small` (2 vCPU, 2 GB) |
| Key pair | create one, download the `.pem` — you need it for SSH |
| Network → security group | allow **SSH (22)** from *My IP*, **HTTPS (443)** from *My IP*, **HTTP (80)** from anywhere (only needed for Let's Encrypt if you use a domain) |
| Storage | 30 GB gp3 |
| Advanced → User data | paste the contents of `deploy/user-data.sh` |

Then **Elastic IPs → Allocate → Associate** with the instance. This fixed IP is what you
whitelist at Zerodha; without it the IP changes on every reboot.

The user-data script installs everything on first boot (5–8 minutes). Watch it:

```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
sudo tail -f /var/log/algotrader-setup.log        # done when it prints "AlgoTrader Pro is running"
sudo cat /root/algotrader-first-login.txt         # dashboard password + API key
```

Using a domain instead of the bare IP? Point an `A` record at the Elastic IP first, then run
`sudo bash /opt/algotrader/deploy/setup-ec2.sh trade.yourdomain.com` — you get a real
certificate. With the bare IP the certificate is self-signed: the browser warns once, accept it.

Already have a server? Skip user data and run:
`curl -fsSL https://raw.githubusercontent.com/kinseyecommerce-alt/victor-/HEAD/algotrader_v4/deploy/setup-ec2.sh | sudo bash -s -- <ip-or-domain>`

## 2. Connect Zerodha (once)

1. [developers.kite.trade](https://developers.kite.trade) → your app → **Redirect URL**:
   `https://<elastic-ip-or-domain>/auth/kite/callback`
2. Same page → whitelist the Elastic IP (required for order placement).
3. Open `https://<elastic-ip>/`, log in with the admin password from the first-login file.
4. Trading view → ⚙ Settings → **Connection**: paste the API key from the same file.
5. ⚙ Settings → **API Keys**: paste the Kite API key + secret and your Anthropic key, Save.
   They are stored encrypted on the server (`SECRETS_MASTER_KEY` lives only in `/opt/algotrader/.env`).
6. **Log in with Zerodha** — every trading morning before 09:15.

The server starts in **PAPER mode on real Kite prices**: real quotes, simulated orders.
Run it that way for 2–4 weeks and judge it on the dashboard's "Are we beating the index?" panel.

## 3. Day to day

| Task | Command (over SSH) |
|---|---|
| Update to the latest code (tests must pass or it won't restart) | `sudo bash /opt/algotrader/deploy/update.sh` |
| Logs | `tail -f /var/log/algotrader/app.log` |
| Restart / stop / status | `sudo systemctl restart algotrader` · `stop` · `status` |
| Change a setting | `sudo nano /opt/algotrader/.env` then restart |
| Kill switch | dashboard → Trading view → SEBI kill switch, or stop the service |

Telegram alerts: add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env` and restart.

## 4. Going live (only after paper trading is profitable after costs)

```
TRADING_MODE=LIVE
AI_AGENT_ALLOW_LIVE=true          # otherwise the AI stays read-only in LIVE
AI_AGENT_MAX_POSITION_VALUE=10000 # start tiny
AI_AGENT_MAX_TRADES_PER_DAY=2
MAX_DAILY_LOSS=1000
```
`sudo systemctl restart algotrader`. Keep `AI_AGENT_FUTURES_ENABLED` and
`AI_AGENT_OPTION_SELLING_ENABLED` off until you've watched the simpler trades for weeks.

## 5. Backups and security

- Back up `/opt/algotrader/.env` (contains the vault's master key) **separately** from
  `/opt/algotrader/.secrets/` and `/opt/algotrader/data/`. Either alone is useless to a thief.
- Never open port 8000 in the security group; nginx is the only entrance.
- `sudo apt update && sudo apt upgrade -y` monthly; reboot afterwards (the app comes back by itself).
- Rotate the Kite API secret if the server is ever compromised; the daily Kite token expires on its own.

## Troubleshooting

| Symptom | Check |
|---|---|
| Setup log stops with an error | `sudo bash /opt/algotrader/deploy/setup-ec2.sh <ip>` re-runs safely |
| Dashboard "Cannot reach the backend" | `systemctl status algotrader`; `tail /var/log/algotrader/app.log` |
| Kite login fails | redirect URL must match exactly, including `https://` and the path |
| No prices after Kite login | the daily token: log in again from Settings → API Keys |
| Orders rejected in LIVE | Elastic IP whitelisted at Kite? Margin available? |
