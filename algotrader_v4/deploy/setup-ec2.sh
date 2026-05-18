#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AlgoTrader Pro — one-shot bootstrap for a fresh Ubuntu 22.04 EC2 instance
#
# Usage (run as ubuntu / root after SSH):
#   curl -sO https://raw.githubusercontent.com/.../deploy/setup-ec2.sh
#   chmod +x setup-ec2.sh
#   sudo ./setup-ec2.sh yourdomain.com git@github.com:spbtextile/JAG.git
#
# Or after cloning the repo:
#   sudo bash algotrader_v4/deploy/setup-ec2.sh yourdomain.com
#
# Args:
#   $1 — domain (or EC2 public IP if no domain — TLS will be skipped)
#   $2 — git repo URL (optional, only needed for fresh clone)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DOMAIN="${1:-}"
REPO_URL="${2:-}"
APP_USER="algotrader"
APP_DIR="/opt/algotrader"
BRANCH="main"
PYTHON="python3.11"

# ── 0. Sanity ─────────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
[[ -n "$DOMAIN" ]] || { echo "Usage: $0 <domain-or-ip> [repo-url]"; exit 1; }

IS_DOMAIN=false
[[ "$DOMAIN" =~ ^[a-zA-Z].*\.[a-zA-Z]{2,}$ ]] && IS_DOMAIN=true

echo "▶ Domain/IP : $DOMAIN"
echo "▶ TLS       : $IS_DOMAIN"
echo "▶ App dir   : $APP_DIR"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends \
    git python3.11 python3.11-venv python3-pip \
    nginx ufw logrotate curl wget gnupg2 ca-certificates \
    build-essential libssl-dev libffi-dev python3.11-dev

$IS_DOMAIN && apt-get install -y --no-install-recommends certbot python3-certbot-nginx

# ── 2. App user ───────────────────────────────────────────────────────────────
id -u "$APP_USER" &>/dev/null || useradd -m -s /bin/bash "$APP_USER"

# ── 3. Clone / update repo ────────────────────────────────────────────────────
if [[ -d "$APP_DIR/.git" ]]; then
    echo "▶ Pulling latest…"
    sudo -u "$APP_USER" git -C "$APP_DIR" pull origin "$BRANCH"
elif [[ -n "$REPO_URL" ]]; then
    echo "▶ Cloning repo…"
    git clone --depth=1 -b "$BRANCH" "$REPO_URL" "$APP_DIR"
    chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
else
    # Running from inside the repo — symlink to /opt
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_DIR="$(dirname "$SCRIPT_DIR")"
    echo "▶ Linking $SRC_DIR → $APP_DIR"
    ln -sfn "$SRC_DIR" "$APP_DIR"
    chown -R "$APP_USER":"$APP_USER" "$(readlink -f "$APP_DIR")"
fi

# ── 4. Python venv + deps ─────────────────────────────────────────────────────
VENV="$APP_DIR/venv"
sudo -u "$APP_USER" $PYTHON -m venv "$VENV"
sudo -u "$APP_USER" "$VENV/bin/pip" install --quiet --upgrade pip wheel
sudo -u "$APP_USER" "$VENV/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

# ── 5. .env ───────────────────────────────────────────────────────────────────
ENV_FILE="$APP_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$APP_DIR/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    chown "$APP_USER":"$APP_USER" "$ENV_FILE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ACTION REQUIRED: fill in secrets"
    echo "  sudo nano $ENV_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "▶ .env already present — skipping"
fi

# ── 6. Log directory ──────────────────────────────────────────────────────────
mkdir -p /var/log/algotrader
chown "$APP_USER":"$APP_USER" /var/log/algotrader

# ── 7. systemd service ────────────────────────────────────────────────────────
cat > /etc/systemd/system/algotrader.service <<EOF
[Unit]
Description=AlgoTrader Pro
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --loop uvloop --http httptools --log-level warning --access-log --use-colors
Restart=always
RestartSec=5
StandardOutput=append:/var/log/algotrader/app.log
StandardError=append:/var/log/algotrader/app.log
# Give extra time for market-open startup
TimeoutStartSec=60
# Ensure clean shutdown (square-off triggers on SIGTERM)
TimeoutStopSec=30
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable algotrader

# ── 8. nginx ──────────────────────────────────────────────────────────────────
cat > /etc/nginx/sites-available/algotrader <<NGINX
upstream algotrader_app {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name $DOMAIN;

    # Health check bypasses auth (for load balancers / uptime monitors)
    location /health {
        proxy_pass http://algotrader_app;
    }

    location / {
        proxy_pass         http://algotrader_app;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;  # long-poll / WebSocket
    }
}
NGINX

ln -sfn /etc/nginx/sites-available/algotrader /etc/nginx/sites-enabled/algotrader
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# ── 9. TLS with Let's Encrypt ─────────────────────────────────────────────────
if $IS_DOMAIN; then
    echo "▶ Requesting TLS certificate for $DOMAIN…"
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
        --email "admin@$DOMAIN" --redirect
    systemctl enable certbot.timer
fi

# ── 10. UFW firewall ──────────────────────────────────────────────────────────
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
echo "▶ UFW rules applied"

# ── 11. Log rotation ──────────────────────────────────────────────────────────
cat > /etc/logrotate.d/algotrader <<LR
/var/log/algotrader/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LR

# ── 12. Install uvloop + httptools for faster event loop ──────────────────────
sudo -u "$APP_USER" "$VENV/bin/pip" install --quiet uvloop httptools 2>/dev/null || true

# ── 13. Start ─────────────────────────────────────────────────────────────────
systemctl start algotrader
sleep 3
systemctl is-active algotrader && echo "▶ algotrader.service is RUNNING" \
                                || { echo "✗ Service failed — check logs:"; journalctl -u algotrader -n 30; exit 1; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AlgoTrader Pro is live!"
$IS_DOMAIN && echo "  URL  : https://$DOMAIN" || echo "  URL  : http://$DOMAIN"
echo "  Login: https://$DOMAIN/login"
echo "  Logs : sudo journalctl -fu algotrader"
echo "         tail -f /var/log/algotrader/app.log"
echo ""
echo "  Next steps:"
echo "  1. Edit $ENV_FILE — set TRADING_MODE=LIVE and all secrets"
echo "  2. Set Kite redirect URL → https://$DOMAIN/auth/kite/callback"
echo "  3. sudo systemctl restart algotrader"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
