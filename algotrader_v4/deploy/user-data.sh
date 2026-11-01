#!/usr/bin/env bash
# EC2 "user data": runs once, as root, when the instance first boots. Clones the repo and runs
# setup-ec2.sh against the instance's own public IP (pass a domain instead by editing HOST).
# Progress: /var/log/algotrader-setup.log ; finished when /root/algotrader-first-login.txt exists.
set -euo pipefail
exec > >(tee -a /var/log/algotrader-setup.log) 2>&1

REPO_URL="${REPO_URL:-https://github.com/kinseyecommerce-alt/victor-.git}"
BRANCH="${BRANCH:-}"
HOST="${HOST:-}"
if [[ -z "$HOST" ]]; then
    TOKEN="$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')"
    HOST="$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)"
fi
apt-get update -qq && apt-get install -y -qq git
rm -rf /tmp/algotrader-src
git clone --quiet ${BRANCH:+--branch "$BRANCH"} "$REPO_URL" /tmp/algotrader-src
bash /tmp/algotrader-src/algotrader_v4/deploy/setup-ec2.sh "$HOST" "$REPO_URL" "$BRANCH"
