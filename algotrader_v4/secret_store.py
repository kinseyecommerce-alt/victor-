"""
secret_store.py — encrypted on-disk vault for API credentials.

Keys entered in the dashboard (Settings → API Keys) are saved here so they survive a restart
without living in plain text in .env:

  .secrets/vault.enc    Fernet-encrypted JSON (AES-128-CBC + HMAC-SHA256)
  .secrets/master.key   the encryption key, created on first save (chmod 600)

For stronger separation set SECRETS_MASTER_KEY (a Fernet key) in the environment instead of
keeping master.key on the same disk:  python3 secret_store.py --new-key

Stored values are never returned by the API — only "set / not set" and the last 4 characters.
The Kite access token is stored with its IST date and ignored the next day (Kite expires it).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_IST = timezone(timedelta(hours=5, minutes=30))

# Only these settings fields can be stored. Anything else is rejected.
FIELDS = (
    "kite_api_key", "kite_api_secret", "kite_access_token",
    "anthropic_api_key", "truedata_username", "truedata_password",
)
_DAILY = {"kite_access_token"}          # valid only on the IST day they were saved


def _today_ist() -> str:
    return datetime.now(_IST).date().isoformat()


class SecretStore:
    def __init__(self, directory: str | os.PathLike | None = None) -> None:
        base = directory or os.getenv("SECRETS_DIR") or Path(__file__).resolve().parent / ".secrets"
        self.dir = Path(base)
        self.vault = self.dir / "vault.enc"
        self.keyfile = self.dir / "master.key"
        self._lock = threading.Lock()

    # ── key handling ────────────────────────────────────────────────────────
    def _fernet(self, create: bool) -> Fernet | None:
        env_key = os.getenv("SECRETS_MASTER_KEY", "").strip()
        if env_key:
            return Fernet(env_key.encode())
        if self.keyfile.exists():
            return Fernet(self.keyfile.read_bytes().strip())
        if not create:
            return None
        self._ensure_dir()
        key = Fernet.generate_key()
        fd = os.open(self.keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        return Fernet(key)

    def _ensure_dir(self) -> None:
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)

    @property
    def key_source(self) -> str:
        if os.getenv("SECRETS_MASTER_KEY", "").strip():
            return "env"
        return "file" if self.keyfile.exists() else "none"

    # ── vault I/O ───────────────────────────────────────────────────────────
    def _read(self) -> dict:
        if not self.vault.exists():
            return {}
        f = self._fernet(create=False)
        if f is None:
            raise RuntimeError("Vault exists but no master key (set SECRETS_MASTER_KEY or restore master.key)")
        try:
            return json.loads(f.decrypt(self.vault.read_bytes()))
        except InvalidToken:
            raise RuntimeError("Vault cannot be decrypted with the current master key") from None

    def _write(self, data: dict) -> None:
        f = self._fernet(create=True)
        self._ensure_dir()
        blob = f.encrypt(json.dumps(data).encode())
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".vault.")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, self.vault)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── public API ──────────────────────────────────────────────────────────
    def load(self) -> dict[str, str]:
        """Usable values: field → secret. Expired daily values are left out."""
        with self._lock:
            raw = self._read()
        out: dict[str, str] = {}
        for name, entry in raw.items():
            if name not in FIELDS or not isinstance(entry, dict) or not entry.get("value"):
                continue
            if name in _DAILY and entry.get("day") != _today_ist():
                continue
            out[name] = entry["value"]
        return out

    def save(self, values: dict[str, str]) -> None:
        bad = set(values) - set(FIELDS)
        if bad:
            raise ValueError(f"Not storable: {sorted(bad)}")
        with self._lock:
            data = self._read()
            now = datetime.now(_IST).isoformat(timespec="seconds")
            for name, value in values.items():
                data[name] = {"value": value, "saved_at": now, "day": _today_ist()}
            self._write(data)

    def delete(self, names: list[str] | tuple[str, ...]) -> None:
        with self._lock:
            data = self._read()
            for n in names:
                data.pop(n, None)
            if data:
                self._write(data)
            else:
                self.vault.unlink(missing_ok=True)

    def status(self) -> dict:
        """Safe to return over the API: no secret values."""
        try:
            with self._lock:
                raw = self._read()
            error = None
        except RuntimeError as e:
            raw, error = {}, str(e)
        fields = {}
        for name in FIELDS:
            e = raw.get(name)
            if not isinstance(e, dict) or not e.get("value"):
                fields[name] = {"stored": False}
                continue
            expired = name in _DAILY and e.get("day") != _today_ist()
            fields[name] = {"stored": True, "hint": "…" + e["value"][-4:] if len(e["value"]) > 8 else "set",
                            "saved_at": e.get("saved_at"), "expired": expired}
        return {"vault": str(self.vault), "key_source": self.key_source, "error": error, "fields": fields}


def apply_to(settings, store: "SecretStore | None" = None) -> list[str]:
    """Copy stored values onto the settings object. Stored values win over .env."""
    try:
        values = (store or secret_store).load()
    except Exception as e:          # a broken vault must not stop the server starting
        print(f"[secret_store] vault not loaded: {e}", file=sys.stderr)
        return []
    for name, value in values.items():
        setattr(settings, name, value)
    return sorted(values)


secret_store = SecretStore()


if __name__ == "__main__":
    if "--new-key" in sys.argv:
        print(Fernet.generate_key().decode())
        print("Put this in the server environment as SECRETS_MASTER_KEY (not in the repo).",
              "Existing vault entries must be re-entered after switching keys.", file=sys.stderr)
    else:
        print(json.dumps(secret_store.status(), indent=2))
