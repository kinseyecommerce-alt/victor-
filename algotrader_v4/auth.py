"""
auth.py — JWT app login + Kite OAuth helpers
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from passlib.context import CryptContext

from config import settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

ALGORITHM = "HS256"


# ── Password helpers ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)

def authenticate(username: str, password: str) -> bool:
    if username != settings.admin_username:
        return False
    if settings.admin_password_hash:
        return verify_password(password, settings.admin_password_hash)
    # fallback for dev: plain password comparison
    return password == settings.admin_password


# ── JWT ───────────────────────────────────────────────────────────

def create_token(username: str) -> tuple[str, int]:
    """Returns (access_token, expires_in_seconds)."""
    expires = timedelta(hours=settings.jwt_expire_hours)
    expire_dt = datetime.utcnow() + expires
    token = jwt.encode(
        {"sub": username, "exp": expire_dt},
        settings.jwt_secret_key,
        algorithm=ALGORITHM,
    )
    return token, int(expires.total_seconds())

def decode_token(token: str) -> Optional[str]:
    """Returns username if token valid, else None."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ── FastAPI dependency ────────────────────────────────────────────

async def current_user(token: str = Depends(_oauth2)) -> str:
    """Returns username; raises 401 if token invalid."""
    username = decode_token(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return username

async def optional_user(token: str = Depends(_oauth2)) -> Optional[str]:
    """Returns username or None — never raises."""
    return decode_token(token) if token else None
