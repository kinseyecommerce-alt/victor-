"""
state_store.py — Durable JSON state with atomic writes.

The coordinator book and other runtime state live in memory; a process restart
loses them. This provides crash-safe persistence (write-temp-then-rename) under
logs/ so critical state survives a restart and can be reconciled against the
broker's actual positions on startup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

_DIR = Path("logs")


def _path(name: str) -> Path:
    return _DIR / name


def save_json(name: str, obj: Any) -> None:
    """Atomically persist `obj` as JSON to logs/<name> (temp file + rename)."""
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        p = _path(name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        os.replace(tmp, p)
    except Exception as exc:
        logger.warning("[state] save {} failed: {}", name, exc)


def load_json(name: str, default: Any = None) -> Any:
    p = _path(name)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("[state] load {} failed: {}", name, exc)
        return default


def clear(name: str) -> None:
    try:
        _path(name).unlink(missing_ok=True)
    except Exception:
        pass
