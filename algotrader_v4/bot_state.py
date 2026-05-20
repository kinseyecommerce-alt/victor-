"""
bot_state.py — shared mutable runtime state (no circular imports).
Both main.py and master_agent_v5.py import from here.
"""

_agent_enabled: dict[str, bool] = {
    "intraday": True,
    "fno":      True,
    "swing":    True,
    "scalping": True,
}


def is_agent_enabled(name: str) -> bool:
    return _agent_enabled.get(name, True)


def set_agent_enabled(name: str, val: bool) -> None:
    if name in _agent_enabled:
        _agent_enabled[name] = val
