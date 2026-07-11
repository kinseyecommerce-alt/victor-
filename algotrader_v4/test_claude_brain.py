"""
test_claude_brain.py — Claude brain (central agent intelligence) tests
Run: cd algotrader_v4 && python test_claude_brain.py

Covers claude_brain.py end-to-end with a MOCKED Anthropic client (no real API
key / network in the sandbox):
  1. graceful offline degradation → rule-based posture mirrors the regime plan
  2. is_enabled() gating on key + flag
  3. response coercion / clamping (size_factor ladder, invalid action, alert)
  4. text extraction (skips thinking blocks) + tolerant JSON parsing
  5. assess() happy path with a mocked AsyncAnthropic → source="claude"
  6. assess() timeout + error → fallback posture (never raises)
  7. status()/log ring buffer
  8. master_agent integration: posture published to the bus + directives applied
"""
from __future__ import annotations

import asyncio
import json
import types

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
def ok(n):   _results.append((n, True, "")); print(f"  OK  {n}")
def fail(n, e): _results.append((n, False, e)); print(f"  XX  {n}: {e}")
def run(n, fn):
    try: fn(); ok(n)
    except Exception as exc: fail(n, str(exc)[:180])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def arun(coro): return asyncio.get_event_loop().run_until_complete(coro)

import claude_brain
from claude_brain import BrainPosture
from config import settings, Settings


# ── Fake Anthropic client (shape-compatible with AsyncAnthropic.messages) ──────

class _Block:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text

class _Resp:
    def __init__(self, content):
        self.content = content

class _FakeMessages:
    def __init__(self, behaviour):
        self._behaviour = behaviour   # callable(**kwargs) -> _Resp | raises
    async def create(self, **kwargs):
        return self._behaviour(**kwargs)

class _FakeClient:
    def __init__(self, behaviour):
        self.messages = _FakeMessages(behaviour)

def _install_client(behaviour):
    claude_brain._client = _FakeClient(behaviour)

def _reset_client():
    claude_brain._client = None


_VALID_JSON = {
    "regime": "trending_up",
    "regime_confidence": 82,
    "size_factor": 0.75,
    "agent_directives": {
        "intraday": {"action": "run", "reason": "clean uptrend"},
        "scalping": {"action": "run", "reason": "momentum"},
        "swing":    {"action": "reduce_size", "reason": "overnight gap risk"},
        "fno":      {"action": "pause", "reason": "IV crush"},
    },
    "halt_new_trades": False,
    "opportunity_alert": "Crude breaking out above range high",
    "reasoning": "Energy complex trending; ride intraday, trim overnight.",
}

def _snapshot(daily_pnl=0.0, size_factor=1.0, paused=None, regime="trending_up"):
    return {
        "regime": regime,
        "regime_plan": {
            "active": ["intraday", "scalping"],
            "paused": paused or [],
            "allocation": {},
            "size_factor": size_factor,
            "reasoning": f"plan for {regime}",
        },
        "risk": {"daily_pnl": daily_pnl},
    }


# ═══════════════════════════════════════════════════════════════════════════════
section("1. OFFLINE / RULE-BASED DEGRADATION")

def t_offline_no_key_rule_based():
    s = Settings(); s.anthropic_api_key = ""; s.use_claude_brain = True
    claude_brain.settings = s
    try:
        p = arun(claude_brain.assess(_snapshot(size_factor=1.0)))
        assert p.source == "rule_based", p.source
        assert p.size_factor == 1.0, p.size_factor
        assert p.agent_directives["intraday"]["action"] == "run"
    finally:
        claude_brain.settings = settings

def t_rule_based_mirrors_paused():
    p = claude_brain._rule_based_posture(_snapshot(paused=["fno"], size_factor=0.5))
    assert p.agent_directives["fno"]["action"] == "pause", p.agent_directives["fno"]
    # size_factor < 0.75 → non-paused agents reduce_size
    assert p.agent_directives["intraday"]["action"] == "reduce_size"

def t_rule_based_halt_on_deep_loss():
    s = Settings(); s.max_daily_loss = 10000.0
    claude_brain.settings = s
    try:
        p = claude_brain._rule_based_posture(_snapshot(daily_pnl=-6000.0))
        assert p.halt_new_trades is True, p.halt_new_trades
        p2 = claude_brain._rule_based_posture(_snapshot(daily_pnl=-1000.0))
        assert p2.halt_new_trades is False
    finally:
        claude_brain.settings = settings

run("offline (no key) → rule-based posture", t_offline_no_key_rule_based)
run("rule-based mirrors paused/reduced plan", t_rule_based_mirrors_paused)
run("rule-based halts on deep daily loss",   t_rule_based_halt_on_deep_loss)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. is_enabled() GATING")

def t_enabled_requires_key_and_flag():
    s = Settings()
    claude_brain.settings = s
    try:
        s.anthropic_api_key = ""; s.use_claude_brain = True
        assert claude_brain.is_enabled() is False
        s.anthropic_api_key = "sk-test"; s.use_claude_brain = False
        assert claude_brain.is_enabled() is False
        s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
        assert claude_brain.is_enabled() is True
    finally:
        claude_brain.settings = settings

run("is_enabled requires key AND flag", t_enabled_requires_key_and_flag)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. RESPONSE COERCION / CLAMPING")

def t_coerce_snaps_size_factor():
    p = claude_brain._coerce_posture({"size_factor": 5.0}, 10)
    assert p.size_factor == 1.0, p.size_factor
    p2 = claude_brain._coerce_posture({"size_factor": 0.01}, 10)
    assert p2.size_factor == 0.25, p2.size_factor

def t_coerce_invalid_action_defaults_run():
    p = claude_brain._coerce_posture(
        {"agent_directives": {"intraday": {"action": "YOLO"}}}, 5)
    assert p.agent_directives["intraday"]["action"] == "run"
    # all four keys always present
    assert set(p.agent_directives.keys()) == set(claude_brain.AGENT_KEYS)

def t_coerce_alert_null_normalised():
    p = claude_brain._coerce_posture({"opportunity_alert": "null"}, 5)
    assert p.opportunity_alert is None
    p2 = claude_brain._coerce_posture({"opportunity_alert": "Real alert"}, 5)
    assert p2.opportunity_alert == "Real alert"

def t_coerce_source_and_confidence():
    p = claude_brain._coerce_posture({"regime_confidence": 999}, 5)
    assert p.regime_confidence == 100
    assert p.source == "claude"

run("size_factor snapped to [0.25,1.0]",     t_coerce_snaps_size_factor)
run("invalid action → run; all keys present", t_coerce_invalid_action_defaults_run)
run("opportunity_alert 'null' normalised",   t_coerce_alert_null_normalised)
run("confidence clamped; source=claude",     t_coerce_source_and_confidence)


# ═══════════════════════════════════════════════════════════════════════════════
section("4. TEXT EXTRACTION + JSON PARSING")

def t_extract_skips_thinking():
    resp = _Resp([_Block("thinking", "internal musing"), _Block("text", '{"a":1}')])
    assert claude_brain._extract_text(resp) == '{"a":1}'

def t_parse_plain_json():
    assert claude_brain._parse_json('{"x": 2}') == {"x": 2}

def t_parse_code_fenced():
    assert claude_brain._parse_json('```json\n{"x": 3}\n```') == {"x": 3}

def t_parse_embedded_object():
    assert claude_brain._parse_json('here you go: {"x": 4} thanks')["x"] == 4

run("extract text skips thinking blocks", t_extract_skips_thinking)
run("parse plain JSON",                   t_parse_plain_json)
run("parse code-fenced JSON",             t_parse_code_fenced)
run("parse embedded JSON object",         t_parse_embedded_object)


# ═══════════════════════════════════════════════════════════════════════════════
section("5. assess() HAPPY PATH (mocked client)")

def t_assess_happy_path():
    s = Settings(); s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
    s.claude_brain_timeout_sec = 5.0
    claude_brain.settings = s
    captured = {}
    def behaviour(**kwargs):
        captured.update(kwargs)
        return _Resp([_Block("text", json.dumps(_VALID_JSON))])
    _install_client(behaviour)
    try:
        p = arun(claude_brain.assess(_snapshot()))
        assert p.source == "claude", p.source
        assert p.regime == "trending_up"
        assert p.size_factor == 0.75
        assert p.agent_directives["fno"]["action"] == "pause"
        assert p.latency_ms >= 0
        # correct model + adaptive thinking were requested
        assert captured["model"] == "claude-opus-4-8", captured.get("model")
        assert captured["thinking"] == {"type": "adaptive"}, captured.get("thinking")
    finally:
        claude_brain.settings = settings
        _reset_client()

def t_assess_disabled_short_circuits_no_call():
    s = Settings(); s.anthropic_api_key = ""; s.use_claude_brain = True
    claude_brain.settings = s
    called = {"n": 0}
    def behaviour(**kwargs):
        called["n"] += 1
        return _Resp([_Block("text", "{}")])
    _install_client(behaviour)
    try:
        p = arun(claude_brain.assess(_snapshot()))
        assert called["n"] == 0, "client must not be called when disabled"
        assert p.source == "rule_based"
    finally:
        claude_brain.settings = settings
        _reset_client()

run("assess() happy path → source=claude", t_assess_happy_path)
run("assess() disabled → no client call",  t_assess_disabled_short_circuits_no_call)


# ═══════════════════════════════════════════════════════════════════════════════
section("6. assess() FAILURE MODES → FALLBACK")

def t_assess_timeout_falls_back():
    s = Settings(); s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
    s.claude_brain_timeout_sec = 0.05
    claude_brain.settings = s
    async def slow_create(**kwargs):
        await asyncio.sleep(1.0)
        return _Resp([_Block("text", "{}")])
    claude_brain._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=slow_create))
    try:
        p = arun(claude_brain.assess(_snapshot(size_factor=0.5, paused=["swing"])))
        assert p.source == "fallback", p.source
        assert p.agent_directives["swing"]["action"] == "pause"
    finally:
        claude_brain.settings = settings
        _reset_client()

def t_assess_error_falls_back():
    s = Settings(); s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
    s.claude_brain_timeout_sec = 5.0
    claude_brain.settings = s
    def behaviour(**kwargs):
        raise RuntimeError("boom")
    _install_client(behaviour)
    try:
        p = arun(claude_brain.assess(_snapshot()))
        assert p.source == "fallback", p.source
        assert isinstance(p, BrainPosture)
    finally:
        claude_brain.settings = settings
        _reset_client()

def t_assess_bad_json_falls_back():
    s = Settings(); s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
    s.claude_brain_timeout_sec = 5.0
    claude_brain.settings = s
    def behaviour(**kwargs):
        return _Resp([_Block("text", "not json at all")])
    _install_client(behaviour)
    try:
        p = arun(claude_brain.assess(_snapshot()))
        assert p.source == "fallback", p.source
    finally:
        claude_brain.settings = settings
        _reset_client()

run("assess() timeout → fallback",   t_assess_timeout_falls_back)
run("assess() API error → fallback", t_assess_error_falls_back)
run("assess() bad JSON → fallback",  t_assess_bad_json_falls_back)


# ═══════════════════════════════════════════════════════════════════════════════
section("7. STATUS + LOG RING BUFFER")

def t_status_shape():
    st = claude_brain.status()
    for k in ("enabled", "has_api_key", "flag_on", "model", "assessments"):
        assert k in st, k

def t_log_records_claude_postures():
    claude_brain._brain_log.clear()
    s = Settings(); s.anthropic_api_key = "sk-test"; s.use_claude_brain = True
    s.claude_brain_timeout_sec = 5.0
    claude_brain.settings = s
    def behaviour(**kwargs):
        return _Resp([_Block("text", json.dumps(_VALID_JSON))])
    _install_client(behaviour)
    try:
        arun(claude_brain.assess(_snapshot()))
        log = claude_brain.get_brain_log(5)
        assert len(log) == 1 and log[0]["source"] == "claude"
    finally:
        claude_brain.settings = settings
        _reset_client()
        claude_brain._brain_log.clear()

run("status() shape",                 t_status_shape)
run("log records claude postures",    t_log_records_claude_postures)


# ═══════════════════════════════════════════════════════════════════════════════
section("8. MASTER AGENT INTEGRATION")

def t_master_publishes_posture_to_bus():
    import master_agent_v5 as mav
    from agent_bus import agent_bus, TOPIC_REGIME
    from market_regime import regime_detector

    agent_bus.clear()
    ma = mav.MasterAgent()
    ma.running = True

    # Stub the regime detector so no network/data is needed.
    plan = regime_detector.current_plan
    async def fake_update():
        return regime_detector.current_regime, plan
    orig_update = regime_detector.update
    regime_detector.update = fake_update

    # Stub the brain so no API key is needed and we get a deterministic posture.
    async def fake_assess(snapshot):
        return BrainPosture(
            regime="volatile", regime_confidence=70, size_factor=0.5,
            agent_directives={k: {"action": "reduce_size", "reason": "vol"}
                              for k in claude_brain.AGENT_KEYS},
            halt_new_trades=False, reasoning="volatile — trim size", source="claude")
    orig_assess = claude_brain.assess
    claude_brain.assess = fake_assess

    try:
        arun(ma._master_review())
        msg = agent_bus.latest(TOPIC_REGIME, "regime")
        assert msg is not None, "no regime posture published"
        assert msg.payload["size_factor"] == 0.5, msg.payload
        assert msg.payload["regime"] == "volatile"
        assert ma.last_posture["source"] == "claude"
        assert ma.last_directives["risk_override"]["halt_new_trades"] is False
    finally:
        regime_detector.update = orig_update
        claude_brain.assess = orig_assess
        agent_bus.clear()

run("master publishes brain posture to bus", t_master_publishes_posture_to_bus)


# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, o, _ in _results if o)
failed = sum(1 for _, o, _ in _results if not o)
print(f"\n{'='*60}\n  RESULTS: {len(_results)} tests -- {passed} passed  {failed} failed\n{'='*60}")
if failed:
    print("\nFailed:")
    for n, o, e in _results:
        if not o: print(f"  XX {n}: {e}")
import sys
sys.exit(1 if failed else 0)
