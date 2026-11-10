"""
llm_provider.py — one interface over Anthropic (Claude) and OpenAI for the AI trader, with
automatic failover.

The trading loop, the debate gate and the research loop speak "Anthropic shape": a system
prompt, tool definitions with input_schema, and a message history whose assistant turns hold
text / tool_use blocks and whose user turns hold tool_result blocks. Both providers are driven
from that shape:

  • AnthropicProvider passes it straight through.
  • OpenAIProvider translates to chat-completions function calling and translates the answer
    back, so the rest of the app never knows which model replied.

LLMRouter tries the primary provider first (AI_PROVIDER_PRIMARY, default anthropic) and, when
AI_PROVIDER_FAILOVER is on, retries the same request on the other one for errors that mean
"this provider cannot serve right now": no credits, bad key, rate limit, overload, outage, or
unknown model. A malformed request (HTTP 400 of our own making) is never retried elsewhere.
Each provider's last outcome is kept for the dashboard's connection status.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

# USD per 1M tokens: input, output, cached input (write cost only applies to Anthropic)
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"in": 4.00,  "out": 20.0, "cache_read": 0.20,  "cache_write": 5.00},
    "claude-sonnet-5-5": {"in": 2.00,  "out": 10.0, "cache_read": 0.20,  "cache_write": 2.50},
    "claude-haiku-5-5":  {"in": 0.10,  "out": 0.50, "cache_read": 0.01,  "cache_write": 0.125},
    "claude-fable-5-1":  {"in": 10.00, "out": 50.0, "cache_read": 0.25,  "cache_write": 12.50},
    "gpt-5":             {"in": 1.25,  "out": 10.0, "cache_read": 0.125, "cache_write": 0.0},
    "gpt-5-mini":        {"in": 0.25,  "out": 2.0,  "cache_read": 0.025, "cache_write": 0.0},
    "gpt-4.1":           {"in": 2.00,  "out": 8.0,  "cache_read": 0.50,  "cache_write": 0.0},
    "gpt-4.1-mini":      {"in": 0.40,  "out": 1.6,  "cache_read": 0.10,  "cache_write": 0.0},
    "gpt-4o":            {"in": 2.50,  "out": 10.0, "cache_read": 1.25,  "cache_write": 0.0},
    "o3":                {"in": 2.00,  "out": 8.0,  "cache_read": 0.50,  "cache_write": 0.0},
}
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

# If the configured model is not available on the account (404 / "model not found"), the provider
# steps down this ladder before the router moves to the other provider.
MODEL_LADDER = {
    "anthropic": ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"],
    "openai":    ["gpt-5", "gpt-4.1", "gpt-4o"],
}


def step_down(provider: str, model: str) -> Optional[str]:
    ladder = MODEL_LADDER.get(provider, [])
    for i, m in enumerate(ladder):
        if model == m or model.startswith(m):
            return ladder[i + 1] if i + 1 < len(ladder) else None
    return ladder[0] if ladder and model not in ladder else None


def price_for(model: str) -> dict[str, float]:
    if model in PRICES:
        return PRICES[model]
    base = model.split("-20")[0]                      # dated snapshots → family price
    for k, v in PRICES.items():
        if base.startswith(k):
            return v
    return PRICES["claude-opus-5-5"] if model.startswith("claude") else PRICES["gpt-4.1"]


class Block(dict):
    """A content block that is both a plain dict (JSON-serialisable, accepted by both SDKs as
    history) and attribute-accessible (block.type / block.name / block.input / block.text)."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None


class Usage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0):
        self.input_tokens = int(input_tokens or 0)
        self.output_tokens = int(output_tokens or 0)
        self.cache_read_input_tokens = int(cache_read_input_tokens or 0)
        self.cache_creation_input_tokens = int(cache_creation_input_tokens or 0)


class LLMResponse:
    def __init__(self, content: list[Block], stop_reason: str, usage: Usage, provider: str, model: str,
                 stop_details=None, raw=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        self.provider = provider
        self.model = model
        self.stop_details = stop_details
        self.raw = raw


class ProviderError(Exception):
    """kind: no_key | billing | bad_key | rate_limited | overloaded | server | connection | not_found |
    invalid_request | refused. failover says whether the other provider should be tried."""
    def __init__(self, provider: str, kind: str, message: str, code: int = 0, failover: bool = True):
        super().__init__(message)
        self.provider, self.kind, self.message, self.code, self.failover = provider, kind, message, code, failover


def _classify(provider: str, code: int, message: str, connection: bool = False) -> ProviderError:
    m = (message or "").lower()
    if connection:
        return ProviderError(provider, "connection", f"could not reach the {provider} API", 0, True)
    if "credit" in m or "billing" in m or "balance" in m or "insufficient_quota" in m or "quota" in m or code == 402:
        return ProviderError(provider, "billing", message, code, True)
    if code == 401 or "api key" in m and ("invalid" in m or "incorrect" in m):
        return ProviderError(provider, "bad_key", message, code, True)
    if code == 429:
        return ProviderError(provider, "rate_limited", message, code, True)
    if code in (529, 503):
        return ProviderError(provider, "overloaded", message, code, True)
    if code == 404 or "model" in m and "not found" in m or "does not exist" in m:
        return ProviderError(provider, "not_found", message, code, True)
    if code >= 500:
        return ProviderError(provider, "server", message, code, True)
    if code == 403:
        return ProviderError(provider, "bad_key", message, code, True)
    return ProviderError(provider, "invalid_request", message, code, False)


def _block_dict(b) -> dict:
    """SDK block, dict, or a simple object with attributes → plain dict."""
    if isinstance(b, dict):
        return dict(b)
    if hasattr(b, "model_dump"):
        return b.model_dump(exclude_none=True)
    return {k: v for k, v in vars(b).items() if not k.startswith("_")}


# ── Anthropic ─────────────────────────────────────────────────────────────

class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        self._client = None
        self.model_override: Optional[str] = None     # set by the router after a step-down

    @property
    def model(self) -> str:
        return self.model_override or settings.ai_agent_model

    def configured(self) -> bool:
        return bool(settings.anthropic_api_key)

    def reset(self) -> None:
        self._client = None

    def client(self):
        import anthropic
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def create(self, *, messages: list, system: str, tools: list, max_tokens: int,
                     effort: Optional[str] = None, tool_choice: Optional[dict] = None) -> LLMResponse:
        import anthropic
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=system, tools=tools,
                                      messages=messages, cache_control={"type": "ephemeral"})
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        try:
            client = self.client()
            if self.model in _FALLBACK_MODELS:
                resp = await client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                         fallbacks="default", **kwargs)
            else:
                resp = await client.messages.create(**kwargs)
        except anthropic.APIConnectionError as exc:
            raise _classify(self.name, 0, str(exc), connection=True) from exc
        except anthropic.APIStatusError as exc:
            body = getattr(exc, "body", None)
            detail = str((body.get("error") or {}).get("message") or "") if isinstance(body, dict) else ""
            raise _classify(self.name, exc.status_code, detail or exc.message) from exc
        content = [Block(_block_dict(b)) for b in resp.content]
        u = resp.usage
        usage = Usage(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0),
                      getattr(u, "cache_read_input_tokens", 0), getattr(u, "cache_creation_input_tokens", 0))
        return LLMResponse(content, resp.stop_reason, usage, self.name, getattr(resp, "model", None) or self.model,
                           getattr(resp, "stop_details", None), resp)


# ── OpenAI ────────────────────────────────────────────────────────────────

def _reasoning_model(model: str) -> bool:
    m = model.lower()
    return (m.startswith("o") and m[1:2].isdigit()) or (m.startswith("gpt-5") and "chat" not in m)


def to_openai_messages(system: str, messages: list) -> list[dict]:
    """Anthropic-shaped history → chat-completions messages."""
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text_parts, tool_calls = [], []
            for b in content:
                t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if t == "text":
                    text_parts.append(b["text"] if isinstance(b, dict) else b.text)
                elif t == "tool_use":
                    bid = b["id"] if isinstance(b, dict) else b.id
                    name = b["name"] if isinstance(b, dict) else b.name
                    inp = (b.get("input") if isinstance(b, dict) else b.input) or {}
                    tool_calls.append({"id": bid, "type": "function",
                                       "function": {"name": name, "arguments": json.dumps(inp, default=str)}})
                # thinking / fallback blocks have no OpenAI equivalent and are dropped
            msg: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            texts = []
            for b in content:
                t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if t == "tool_result":
                    c = b.get("content") if isinstance(b, dict) else b.content
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"] if isinstance(b, dict) else b.tool_use_id,
                                "content": str(c)})
                elif t == "text":
                    texts.append(b["text"] if isinstance(b, dict) else b.text)
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
            for t in tools]


_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAIProvider:
    name = "openai"

    def __init__(self, http_client=None):
        self._client = None
        self._http_client = http_client          # tests inject a mock transport
        self.model_override: Optional[str] = None

    @property
    def model(self) -> str:
        return self.model_override or settings.openai_model

    def configured(self) -> bool:
        return bool(settings.openai_api_key)

    def reset(self) -> None:
        self._client = None

    def client(self):
        import openai
        if self._client is None:
            kw = {"api_key": settings.openai_api_key}
            if self._http_client is not None:
                kw["http_client"] = self._http_client
            self._client = openai.AsyncOpenAI(**kw)
        return self._client

    async def create(self, *, messages: list, system: str, tools: list, max_tokens: int,
                     effort: Optional[str] = None, tool_choice: Optional[dict] = None) -> LLMResponse:
        import openai
        kwargs: dict[str, Any] = dict(model=self.model, messages=to_openai_messages(system, messages),
                                      max_completion_tokens=max_tokens)
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if tool_choice:
            if tool_choice.get("type") == "tool":
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
            elif tool_choice.get("type") in ("any", "auto"):
                kwargs["tool_choice"] = "required" if tool_choice["type"] == "any" else "auto"
        if effort and _reasoning_model(self.model):
            kwargs["reasoning_effort"] = _EFFORT.get(effort, "medium")
        try:
            resp = await self.client().chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            raise _classify(self.name, 0, str(exc), connection=True) from exc
        except openai.APIStatusError as exc:
            body = getattr(exc, "body", None)
            detail = ""
            if isinstance(body, dict):
                err = body.get("error") if isinstance(body.get("error"), dict) else body
                detail = str(err.get("message") or "") + (f" ({err.get('code')})" if err.get("code") else "")
            raise _classify(self.name, exc.status_code, detail or str(exc)) from exc
        choice = resp.choices[0]
        msg = choice.message
        content: list[Block] = []
        if msg.content:
            content.append(Block(type="text", text=msg.content))
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {"_raw": tc.function.arguments}
            content.append(Block(type="tool_use", id=tc.id, name=tc.function.name, input=args))
        fr = choice.finish_reason
        stop = ("tool_use" if (fr == "tool_calls" or msg.tool_calls) else
                "max_tokens" if fr == "length" else "refusal" if fr == "content_filter" else "end_turn")
        if getattr(msg, "refusal", None):
            stop = "refusal"
        u = resp.usage
        cached = 0
        try:
            cached = int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
        except Exception:
            pass
        usage = Usage(getattr(u, "prompt_tokens", 0) - cached if u else 0, getattr(u, "completion_tokens", 0) if u else 0,
                      cached, 0)
        return LLMResponse(content, stop, usage, self.name, resp.model or self.model, None, resp)


# ── Router ────────────────────────────────────────────────────────────────

class LLMRouter:
    def __init__(self, anthropic_provider=None, openai_provider=None):
        self.providers = {"anthropic": anthropic_provider or AnthropicProvider(),
                          "openai": openai_provider or OpenAIProvider()}
        self.health: dict[str, dict] = {n: {"state": "untested", "detail": "no call made yet", "at": None}
                                        for n in self.providers}
        self.last_used: Optional[str] = None
        self.failovers = 0

    def order(self) -> list[str]:
        primary = settings.ai_provider_primary if settings.ai_provider_primary in self.providers else "anthropic"
        others = [n for n in self.providers if n != primary]
        return [primary] + (others if settings.ai_provider_failover else [])

    def reset(self) -> None:
        for p in self.providers.values():
            p.reset()

    def configured(self) -> list[str]:
        return [n for n in self.order() if self.providers[n].configured()]

    def _mark(self, name: str, state: str, detail: str, **extra) -> None:
        self.health[name] = {"state": state, "detail": detail[:300], "at": now_ist().isoformat(timespec="seconds"),
                             "model": self.providers[name].model, **extra}

    async def create(self, **kwargs) -> LLMResponse:
        order = self.order()
        candidates = [n for n in order if self.providers[n].configured()]
        if not candidates:
            raise ProviderError("none", "no_key", "no AI provider key set (Anthropic or OpenAI)", 0, False)
        last: Optional[ProviderError] = None
        for i, name in enumerate(candidates):
            prov = self.providers[name]
            try:
                resp = await self._create_with_step_down(prov, kwargs)
            except ProviderError as exc:
                self._mark(name, exc.kind, exc.message, code=exc.code)
                last = exc
                if not exc.failover:
                    raise
                if i + 1 < len(candidates):
                    self.failovers += 1
                    logger.warning("[llm] {} unavailable ({}: {}) — failing over to {}", name, exc.kind,
                                   exc.message[:120], candidates[i + 1])
                continue
            self._mark(name, "ok", f"{prov.model} responding")
            if self.last_used and self.last_used != name:
                logger.info("[llm] now served by {}", name)
            self.last_used = name
            return resp
        assert last is not None
        raise last

    async def _create_with_step_down(self, prov, kwargs) -> LLMResponse:
        """Try the configured model; on 'model not found' walk down the provider's ladder."""
        tried = []
        while True:
            try:
                return await prov.create(**kwargs)
            except ProviderError as exc:
                if exc.kind != "not_found":
                    raise
                tried.append(prov.model)
                nxt = step_down(prov.name, prov.model)
                if not nxt or nxt in tried:
                    raise
                logger.warning("[llm] {} model {} not available — stepping down to {}", prov.name, prov.model, nxt)
                prov.model_override = nxt

    def status(self) -> dict:
        order = self.order()
        prov: dict[str, dict] = {}
        for n in self.providers:
            p = self.providers[n]
            h = dict(self.health[n])
            if not p.configured():
                h = {"state": "no_key", "detail": f"{n} API key not set", "at": None}
            h["model"] = p.model
            h["role"] = "primary" if n == order[0] else ("fallback" if n in order else "off")
            prov[n] = h
        configured = [n for n in order if self.providers[n].configured()]
        if not configured:
            top = {"state": "no_key", "detail": "No AI key set — add an Anthropic or OpenAI key"}
        else:
            healthy = [n for n in configured if prov[n]["state"] in ("ok", "untested")]
            active = self.last_used if self.last_used in configured else configured[0]
            if prov[active]["state"] != "ok":
                ok = [n for n in configured if prov[n]["state"] == "ok"]   # a verified route beats a failing primary
                if ok:
                    active = ok[0]
            if prov[active]["state"] == "ok":
                top = {"state": "ok", "detail": f"AI served by {active} ({prov[active]['model']})"
                                                 + (" — primary down, on fallback" if active != order[0] else "")}
            elif healthy:
                top = {"state": "untested", "detail": f"{healthy[0]} key saved — no call made yet"}
            else:
                worst = prov[active]
                reason = {"billing": "no credits", "bad_key": "key rejected", "rate_limited": "rate limited",
                          "overloaded": "overloaded", "server": "provider error", "connection": "unreachable",
                          "not_found": "model not found", "invalid_request": "request rejected"}.get(worst["state"], worst["state"])
                top = {"state": worst["state"], "detail": f"All AI providers failing — {active}: {reason}. {worst['detail'][:120]}"}
        return {**top, "providers": prov, "active": self.last_used, "primary": order[0], "failovers": self.failovers,
                "model": prov[order[0]]["model"]}


llm_router = LLMRouter()
