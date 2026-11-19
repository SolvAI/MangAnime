"""Single client wrapper for ALL LLM calls (brief §3b).

- endpoint "local":    llama.cpp server (Qwen 3.8 27B Q4 + mmproj), OpenAI-compatible.
- endpoint "deepseek": DeepSeek V4.1 Flash API (everything written: scripts, dialogue, prompts, text checks).
- endpoint "gemini":   Gemini 3.8 Flash API (everything visual: reading manga pages, identifying characters, visual checks).

Features: structured JSON output with JSON-schema validation and retries, disk cache keyed by
content hash (reruns are free; offline replays work), token/cost logging, budget guard with automatic
fallback to cache-only + local Qwen, backoff on rate limits. The API key is read from .env and never logged.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import jsonschema

from src.common.config import CFG, P, load_env
from src.common.io import append_jsonl, read_json, read_jsonl, sha, write_json

_LOCK = threading.Lock()
COST_LOG = lambda: P("runs") / "api_cost.jsonl"  # noqa: E731


class CacheMiss(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def encode_image(img, max_side: int = 768, quality: int = 85) -> str:
    """PIL image / path / ndarray -> data URL (downscaled JPEG)."""
    from PIL import Image
    import numpy as np

    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _is_peak(now: datetime | None = None) -> bool:
    """DeepSeek peak hours: 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    h = now.hour + now.minute / 60
    return (1 <= h < 4) or (6 <= h < 10)


def api_spend() -> float:
    return sum(r.get("cost_usd", 0.0) for r in read_jsonl(COST_LOG()))


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply (tolerates code fences / leading prose / thinking tags)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # first balanced {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i = text.find(open_c)
        if i < 0:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_c:
                    depth += 1
                elif c == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except Exception:
                            break
    raise ValueError("no valid JSON in reply")


class LLM:
    """llm = LLM('local') / LLM('deepseek', role='judge')."""

    def __init__(self, endpoint: str = "local", role: str = "misc", timeout: float = 600.0):
        assert endpoint in ("local", "deepseek", "gemini")
        self.endpoint = endpoint
        self.role = role
        self.cfg = CFG.llm[endpoint]
        self.cache_dir = P("cache") / "llm" / endpoint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.offline = os.environ.get("KOMI_OFFLINE", "0") == "1"
        load_env()

    # ---------------------------------------------------------------- core
    def chat(
        self,
        messages: list[dict],
        *,
        schema: dict | None = None,
        images: list | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        thinking: bool = False,
        effort: str = "low",
        seed: int = 0,
        retries: int = 3,
        image_max_side: int = 768,
        cache: bool = True,
        tag: str = "",
        media_resolution: str | None = None,
    ) -> Any:
        """Returns parsed JSON (if schema given) or text. `images` are appended to the last user message."""
        msgs = [dict(m) for m in messages]
        if images:
            urls = [encode_image(i, image_max_side) for i in images]
            last = msgs[-1]
            content = [{"type": "text", "text": last["content"]}] if isinstance(last["content"], str) else list(last["content"])
            content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
            msgs[-1] = {"role": last["role"], "content": content}
        key = sha(
            {"m": msgs, "schema": schema, "model": self.cfg.model, "max_tokens": max_tokens, "t": temperature,
             "thinking": thinking, "effort": effort if (thinking or self.endpoint == "gemini") else "", "seed": seed,
             **({"mres": media_resolution} if media_resolution else {})},
            n=32,
        )
        self._media_resolution = media_resolution
        cpath = self.cache_dir / key[:2] / f"{key}.json"
        if cache and cpath.exists():
            hit = read_json(cpath)
            return hit["parsed"] if schema is not None else hit["text"]
        if self.offline and self.endpoint in ("deepseek", "gemini"):
            raise CacheMiss(f"offline and no cache for {self.role}:{tag}:{key}")
        if self.endpoint in ("deepseek", "gemini"):
            self._budget_check()

        last_err: Exception | None = None
        work_msgs = msgs
        for attempt in range(retries + 1):
            try:
                text, usage = self._post(work_msgs, schema is not None, max_tokens, temperature, thinking, effort, seed + attempt)
            except Exception as e:  # network / server errors: backoff and retry
                last_err = e
                time.sleep(min(60, 3 * 2**attempt))
                continue
            self._log_cost(usage, tag, key)
            if schema is None:
                if cache:
                    write_json(cpath, {"text": text, "usage": usage})
                return text
            try:
                parsed = extract_json(text)
                jsonschema.validate(parsed, schema)
                if cache:
                    write_json(cpath, {"text": text, "parsed": parsed, "usage": usage})
                return parsed
            except Exception as e:
                last_err = e
                append_jsonl(P("runs") / "llm_validation_errors.jsonl", {"ts": time.strftime("%F %T"), "role": self.role, "tag": tag,
                                                                         "attempt": attempt, "error": str(e)[:400], "reply_head": text[:300]})
                work_msgs = msgs + [
                    {"role": "assistant", "content": text[:6000]},
                    {"role": "user", "content": f"Your reply was not valid for the required JSON schema ({str(e)[:300]}). "
                                                 "Reply again with ONLY one valid JSON object that follows the schema."},
                ]
        raise RuntimeError(f"LLM call failed after retries ({self.endpoint}/{self.role}/{tag}): {last_err}")

    # ---------------------------------------------------------------- transport
    def _post_gemini(self, msgs, want_json, max_tokens, effort):
        """generateContent. OpenAI-style messages (text + data-URL images) -> Gemini contents. Thinking cannot be disabled on
        3.8 Flash (lowest level: low); temperature / top_p are not sent (the model's migration notes say to drop them)."""
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system" and isinstance(m["content"], str))
        contents = []
        for m in msgs:
            if m["role"] == "system":
                continue
            parts = []
            for c in ([{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]):
                if c.get("type") == "text":
                    parts.append({"text": c["text"]})
                elif c.get("type") == "image_url":
                    head, b64 = c["image_url"]["url"].split(",", 1)
                    parts.append({"inlineData": {"mimeType": head.split(":")[1].split(";")[0], "data": b64}})
            contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": parts})
        gen: dict[str, Any] = {"maxOutputTokens": max_tokens,
                               "thinkingConfig": {"thinkingLevel": effort if effort in ("low", "medium", "high") else str(self.cfg.thinking_level)}}
        if want_json:
            gen["responseMimeType"] = "application/json"
        if getattr(self, "_media_resolution", None):
            gen["mediaResolution"] = "MEDIA_RESOLUTION_" + self._media_resolution.upper()
        body: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.cfg.base_url}/models/{self.cfg.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
        for attempt in range(6):
            r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(90, 4 * 2**attempt))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            d = r.json()
            cands = d.get("candidates") or []
            if not cands or not (cands[0].get("content") or {}).get("parts"):
                why = (d.get("promptFeedback") or {}).get("blockReason") or (cands[0].get("finishReason") if cands else "no candidates")
                raise RuntimeError(f"Gemini returned no content ({why})")
            text = "".join(p.get("text", "") for p in cands[0]["content"]["parts"] if not p.get("thought"))
            u = d.get("usageMetadata") or {}
            usage = {"prompt_tokens": u.get("promptTokenCount", 0),
                     "completion_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                     "thought_tokens": u.get("thoughtsTokenCount", 0), "cached_tokens": u.get("cachedContentTokenCount", 0),
                     "finish_reason": cands[0].get("finishReason")}
            if cands[0].get("finishReason") == "MAX_TOKENS":
                usage["truncated"] = True
            return text, usage
        raise RuntimeError("rate-limited / server errors after backoff")

    def _post(self, msgs, want_json, max_tokens, temperature, thinking, effort, seed):
        if self.endpoint == "gemini":
            return self._post_gemini(msgs, want_json, max_tokens, effort)
        body: dict[str, Any] = {"model": self.cfg.model, "messages": msgs, "max_tokens": max_tokens, "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.endpoint == "deepseek":
            headers["Authorization"] = "Bearer " + os.environ["DEEPSEEK_API_KEY"]
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
            if thinking:
                body["reasoning_effort"] = effort
            else:
                body["temperature"] = temperature
            if want_json:
                body["response_format"] = {"type": "json_object"}
            url = self.cfg.base_url + "/chat/completions"
        else:
            body["temperature"] = temperature
            body["seed"] = seed
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
            if want_json:
                body["response_format"] = {"type": "json_object"}
            url = self.cfg.base_url + "/chat/completions"
        for attempt in range(6):
            r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(90, 4 * 2**attempt))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            d = r.json()
            msg = d["choices"][0]["message"]
            return (msg.get("content") or ""), d.get("usage", {})
        raise RuntimeError("rate-limited / server errors after backoff")

    # ---------------------------------------------------------------- cost
    def _cost(self, usage: dict) -> float:
        if self.endpoint == "gemini":
            c = self.cfg
            cached = usage.get("cached_tokens", 0) or 0
            return ((usage.get("prompt_tokens", 0) - cached) * c.price_in + cached * c.price_in_cached
                    + usage.get("completion_tokens", 0) * c.price_out) / 1e6
        if self.endpoint != "deepseek":
            return 0.0
        c = self.cfg
        peak = _is_peak()
        pin, pc, pout = (
            (c.price_in_peak, c.price_in_cached_peak, c.price_out_peak) if peak
            else (c.price_in_offpeak, c.price_in_cached_offpeak, c.price_out_offpeak)
        )
        prompt = usage.get("prompt_tokens", 0)
        cached = usage.get("prompt_cache_hit_tokens", 0) or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        out = usage.get("completion_tokens", 0)
        return ((prompt - cached) * pin + cached * pc + out * pout) / 1e6

    def _log_cost(self, usage: dict, tag: str, key: str) -> None:
        with _LOCK:
            append_jsonl(COST_LOG(), {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "endpoint": self.endpoint, "role": self.role, "tag": tag,
                "key": key, "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0), "cost_usd": round(self._cost(usage), 6),
            })

    def _budget_check(self) -> None:
        cap = min(float(CFG.budget.api_usd), float(CFG.budget.api_usd_effective))
        spent = api_spend()
        if spent >= cap * float(CFG.budget.api_soft_fraction):
            raise BudgetExceeded(f"API spend (DeepSeek + Gemini) ${spent:.2f} has reached the stop line ${cap * float(CFG.budget.api_soft_fraction):.2f}: "
                                 "cache-only from here; the remaining calls go to Claude subagents (D-055)")


def remote_or_local(role: str) -> LLM:
    """Director/judge client: DeepSeek while budget allows, else local Qwen (logged by caller)."""
    cap = min(float(CFG.budget.api_usd), float(CFG.budget.api_usd_effective))
    if os.environ.get("KOMI_FORCE_LOCAL", "0") == "1" or api_spend() >= cap * float(CFG.budget.api_soft_fraction):
        return LLM("local", role=role)
    return LLM("deepseek", role=role)


def vision(role: str) -> LLM:
    """Everything visual goes to Gemini (D-044)."""
    return LLM("gemini", role=role)


def writer(role: str) -> LLM:
    """Everything written goes to DeepSeek (D-044)."""
    return LLM("deepseek", role=role)
