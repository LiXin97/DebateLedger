"""Minimal OpenRouter chat-completions wrapper for ABC cohort expansion.

Reads OPENROUTER_API_KEY from env (or .env at repo root). Pins exact model IDs.
Logs response timestamp, token usage, latency. Returns dict with both content
and metadata so debate harness can persist provenance.

Verified model IDs against GET https://openrouter.ai/api/v1/models on 2026-04-25:
  - deepseek/deepseek-v4-pro     [exists]
  - moonshotai/kimi-k2.6         [exists]
  - deepseek/deepseek-v3.2       [exists; original A4 substitute, kept for back-compat]
  - deepseek/deepseek-v4-flash   [exists; created=1777000666; A6 restores V4-Flash slot]
Additional IDs verified on 2026-04-29:
  - minimax/minimax-m2.7         [exists]
  - tencent/hy3-preview:free     [exists]
  - x-ai/grok-4.1-fast           [exists]
  - stepfun/step-3.5-flash       [exists]
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PINNED_MODELS = {
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking",
    "minimax-m2.7": "minimax/minimax-m2.7",
    "hy3-preview-free": "tencent/hy3-preview:free",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "step-3.5-flash": "stepfun/step-3.5-flash",
    "glm-4.6": "z-ai/glm-4.6",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "mistral-small-4": "mistralai/mistral-small-2603",
    "mistral-large-3": "mistralai/mistral-large-2512",
    "mistral-medium-3.1": "mistralai/mistral-medium-3.1",
    "qwen3.6-plus": "qwen/qwen3.6-plus",
    "qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
    # V3.2 was the A4 substitute when V4-Flash wasn't on HF. Kept for
    # back-compat with any in-flight artifacts referencing it.
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    # V4-Flash listed on OpenRouter as of 2026-04-22 (created=1777000666).
    # A6 restores the V4-Flash cohort slot that A4 had to substitute.
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    # Rebuttal additions (2026-07-25): effort-gradient anchor + weak-agent slot
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "qwen3-4b-or": "qwen/qwen3-4b",
    "hy3-preview": "tencent/hy3-preview",
}


def _load_env_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # Fallback: parse .env at common worktree root
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found in env or .env")


def _parse_openrouter_body(body_text: str) -> dict[str, Any]:
    """Parse OpenRouter response body, tolerating SSE keep-alive comments.

    OpenRouter prepends `: OPENROUTER PROCESSING` comment lines on slow
    upstreams (reasoning models like deepseek-v4-pro / kimi-k2.6) even when
    the request is non-streaming. Plain `json.loads` then chokes on those
    leading lines. Locate the first `{` and use raw_decode to consume the
    JSON object only.
    """
    idx = body_text.find("{")
    if idx < 0:
        raise RuntimeError(f"OpenRouter body has no JSON object: {body_text[:500]!r}")
    obj, _ = json.JSONDecoder().raw_decode(body_text[idx:])
    return obj


def chat(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call OpenRouter chat-completions. Returns structured dict.

    Result schema:
        {
          "model_id": str,                # pinned id sent
          "model_returned": str,          # id returned by upstream
          "content": str,                 # assistant text
          "request_ts": float,            # epoch seconds, pre-call
          "response_ts": float,           # epoch seconds, post-call
          "latency_s": float,
          "usage": {prompt_tokens, completion_tokens, total_tokens},
          "raw": {...}                    # full upstream JSON
        }
    """
    key = _load_env_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/abc-oral-push",
        "X-Title": "ABC NeurIPS 2026 cohort expansion",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
    request_ts = time.time()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_text = resp.read().decode("utf-8")
        raw = _parse_openrouter_body(body_text)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTPError {e.code}: {err_body}") from e
    latency_s = time.perf_counter() - t0
    response_ts = time.time()
    choice0 = (raw.get("choices") or [{}])[0]
    content = (choice0.get("message") or {}).get("content") or ""
    return {
        "model_id": model_id,
        "model_returned": raw.get("model", model_id),
        "content": content,
        "request_ts": request_ts,
        "response_ts": response_ts,
        "latency_s": latency_s,
        "usage": raw.get("usage", {}),
        "provider_resolved": raw.get("provider"),
        "generation_id": raw.get("id"),
        "raw": raw,
    }


if __name__ == "__main__":
    import sys

    mid = sys.argv[1] if len(sys.argv) > 1 else PINNED_MODELS["deepseek-v4-pro"]
    out = chat(mid, [{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=16)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=2))


# ---------------------------------------------------------------------------
# Async sister API: httpx.AsyncClient-based for concurrent debate harness use.
# Identical schema to chat(); supports OpenRouter `provider` field for
# upstream-pinning (e.g. force DeepSeek for v4-pro to reduce cross-provider
# 429 thrash) and accepts a shared AsyncClient so the caller controls
# connection pooling.
# ---------------------------------------------------------------------------


def _build_request(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    extra_headers: dict[str, str] | None,
    provider: dict[str, Any] | None,
    reasoning: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    key = _load_env_key()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/abc-oral-push",
        "X-Title": "ABC NeurIPS 2026 cohort expansion",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if provider is not None:
        payload["provider"] = provider
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return headers, payload


async def achat(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: float = 180.0,
    extra_headers: dict[str, str] | None = None,
    provider: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    http_client: "Any | None" = None,
) -> dict[str, Any]:
    """Async sibling of `chat()`. Uses httpx.AsyncClient.

    If `http_client` is provided (recommended for many calls), reuses the
    pool. Otherwise creates a single-shot client.
    """
    import httpx  # local import — only async path needs it

    headers, payload = _build_request(
        model_id, messages,
        max_tokens=max_tokens, temperature=temperature,
        extra_headers=extra_headers, provider=provider, reasoning=reasoning,
    )
    request_ts = time.time()
    t0 = time.perf_counter()

    async def _do(client):
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        body_text = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTPError {resp.status_code}: {body_text[:1000]}")
        return _parse_openrouter_body(body_text)

    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            raw = await _do(client)
    else:
        raw = await _do(http_client)

    latency_s = time.perf_counter() - t0
    response_ts = time.time()
    choice0 = (raw.get("choices") or [{}])[0]
    content = (choice0.get("message") or {}).get("content") or ""
    return {
        "model_id": model_id,
        "model_returned": raw.get("model", model_id),
        "content": content,
        "request_ts": request_ts,
        "response_ts": response_ts,
        "latency_s": latency_s,
        "usage": raw.get("usage", {}),
        "provider_resolved": raw.get("provider"),
        "generation_id": raw.get("id"),
        "raw": raw,
    }
