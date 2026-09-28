"""Async Gemini API client with concurrency control, retries, and budget tracking."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from .cost import BudgetExhaustedError, CostTracker
from .types import BatchItem, GenerateResult

logger = logging.getLogger(__name__)


class GeminiClient:
    """High-level async client for Gemini with budget control and batching.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"models/gemini-2.5-flash"``.
    budget_cap : float
        Maximum spend in USD.
    max_concurrent : int
        Maximum number of concurrent API requests (semaphore width).
    max_retries : int
        Maximum retry attempts on transient errors.
    log_dir : str | Path | None
        If set, every request/response pair is appended to a JSONL file in
        this directory.
    thinking_budget : int | None
        Gemini thinking budget in tokens. ``None`` uses a model-aware default:
        512 for Gemini 3/3.1 Pro preview models and 0 otherwise.
    """

    def __init__(
        self,
        model: str = "models/gemini-2.5-flash",
        budget_cap: float = 200.0,
        max_concurrent: int = 20,
        max_retries: int = 5,
        log_dir: str | Path | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.thinking_budget = (
            self._default_thinking_budget(model)
            if thinking_budget is None else thinking_budget
        )

        # Google GenAI client -- picks up GEMINI_API_KEY from env automatically.
        self._client = genai.Client()

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.cost_tracker = CostTracker(budget=budget_cap)

        # Optional JSONL logging.
        self._log_path: Path | None = None
        self._log_file = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"gemini_log_{ts}.jsonl"
            self._log_file = open(self._log_path, "a", encoding="utf-8")
            logger.info("Logging API calls to %s", self._log_path)

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    async def generate(
        self,
        contents: str | list,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
        metadata: dict | None = None,
    ) -> GenerateResult:
        """Send a single generate request with budget/concurrency control.

        Parameters
        ----------
        contents : str | list
            Prompt text or structured message list.
        temperature : float
            Sampling temperature.
        max_output_tokens : int
            Maximum tokens to generate.
        system_instruction : str | None
            Optional system prompt.
        metadata : dict | None
            Arbitrary context passed through to ``GenerateResult.metadata``.

        Returns
        -------
        GenerateResult
        """
        if metadata is None:
            metadata = {}

        async with self._semaphore:
            # Pre-flight budget check.
            self.cost_tracker.check_budget()

            config = self._build_config(temperature, max_output_tokens, system_instruction)
            response = await self._call_with_retry(contents, config)

            # Extract text -- may be None if blocked by safety filters.
            text = ""
            try:
                text = response.text or ""
            except Exception:
                logger.warning("Could not extract text from response (possibly safety-blocked)")

            # Token counts.
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

            if thinking_tokens > 0 and self.thinking_budget == 0:
                logger.warning(
                    "Thinking tokens > 0 (%d) despite requested thinking_budget=0. "
                    "Check model/config.",
                    thinking_tokens,
                )

            # Record cost.
            # Gemini bills thinking tokens as output tokens, while the SDK keeps
            # them separate from candidate tokens in usage_metadata.
            billable_output_tokens = output_tokens + thinking_tokens
            cost = await self.cost_tracker.record(
                input_tokens, billable_output_tokens, self.model
            )

            timestamp = datetime.now(timezone.utc).isoformat()

            result = GenerateResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model=self.model,
                timestamp=timestamp,
                metadata=metadata,
                thinking_tokens=thinking_tokens,
            )

            # JSONL logging.
            if self._log_file is not None:
                log_entry = {
                    "timestamp": timestamp,
                    "model": self.model,
                    "contents": contents if isinstance(contents, str) else str(contents),
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_text": text[:500],  # truncate for log size
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "cost": cost,
                    "metadata": metadata,
                }
                self._log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                self._log_file.flush()

            return result

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        contents: str | list,
        config: types.GenerateContentConfig,
    ):
        """Call the Gemini API with exponential backoff on transient errors."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response
            except Exception as exc:
                last_exc = exc
                exc_name = type(exc).__name__

                # Determine if retryable.
                retryable = False
                # google-api-core exceptions for 429 / 503.
                try:
                    from google.api_core import exceptions as gapi_exc

                    if isinstance(exc, (gapi_exc.ResourceExhausted, gapi_exc.ServiceUnavailable)):
                        retryable = True
                except ImportError:
                    pass

                # google-genai SDK may wrap errors differently.
                err_str = str(exc).lower()
                if any(
                    kw in err_str
                    for kw in ("429", "503", "resource exhausted", "service unavailable",
                               "rate limit", "connection", "timeout", "internal")
                ):
                    retryable = True

                if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                    retryable = True

                if not retryable or attempt == self.max_retries:
                    logger.error(
                        "Non-retryable error or max retries reached: %s: %s",
                        exc_name, exc,
                    )
                    raise

                wait = min(2**attempt + random.random(), 60.0)
                logger.warning(
                    "Retryable error (%s: %s), attempt %d/%d, waiting %.1fs",
                    exc_name, exc, attempt + 1, self.max_retries, wait,
                )
                await asyncio.sleep(wait)

        # Should not reach here, but just in case.
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Batch generate
    # ------------------------------------------------------------------

    async def generate_batch(
        self,
        items: list[BatchItem],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        desc: str = "",
    ) -> list[GenerateResult]:
        """Generate responses for a batch of items concurrently.

        Concurrency is bounded by the semaphore set in ``__init__``.
        A ``tqdm`` progress bar is displayed.

        Parameters
        ----------
        items : list[BatchItem]
            Items to process.
        temperature : float
            Default temperature (overridden per-item if ``BatchItem.temperature``
            is set).
        max_output_tokens : int
            Maximum output tokens per call.
        desc : str
            Description for the progress bar.

        Returns
        -------
        list[GenerateResult]
            Results in the same order as *items*.
        """
        from tqdm.asyncio import tqdm_asyncio

        async def _process(item: BatchItem) -> GenerateResult:
            temp = item.temperature if item.temperature is not None else temperature
            result = await self.generate(
                contents=item.contents,
                temperature=temp,
                max_output_tokens=max_output_tokens,
                system_instruction=item.system_instruction,
                metadata=item.metadata,
            )
            return result

        tasks = [_process(item) for item in items]
        results = await tqdm_asyncio.gather(*tasks, desc=desc or "Generating")
        return list(results)

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def generate_sync(self, contents: str | list, **kwargs) -> GenerateResult:
        """Synchronous wrapper around :meth:`generate`.

        Handles the case where an event loop is already running by dispatching
        to a background thread.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop -- run in a separate thread.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.generate(contents, **kwargs))
                return future.result()
        else:
            return asyncio.run(self.generate(contents, **kwargs))

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(
        self,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` with model-aware thinking."""
        kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _default_thinking_budget(model: str) -> int:
        """Return a conservative thinking budget compatible with Gemini previews."""
        name = model.lower().removeprefix("models/")
        if name.startswith("gemini-3.1-pro") or name.startswith("gemini-3-pro"):
            return 512
        return 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the log file if open."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    async def aclose(self) -> None:
        """Close local files and async HTTP sessions owned by the GenAI SDK."""
        self.close()
        api_client = getattr(self._client, "_api_client", None)
        if api_client is None:
            return

        aiohttp_session = getattr(api_client, "_aiohttp_session", None)
        if aiohttp_session is not None and not aiohttp_session.closed:
            await aiohttp_session.close()

        async_httpx_client = getattr(api_client, "_async_httpx_client", None)
        if async_httpx_client is not None:
            await async_httpx_client.aclose()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model={self.model!r}, "
            f"cost={self.cost_tracker})"
        )
