"""OpenRouter cross-benchmark collapse/correction smoke harness.

This script mirrors ``cross_benchmark_forecast_replication.py`` but swaps the
Gemini client for the repository's pinned OpenRouter chat wrapper. It is meant
for bounded reviewer-hardening runs under an explicit OpenRouter cost cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abc_exp.experiments.cross_benchmark_forecast_replication import (  # noqa: E402
    Config,
    load_completed,
    load_questions_for_config,
    process_question,
    summarize,
)
from abc_exp.scripts.openrouter_client import PINNED_MODELS, achat  # noqa: E402
from abc_exp.src.api.types import GenerateResult  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OpenRouterBudgetExceeded(RuntimeError):
    pass


def _parts_to_text(parts: list[dict[str, Any]] | str | None) -> str:
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            chunks.append(str(part.get("text", "")))
        else:
            chunks.append(str(part))
    return "\n".join(c for c in chunks if c)


def _to_openrouter_messages(contents: str | list, system_instruction: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages
    for item in contents:
        role = item.get("role", "user") if isinstance(item, dict) else "user"
        if role == "model":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        text = _parts_to_text(item.get("parts") if isinstance(item, dict) else str(item))
        if text:
            messages.append({"role": role, "content": text})
    return messages


def _openrouter_cost(result: dict[str, Any]) -> float:
    for container in (result.get("usage") or {}, (result.get("raw") or {}).get("usage") or {}):
        value = container.get("cost")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _token_count(result: dict[str, Any], key: str) -> int:
    for container in (result.get("usage") or {}, (result.get("raw") or {}).get("usage") or {}):
        value = container.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _provider_name(result: dict[str, Any]) -> str | None:
    provider = result.get("provider_resolved") or (result.get("raw") or {}).get("provider")
    if isinstance(provider, str):
        return provider
    if isinstance(provider, dict):
        return provider.get("name") or provider.get("provider") or provider.get("id")
    return None


def _provider_payload(pin_provider: str | None) -> dict[str, Any] | None:
    if not pin_provider:
        return None
    return {"order": [pin_provider], "allow_fallbacks": False}


def _retry_delay(exc: BaseException, attempt: int) -> float:
    text = str(exc).lower()
    if "429" in text or "rate" in text:
        return min(45.0, 10.0 + 5.0 * attempt + random.random() * 3.0)
    return min(30.0, 2.0 ** attempt) + random.random()


class OpenRouterGenerateClient:
    """Small adapter exposing the GeminiClient.generate shape."""

    def __init__(
        self,
        model: str,
        *,
        budget_cap: float,
        max_concurrent: int,
        log_dir: str | Path,
        pin_provider: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        self.model_alias = model
        self.model_id = PINNED_MODELS.get(model, model)
        self.budget_cap = float(budget_cap)
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.n_calls = 0
        self.provider_counts: Counter[str] = Counter()
        self.returned_model_counts: Counter[str] = Counter()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._provider = _provider_payload(pin_provider)
        self._reasoning = reasoning
        self._http = httpx.AsyncClient(timeout=180.0)
        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_path = log_root / f"openrouter_log_{ts}.jsonl"
        self._log_file = self._log_path.open("a", encoding="utf-8")
        logger.info("Logging OpenRouter calls to %s", self._log_path)

    async def _reserve_budget(self) -> None:
        async with self._lock:
            if self.total_cost >= self.budget_cap:
                raise OpenRouterBudgetExceeded(
                    f"OpenRouter budget exhausted: ${self.total_cost:.4f} >= ${self.budget_cap:.2f}"
                )

    async def _record(self, result: dict[str, Any], metadata: dict[str, Any]) -> float:
        cost = _openrouter_cost(result)
        input_tokens = _token_count(result, "prompt_tokens")
        output_tokens = _token_count(result, "completion_tokens")
        provider = _provider_name(result) or "unknown"
        returned_model = result.get("model_returned") or self.model_id
        async with self._lock:
            if self.total_cost + cost > self.budget_cap:
                raise OpenRouterBudgetExceeded(
                    f"OpenRouter budget exhausted: current ${self.total_cost:.4f} + "
                    f"this call ${cost:.4f} > cap ${self.budget_cap:.2f}"
                )
            self.total_cost += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.n_calls += 1
            self.provider_counts[provider] += 1
            self.returned_model_counts[str(returned_model)] += 1
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_alias": self.model_alias,
            "model_id": self.model_id,
            "model_returned": returned_model,
            "provider_resolved": provider,
            "generation_id": result.get("generation_id"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata,
        }
        self._log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self._log_file.flush()
        return cost

    async def generate(
        self,
        contents: str | list,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
        metadata: dict | None = None,
    ) -> GenerateResult:
        metadata = metadata or {}
        messages = _to_openrouter_messages(contents, system_instruction)
        last_exc: BaseException | None = None
        for attempt in range(1, 6):
            await self._reserve_budget()
            try:
                async with self._semaphore:
                    result = await achat(
                        self.model_id,
                        messages,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                        provider=self._provider,
                        reasoning=self._reasoning,
                        http_client=self._http,
                    )
                cost = await self._record(result, metadata)
                usage = result.get("usage") or {}
                return GenerateResult(
                    text=result.get("content", ""),
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    cost=cost,
                    model=result.get("model_returned") or self.model_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metadata=metadata,
                    thinking_tokens=0,
                )
            except OpenRouterBudgetExceeded:
                raise
            except Exception as exc:
                last_exc = exc
                text = str(exc).lower()
                retryable = any(marker in text for marker in ("429", "503", "504", "rate", "timeout", "timed out"))
                if not retryable or attempt == 5:
                    raise
                delay = _retry_delay(exc, attempt)
                logger.warning("Retrying OpenRouter call after %s (attempt %d/5, %.1fs)", exc, attempt, delay)
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def aclose(self) -> None:
        self._log_file.close()
        await self._http.aclose()

    def summary(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "model_id": self.model_id,
            "budget_cap": self.budget_cap,
            "total_cost": self.total_cost,
            "remaining": self.budget_cap - self.total_cost,
            "n_calls": self.n_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "providers": dict(self.provider_counts),
            "returned_models": dict(self.returned_model_counts),
            "log_path": str(self._log_path),
        }


def output_paths(config: Config) -> tuple[Path, Path, Path]:
    root = Path(config.results_dir)
    return (
        root / f"{config.output_stem}.jsonl",
        root / f"{config.output_stem}.json",
        root / f"{config.output_stem.upper()}.md",
    )


def write_markdown(path: Path, config: Config, summary: dict[str, Any], client_summary: dict[str, Any]) -> None:
    lines = [
        "# OpenRouter Cross-Benchmark Forecast Replication",
        "",
        "Post-hoc reviewer-hardening smoke/expansion run. This is not a new primary headline unless explicitly promoted in the paper text.",
        "",
        f"Model alias: `{client_summary['model_alias']}`; model id: `{client_summary['model_id']}`.",
        f"Benchmarks: {', '.join(config.benchmarks)}; requested n/benchmark: {config.n_questions}; seed: {config.seed}.",
        f"Observed OpenRouter cost: ${client_summary['total_cost']:.4f} / ${client_summary['budget_cap']:.2f} cap.",
        f"Providers: `{json.dumps(client_summary['providers'], sort_keys=True)}`.",
        "",
        "## Summary",
        "",
        "| Split | N | init acc | final acc | collapses | C^cond | corrections | mean alpha | rho(alpha, collapse) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("overall", summary["overall"])] + list(summary["per_benchmark"].items())
    for name, s in rows:
        rho = s["spearman_alpha_vs_collapse"].get("rho")
        cond = "NA" if s["conditional_collapse_pct"] is None else f"{s['conditional_collapse_pct']:.1f}%"
        rho_s = "NA" if rho is None else f"{rho:.3f}"
        lines.append(
            f"| {name} | {s['n']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | "
            f"{s['n_collapses']} | {cond} | {s['n_corrections']} | {s['mean_alpha']:.3f} | {rho_s} |"
        )
    path.write_text("\n".join(lines) + "\n")


async def main(config: Config, pin_provider: str | None, reasoning: dict[str, Any] | None) -> None:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, summary_path, md_path = output_paths(config)
    if config.overwrite:
        for p in (jsonl_path, summary_path, md_path):
            if p.exists():
                p.unlink()
    completed = load_completed(jsonl_path)

    questions = load_questions_for_config(config)
    remaining = [q for q in questions if q.id not in completed]
    logger.info("Loaded %d questions, %d remaining", len(questions), len(remaining))

    client = OpenRouterGenerateClient(
        config.model,
        budget_cap=config.budget_cap,
        max_concurrent=config.max_concurrent,
        log_dir=results_dir / "logs",
        pin_provider=pin_provider,
        reasoning=reasoning,
    )
    rng = np.random.default_rng(config.seed)
    all_q_seeds = {q.id: int(rng.integers(0, 2**32)) for q in questions}
    q_seeds = {q.id: all_q_seeds[q.id] for q in remaining}
    question_sem = asyncio.Semaphore(max(1, config.question_concurrency))
    write_lock = asyncio.Lock()

    async def _run_one(idx: int, q) -> dict | None:
        async with question_sem:
            logger.info("[%d/%d] %s", idx, len(remaining), q.id)
            q_rng = np.random.default_rng(q_seeds[q.id])
            try:
                rec = await process_question(client, q, config, q_rng)
            except OpenRouterBudgetExceeded:
                raise
            except Exception:
                logger.exception("Question failed: %s", q.id)
                return None
            async with write_lock:
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                "  alpha=%.3f init=%s final=%s collapse=%s cost=$%.4f total=$%.4f",
                rec["alpha"]["mean"],
                rec["standard_debate"]["initial_correct"],
                rec["standard_debate"]["final_correct"],
                rec["standard_debate"]["collapsed"],
                rec["total_cost"],
                client.total_cost,
            )
            return rec

    try:
        await asyncio.gather(*[_run_one(i, q) for i, q in enumerate(remaining, start=1)])
    finally:
        await client.aclose()

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    client_summary = client.summary()
    summary = {
        "config": {
            "backend": "openrouter",
            "model": config.model,
            "model_id": client.model_id,
            "benchmarks": config.benchmarks,
            "n_questions": config.n_questions,
            "seed": config.seed,
            "n_agents": config.n_agents,
            "n_rounds": config.n_rounds,
            "question_concurrency": config.question_concurrency,
            "max_concurrent": config.max_concurrent,
            "pin_provider": pin_provider,
            "question_ids_file": config.question_ids_file,
        },
        "openrouter_usage": client_summary,
        "summary": summarize(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown(md_path, config, summary["summary"], client_summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {jsonl_path}, {summary_path}, {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa", "truthfulqa"])
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--model", default="mistral-small-4")
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--question_concurrency", type=int, default=1)
    parser.add_argument("--output_stem", default="cross_benchmark_openrouter_replication")
    parser.add_argument("--question_ids_file", default=None)
    parser.add_argument("--pin_provider", default=None)
    parser.add_argument("--reasoning_exclude", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reasoning = {"exclude": True} if args.reasoning_exclude else None
    cfg = Config(
        model=args.model,
        benchmarks=args.benchmarks,
        n_questions=args.n_questions,
        seed=args.seed,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        budget_cap=args.budget,
        max_concurrent=args.max_concurrent,
        question_concurrency=args.question_concurrency,
        output_stem=args.output_stem,
        question_ids_file=args.question_ids_file,
        overwrite=args.overwrite,
    )
    asyncio.run(main(cfg, args.pin_provider, reasoning))
