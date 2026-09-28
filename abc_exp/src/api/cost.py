"""Cost tracking and budget enforcement for Gemini API calls."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-token pricing in USD.
PRICING = {
    "models/gemini-2.5-flash": {"input": 0.30e-6, "output": 2.50e-6},
    "models/gemini-2.5-pro": {"input": 1.25e-6, "output": 10.0e-6},
    "models/gemini-2.5-flash-lite": {"input": 0.10e-6, "output": 0.40e-6},
    "models/gemini-3-flash-preview": {"input": 0.50e-6, "output": 3.00e-6},
    "models/gemini-3-pro-preview": {"input": 2.00e-6, "output": 12.00e-6},
    "models/gemini-3.1-pro-preview": {"input": 2.00e-6, "output": 12.00e-6},
    "models/gemini-3.1-pro-preview-customtools": {"input": 2.00e-6, "output": 12.00e-6},
    "models/gemini-3.1-flash-lite-preview": {"input": 0.25e-6, "output": 1.50e-6},
    # Without "models/" prefix for flexibility.
    "gemini-2.5-flash": {"input": 0.30e-6, "output": 2.50e-6},
    "gemini-2.5-pro": {"input": 1.25e-6, "output": 10.0e-6},
    "gemini-2.5-flash-lite": {"input": 0.10e-6, "output": 0.40e-6},
    "gemini-3-flash-preview": {"input": 0.50e-6, "output": 3.00e-6},
    "gemini-3-pro-preview": {"input": 2.00e-6, "output": 12.00e-6},
    "gemini-3.1-pro-preview": {"input": 2.00e-6, "output": 12.00e-6},
    "gemini-3.1-pro-preview-customtools": {"input": 2.00e-6, "output": 12.00e-6},
    "gemini-3.1-flash-lite-preview": {"input": 0.25e-6, "output": 1.50e-6},
}


class BudgetExhaustedError(Exception):
    """Raised when the spending budget has been fully consumed."""


class CostTracker:
    """Thread/async-safe cost tracker with budget enforcement.

    Parameters
    ----------
    budget : float
        Maximum allowed spend in USD.  Set to ``float('inf')`` to disable.
    """

    def __init__(self, budget: float = 200.0) -> None:
        self._budget = budget
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._n_calls: int = 0
        self._per_model: dict[str, dict] = {}  # model -> {cost, input, output, calls}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def record(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Record token usage and return the cost (USD) for this call.

        Raises ``BudgetExhaustedError`` if the budget would be exceeded.
        """
        prices = PRICING.get(model)
        if prices is None:
            logger.warning(
                "Unknown model %r for pricing; assuming gemini-2.5-flash rates", model
            )
            prices = PRICING["models/gemini-2.5-flash"]

        cost = input_tokens * prices["input"] + output_tokens * prices["output"]

        async with self._lock:
            if self._total_cost + cost > self._budget:
                raise BudgetExhaustedError(
                    f"Budget exhausted: current ${self._total_cost:.4f} + "
                    f"this call ${cost:.4f} > cap ${self._budget:.2f}"
                )
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._n_calls += 1

            entry = self._per_model.setdefault(
                model, {"cost": 0.0, "input": 0, "output": 0, "calls": 0}
            )
            entry["cost"] += cost
            entry["input"] += input_tokens
            entry["output"] += output_tokens
            entry["calls"] += 1

        return cost

    def check_budget(self) -> None:
        """Raise ``BudgetExhaustedError`` if remaining budget is <= 0."""
        if self.remaining <= 0:
            raise BudgetExhaustedError(
                f"Budget exhausted: spent ${self._total_cost:.4f} "
                f"of ${self._budget:.2f}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def remaining(self) -> float:
        return self._budget - self._total_cost

    @property
    def n_calls(self) -> int:
        return self._n_calls

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist tracker state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "budget": self._budget,
            "total_cost": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "n_calls": self._n_calls,
            "per_model": self._per_model,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("CostTracker saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, budget: float | None = None) -> "CostTracker":
        """Load tracker state from a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON state file.
        budget : float | None
            Override the budget stored in the file.  If ``None``, the saved
            budget is reused.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        tracker = cls(budget=budget if budget is not None else data["budget"])
        tracker._total_cost = data["total_cost"]
        tracker._total_input_tokens = data["total_input_tokens"]
        tracker._total_output_tokens = data["total_output_tokens"]
        tracker._n_calls = data["n_calls"]
        tracker._per_model = data.get("per_model", {})
        logger.info("CostTracker loaded from %s (spent $%.4f)", path, tracker._total_cost)
        return tracker

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of costs so far."""
        lines = [
            f"=== Cost Summary ===",
            f"Total calls : {self._n_calls}",
            f"Input tokens : {self._total_input_tokens:,}",
            f"Output tokens: {self._total_output_tokens:,}",
            f"Total cost   : ${self._total_cost:.4f}",
            f"Budget       : ${self._budget:.2f}",
            f"Remaining    : ${self.remaining:.4f}",
        ]
        if self._per_model:
            lines.append("")
            lines.append("Per-model breakdown:")
            for model, info in sorted(self._per_model.items()):
                lines.append(
                    f"  {model}: {info['calls']} calls, "
                    f"{info['input']:,} in / {info['output']:,} out, "
                    f"${info['cost']:.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CostTracker(spent=${self._total_cost:.4f}, "
            f"remaining=${self.remaining:.4f}, calls={self._n_calls})"
        )
