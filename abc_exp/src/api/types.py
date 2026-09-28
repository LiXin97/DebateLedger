"""Data types for the Gemini API client layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerateResult:
    """Result from a single generate call."""

    text: str
    input_tokens: int
    output_tokens: int
    cost: float  # USD
    model: str
    timestamp: str  # ISO format
    metadata: dict  # caller-provided context (question_id, agent_id, etc.)
    thinking_tokens: int = 0  # should always be 0 if we disable thinking

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "model": self.model,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "thinking_tokens": self.thinking_tokens,
        }


@dataclass
class BatchItem:
    """A single item in a batch generation request."""

    contents: str | list  # prompt text or message list
    metadata: dict  # passed through to GenerateResult
    system_instruction: str | None = None
    temperature: float | None = None  # override default
