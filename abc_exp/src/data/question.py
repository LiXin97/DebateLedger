from dataclasses import dataclass, field
import string


@dataclass
class Question:
    id: str                    # unique identifier (benchmark_idx format)
    text: str                  # question text
    options: list[str]         # option texts
    option_labels: list[str]   # ["A", "B", "C", ...] matching len(options)
    correct_label: str         # e.g. "A"
    correct_index: int         # 0-based index into options
    benchmark: str             # "mmlu_pro", "gpqa", "gsm8k", "arc", "truthfulqa"
    metadata: dict = field(default_factory=dict)  # category, difficulty, etc.

    def format_for_prompt(self) -> str:
        """Format question with numbered options for LLM prompt."""
        lines = [self.text, ""]
        for label, opt in zip(self.option_labels, self.options):
            lines.append(f"{label}) {opt}")
        return "\n".join(lines)

    def random_wrong_label(self, rng) -> str:
        """Return a random incorrect option label."""
        wrong = [l for l in self.option_labels if l != self.correct_label]
        return rng.choice(wrong)

    def random_wrong_option(self, rng) -> tuple[str, str]:
        """Return (label, text) of a random wrong option."""
        wrong_indices = [i for i, l in enumerate(self.option_labels) if l != self.correct_label]
        idx = rng.choice(wrong_indices)
        return self.option_labels[idx], self.options[idx]
