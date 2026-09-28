"""Answer extraction and revision detection for MCQ tasks."""

from __future__ import annotations

import re


def extract_answer(text: str, valid_labels: list[str] | None = None) -> str | None:
    """Extract the answer letter from model output.

    Uses a priority list of regex patterns:
    1. "Final Answer: X"
    2. "answer is X" variants
    3. Standalone letter on a line
    4. Last valid letter in the text

    Returns None if no valid answer found.
    """
    if not text:
        return None

    if valid_labels is None:
        valid_labels = list("ABCDEFGHIJ")

    valid_set = set(valid_labels)
    text_clean = text.strip()

    # Priority 1: "Final Answer: X"
    m = re.search(r"Final\s+Answer\s*:\s*([A-J])\b", text_clean, re.IGNORECASE)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    # Priority 2: "answer is X" / "answer: X" / "choose X"
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*([A-J])\b",
        r"(?:I\s+)?(?:choose|select|pick)\s+(?:option\s+)?([A-J])\b",
        r"(?:correct\s+)?answer\s*:\s*([A-J])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()

    # Priority 3: standalone letter on a line (common in short responses)
    for line in reversed(text_clean.split("\n")):
        line = line.strip()
        if len(line) == 1 and line.upper() in valid_set:
            return line.upper()
        # Also match "A." or "(A)" or "**A**"
        m = re.match(r"^[\(\*]*([A-J])[\)\.\*]*$", line, re.IGNORECASE)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()

    # Priority 4: last valid letter in text
    all_letters = re.findall(r"\b([A-J])\b", text_clean)
    for letter in reversed(all_letters):
        if letter.upper() in valid_set:
            return letter.upper()

    return None


def detect_revision(original_answer: str, probe_answer: str | None) -> bool:
    """Return True if the agent revised its answer after a probe.

    If probe_answer is None (extraction failed), assume no revision.
    """
    if probe_answer is None:
        return False
    return original_answer.upper() != probe_answer.upper()
