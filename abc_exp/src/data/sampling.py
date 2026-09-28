from __future__ import annotations

from collections import defaultdict

import numpy as np

from .question import Question


def stratified_sample(
    questions: list[Question],
    n: int,
    stratify_key: str = "category",
    seed: int = 42,
) -> list[Question]:
    """Stratified sample n questions, proportional to group sizes.

    If stratify_key not in metadata (e.g. for GPQA), fall back to simple
    random sample.
    """
    if n >= len(questions):
        return questions

    rng = np.random.default_rng(seed)

    # Check whether the stratify key is present in at least one question
    has_key = any(stratify_key in q.metadata for q in questions)
    if not has_key:
        indices = rng.choice(len(questions), size=n, replace=False)
        indices.sort()
        return [questions[i] for i in indices]

    # ---- Group by metadata[stratify_key] ----
    groups: dict[str, list[int]] = defaultdict(list)
    ungrouped: list[int] = []
    for i, q in enumerate(questions):
        key = q.metadata.get(stratify_key)
        if key is not None:
            groups[key].append(i)
        else:
            ungrouped.append(i)

    # Treat ungrouped questions as their own pseudo-group
    if ungrouped:
        groups["__ungrouped__"] = ungrouped

    total = len(questions)
    group_names = sorted(groups.keys())

    # ---- Proportional allocation (largest-remainder method) ----
    raw_alloc = {g: n * len(groups[g]) / total for g in group_names}
    floor_alloc = {g: int(raw_alloc[g]) for g in group_names}
    remainders = {g: raw_alloc[g] - floor_alloc[g] for g in group_names}

    allocated = sum(floor_alloc.values())
    deficit = n - allocated

    # Give the remaining slots to groups with largest fractional remainder
    sorted_by_rem = sorted(group_names, key=lambda g: remainders[g], reverse=True)
    for g in sorted_by_rem[:deficit]:
        floor_alloc[g] += 1

    # ---- Sample from each group ----
    sampled: list[int] = []
    for g in group_names:
        pool = np.array(groups[g])
        k = min(floor_alloc[g], len(pool))
        if k > 0:
            chosen = rng.choice(pool, size=k, replace=False)
            sampled.extend(chosen.tolist())

    # Sort by original index to maintain a stable ordering
    sampled.sort()
    return [questions[i] for i in sampled]
