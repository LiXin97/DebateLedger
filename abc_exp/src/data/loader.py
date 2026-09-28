"""Load benchmarks from HuggingFace datasets into Question objects."""

from __future__ import annotations

import hashlib
import string

import numpy as np
from datasets import load_dataset

from .question import Question
from .sampling import stratified_sample


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_benchmark(
    name: str,
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Question]:
    """Load a benchmark by name.  Optionally subsample."""
    loaders = {
        "mmlu_pro": load_mmlu_pro,
        "gpqa": load_gpqa,
        "gsm8k": load_gsm8k,
        "arc": load_arc_challenge,
        "truthfulqa": load_truthfulqa,
        "supergpqa": load_supergpqa,
        "math_l5": load_math_l5,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {sorted(loaders)}"
        )
    questions = loaders[name]()
    if n_samples and n_samples < len(questions):
        return stratified_sample(questions, n_samples, seed=seed)
    return questions


# ---------------------------------------------------------------------------
# MMLU-Pro  (TIGER-Lab/MMLU-Pro, split="test")
# ---------------------------------------------------------------------------

def load_mmlu_pro() -> list[Question]:
    """Load MMLU-Pro (10-way MCQ, variable option count)."""
    print("Loading MMLU-Pro …")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    questions: list[Question] = []
    for i, row in enumerate(ds):
        opts = list(row["options"])
        n_opts = len(opts)
        labels = list(string.ascii_uppercase[:n_opts])
        answer_letter = row["answer"]          # e.g. "A"
        answer_idx = row["answer_index"]       # 0-based
        questions.append(Question(
            id=f"mmlu_pro_{i}",
            text=row["question"],
            options=opts,
            option_labels=labels,
            correct_label=answer_letter,
            correct_index=answer_idx,
            benchmark="mmlu_pro",
            metadata={"category": row["category"]},
        ))
    print(f"  → loaded {len(questions)} MMLU-Pro questions")
    return questions


# ---------------------------------------------------------------------------
# MATH, Level-5 test problems (EleutherAI/hendrycks_math); free-form answers, no options
# ---------------------------------------------------------------------------

def last_boxed(text: str):
    """Content of the last \\boxed{...} (balanced braces) in `text`, or None."""
    i = text.rfind("\\boxed")
    while i != -1:
        j = text.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(text):
                depth += (text[k] == "{") - (text[k] == "}")
                if depth == 0:
                    return text[j + 1:k].strip()
                k += 1
        i = text.rfind("\\boxed", 0, i)
    return None


def load_math_l5() -> list[Question]:
    print("Loading MATH level 5 …")
    questions: list[Question] = []
    for cfg in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for i, row in enumerate(ds):
            if row["level"] != "Level 5":
                continue
            gold = last_boxed(row["solution"])
            if not gold:
                continue
            questions.append(Question(id=f"math_l5_{cfg}_{i}", text=row["problem"], options=[], option_labels=[],
                                      correct_label=gold, correct_index=-1, benchmark="math_l5",
                                      metadata={"category": row["type"]}))
    print(f"  → loaded {len(questions)} MATH level-5 problems")
    return questions


# ---------------------------------------------------------------------------
# SuperGPQA  (m-a-p/SuperGPQA, split="train"; graduate-level MCQ, up to 10 options)
# ---------------------------------------------------------------------------

def load_supergpqa() -> list[Question]:
    """Load SuperGPQA. Stratification key is the discipline; ids use the dataset uuid."""
    print("Loading SuperGPQA …")
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    questions: list[Question] = []
    for row in ds:
        opts = list(row["options"])
        if not 2 <= len(opts) <= 10:
            continue
        labels = list(string.ascii_uppercase[:len(opts)])
        letter = row["answer_letter"]
        if letter not in labels:
            continue
        questions.append(Question(
            id=f"supergpqa_{row['uuid']}",
            text=row["question"],
            options=opts,
            option_labels=labels,
            correct_label=letter,
            correct_index=labels.index(letter),
            benchmark="supergpqa",
            metadata={"category": row["discipline"], "field": row["field"],
                      "difficulty": row["difficulty"], "is_calculation": row["is_calculation"]},
        ))
    print(f"  → loaded {len(questions)} SuperGPQA questions")
    return questions


# ---------------------------------------------------------------------------
# GPQA  (Idavidrein/gpqa, config="gpqa_diamond", split="train")
# ---------------------------------------------------------------------------

def _deterministic_seed_for(text: str) -> int:
    """Derive a stable integer seed from a string (for per-question shuffle)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def load_gpqa() -> list[Question]:
    """Load GPQA Diamond (4-way MCQ, options shuffled deterministically)."""
    print("Loading GPQA Diamond …")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    questions: list[Question] = []
    labels_4 = list(string.ascii_uppercase[:4])
    for i, row in enumerate(ds):
        raw_options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        correct_text = raw_options[0]

        # Deterministic shuffle keyed on the question text
        perm_rng = np.random.default_rng(_deterministic_seed_for(row["Question"]))
        perm = perm_rng.permutation(4)
        shuffled_opts = [raw_options[j] for j in perm]
        correct_idx = int(np.where(perm == 0)[0][0])

        questions.append(Question(
            id=f"gpqa_{i}",
            text=row["Question"],
            options=shuffled_opts,
            option_labels=labels_4,
            correct_label=labels_4[correct_idx],
            correct_index=correct_idx,
            benchmark="gpqa",
            metadata={},
        ))
    print(f"  → loaded {len(questions)} GPQA Diamond questions")
    return questions


# ---------------------------------------------------------------------------
# GSM8K  (openai/gsm8k, config="main", split="test")
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer after the '####' marker."""
    if "####" in answer_text:
        raw = answer_text.split("####")[-1].strip()
        # Remove commas from numbers like "1,234"
        return raw.replace(",", "")
    return answer_text.strip()


def load_gsm8k() -> list[Question]:
    """Load GSM8K (free-form math, not MCQ)."""
    print("Loading GSM8K …")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions: list[Question] = []
    for i, row in enumerate(ds):
        full_answer = row["answer"]
        numeric = _extract_gsm8k_answer(full_answer)
        questions.append(Question(
            id=f"gsm8k_{i}",
            text=row["question"],
            options=[],
            option_labels=[],
            correct_label=numeric,
            correct_index=-1,  # not applicable for free-form
            benchmark="gsm8k",
            metadata={"answer_text": full_answer},
        ))
    print(f"  → loaded {len(questions)} GSM8K questions")
    return questions


# ---------------------------------------------------------------------------
# ARC-Challenge  (allenai/ai2_arc, config="ARC-Challenge", split="test")
# ---------------------------------------------------------------------------

def load_arc_challenge() -> list[Question]:
    """Load ARC-Challenge (variable 3–5 way MCQ)."""
    print("Loading ARC-Challenge …")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    questions: list[Question] = []
    for i, row in enumerate(ds):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer_key = row["answerKey"]

        # answerKey is sometimes a digit string ("1", "2", …) instead of a letter
        if answer_key.isdigit():
            correct_idx = int(answer_key) - 1
            if correct_idx < len(labels):
                answer_key = labels[correct_idx]
            else:
                answer_key = labels[0]
                correct_idx = 0
        else:
            correct_idx = labels.index(answer_key) if answer_key in labels else 0

        questions.append(Question(
            id=f"arc_{i}",
            text=row["question"],
            options=texts,
            option_labels=labels,
            correct_label=answer_key,
            correct_index=correct_idx,
            benchmark="arc",
            metadata={},
        ))
    print(f"  → loaded {len(questions)} ARC-Challenge questions")
    return questions


# ---------------------------------------------------------------------------
# TruthfulQA  (truthfulqa/truthful_qa, config="multiple_choice", split="validation")
# ---------------------------------------------------------------------------

def load_truthfulqa() -> list[Question]:
    """Load TruthfulQA MC1 (variable MCQ, one correct answer)."""
    print("Loading TruthfulQA …")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    questions: list[Question] = []
    for i, row in enumerate(ds):
        choices = list(row["mc1_targets"]["choices"])
        labels_int = list(row["mc1_targets"]["labels"])

        n_opts = len(choices)
        opt_labels = list(string.ascii_uppercase[:n_opts])

        # Find the correct option (label == 1)
        correct_idx = labels_int.index(1) if 1 in labels_int else 0

        questions.append(Question(
            id=f"truthfulqa_{i}",
            text=row["question"],
            options=choices,
            option_labels=opt_labels,
            correct_label=opt_labels[correct_idx],
            correct_index=correct_idx,
            benchmark="truthfulqa",
            metadata={},
        ))
    print(f"  → loaded {len(questions)} TruthfulQA questions")
    return questions
