"""B-contribution monotonicity pilot harness (frozen pre-data).

Pre-registration:
- Amendment 5 v2 (`abc_exp/AMENDMENT_5_PILOT_EXTEND.md`) at SHA `204193c`
- Probe ladder (`abc_exp/src/gates/probe_ladder_v1.json`), sha256 verified at load
- A4 SHA `73cec9d`, A6 SHA `256ad21`
- Question pool (parent `abc_exp/results/high_fr_question_ids.json`, the canonical N=200 high-FR pool the A6 cohort headline uses; per Deviation 6 2026-04-26 — was previously the worktree-local `abc_exp/data/high_fr_question_ids.json` stale artifact, now removed), sha256 verified at load

Usage:
    python -m abc_exp.scripts.B_pilot_monotonicity \\
        --models Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.6-27B-FP8 \\
        --output abc_exp/results/B_pilot_monotonicity.json

Decoding: greedy (T=0, top_p=1, max_tokens=64). Per-cell n=50.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from itertools import permutations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abc_exp.src.alpha.revision import detect_revision, extract_answer  # noqa: E402
from abc_exp.src.data.loader import load_benchmark  # noqa: E402

# Pre-registration constants. DO NOT EDIT POST-DATA.
A4_SHA = "73cec9d"
A5_SHA = "204193c"
A6_SHA = "256ad21"
LADDER_SHA256 = "eef635b0130e0c13f0c25bfb64eded44e08317996939cd8b36bc34974f876228"
POOL_SHA256 = "05d3f5b98e3b375f9ad10398bf4f0f14b2cd0bdbef575e447927148c47c41fef"  # D6 follow-up: canonical pool hash (was stale 0f1e6e… from worktree-local artifact, now repointed to abc_exp/results/high_fr_question_ids.json shipped into worktree branch — completes D6 dd1bb00 intent)
N_PER_CELL = 50
PILOT_MODELS_FROZEN = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B-FP8",
)


def _verify_artifacts() -> tuple[dict, list[str]]:
    """Load + hash-verify ladder JSON and pool JSON; abort on mismatch."""
    ladder_path = REPO_ROOT / "abc_exp/src/gates/probe_ladder_v1.json"
    # Deviation 6 (2026-04-26): repointed from worktree-local `abc_exp/data/high_fr_question_ids.json` (stale artifact, zero-overlap with canonical first-50) to parent `abc_exp/results/high_fr_question_ids.json` (the canonical pool A5 mandates and the A6 cohort headline uses). Worktree stale file `git rm`'d.
    pool_path = REPO_ROOT / "abc_exp/results/high_fr_question_ids.json"

    with open(ladder_path) as f:
        ladder = json.load(f)
    rungs_only = [[r["k"], r["name"], r["text"]] for r in ladder["rungs"]]
    canonical = json.dumps(rungs_only, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != LADDER_SHA256:
        raise RuntimeError(f"Ladder sha256 mismatch: got {actual}, expected {LADDER_SHA256}")

    pool_bytes = pool_path.read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest() != POOL_SHA256:
        raise RuntimeError(
            f"Pool sha256 mismatch: got {hashlib.sha256(pool_bytes).hexdigest()}, expected {POOL_SHA256}"
        )
    pool = json.loads(pool_bytes.decode("utf-8"))[:N_PER_CELL]
    return ladder, pool


def _format_mcq_prompt(q) -> str:
    lines = [q.text, ""]
    for label, opt in zip(q.option_labels, q.options):
        lines.append(f"{label}. {opt}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option.")
    return "\n".join(lines)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)

    def _ranks(v):
        idx = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def _spearman_exact_perm_p(observed_rho: float, k: int = 9) -> float:
    """Two-sided exact permutation p, then converted to one-sided upper-tail."""
    base = list(range(k))
    count_ge = 0
    total = 0
    for perm in permutations(base):
        rho = _spearman_rho(base, list(perm))
        if rho >= observed_rho - 1e-12:
            count_ge += 1
        total += 1
    return count_ge / total


def _page_L_and_p(matrix: list[list[float]]) -> tuple[float, float, float | None]:
    """Page's L and one-sided p. matrix shape: (N_subjects, k_treatments).

    Returns (L, p_normal_approx, p_mc_or_None).
    Per stat-rigor op-note: also compute Monte Carlo p (n=10^6 permutation
    samples) when normal-approx one-sided p is in [0.04, 0.06] (boundary).
    SE on tail prob ~0.05 at n=1e6 is ~2.2e-4, ample for the discrimination.
    """
    N = len(matrix)
    k = len(matrix[0])

    def _ranks_within(row: list[float]) -> list[float]:
        idx = sorted(range(k), key=lambda i: row[i])
        ranks = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and row[idx[j + 1]] == row[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for kk in range(i, j + 1):
                ranks[idx[kk]] = avg
            i = j + 1
        return ranks

    rank_matrix = [_ranks_within(row) for row in matrix]
    col_sums = [sum(rank_matrix[r][c] for r in range(N)) for c in range(k)]
    L = sum((c + 1) * col_sums[c] for c in range(k))

    mu = N * k * (k + 1) ** 2 / 4.0
    var = N * (k ** 3 - k) ** 2 / (144.0 * (k - 1))
    z = (L - mu) / math.sqrt(var) if var > 0 else 0.0
    p_norm = 0.5 * math.erfc(z / math.sqrt(2))

    p_exact: float | None = None
    if 0.04 <= p_norm <= 0.06:
        # Boundary case: exact-enumeration over per-subject permutations is
        # (k!)**N which blows up at k=9, N=3 (4.8e16). Use Monte Carlo with
        # 10**6 samples; SE on a tail prob of ~0.05 is sqrt(0.05*0.95/1e6) ~
        # 2.2e-4 -- ample for the [0.04, 0.06] discrimination per stat-rigor
        # op-note (5 Sep 2026 sign-off, A5v2 thread).
        import random
        rng = random.Random(0xA5BCD)  # deterministic
        all_perms = list(permutations(range(1, k + 1)))
        n_perms = len(all_perms)
        n_mc = 1_000_000
        col_weight = list(range(1, k + 1))
        count_ge = 0
        for _ in range(n_mc):
            cs = [0.0] * k
            for _r in range(N):
                ranks = all_perms[rng.randrange(n_perms)]
                for c in range(k):
                    cs[c] += ranks[c]
            L_perm = sum(col_weight[c] * cs[c] for c in range(k))
            if L_perm >= L - 1e-12:
                count_ge += 1
        p_exact = count_ge / n_mc

    return L, p_norm, p_exact


def _mcnemar_one_sided_p(b: int, c: int) -> float:
    """One-sided exact McNemar p for paired binary outcomes.

    b = count of (k=0: no flip, k=8: flip) -- F3-favoring discordant pairs
    c = count of (k=0: flip,    k=8: no flip)
    Tests H1: rate at k=8 > rate at k=0.
    """
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for k in range(b, n + 1):
        p += math.comb(n, k) * (0.5 ** n)
    return p


def _binomial_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0


def run_model(model_id: str, ladder: dict, qids_subset: list[str], questions_by_id: dict, *, dry_run: bool = False):
    """Run pilot for one model. Returns per_model dict.

    Real run: requires vLLM. Dry run: emits zeros (for harness wiring tests).
    """
    rungs = ladder["rungs"]
    n_q = len(qids_subset)
    flip_matrix = [[False] * n_q for _ in range(len(rungs))]  # (k, q)
    r0_letters: list[str | None] = [None] * n_q

    if not dry_run:
        from vllm import LLM, SamplingParams  # noqa: E501
        # B-pilot 2026-04-26: enforce_eager=True after CUDA OOM at cudagraph
        # capture peak (46.44 GiB / 47.38 GiB on GPU 2). max_model_len=8192
        # added after KV-budget bust on Q3.6-27B-FP8 (full 262144 needed
        # 16.17 GiB KV vs 9.79 GiB available; MMLU-Pro stems <3K + max_tokens=64
        # leaves >2x safety margin). Decoding params unchanged.
        llm = LLM(model=model_id, tensor_parallel_size=1, dtype="auto", gpu_memory_utilization=0.85, enforce_eager=True, max_model_len=8192)
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

        prompts_r0 = [_format_mcq_prompt(questions_by_id[qid]) for qid in qids_subset]
        outs = llm.generate(prompts_r0, sp)
        for i, out in enumerate(outs):
            r0_letters[i] = extract_answer(out.outputs[0].text)

        for ki, rung in enumerate(rungs):
            prompts_r1 = []
            keep_idx = []
            for i, qid in enumerate(qids_subset):
                if r0_letters[i] is None:
                    continue
                prompt_r1 = (
                    prompts_r0[i]
                    + "\n\nMy answer: " + r0_letters[i]
                    + "\n\n" + rung["text"]
                    + "\n\nFinal Answer:"
                )
                prompts_r1.append(prompt_r1)
                keep_idx.append(i)
            outs1 = llm.generate(prompts_r1, sp)
            for j, out in enumerate(outs1):
                i = keep_idx[j]
                r1_letter = extract_answer(out.outputs[0].text)
                flip_matrix[ki][i] = detect_revision(r0_letters[i], r1_letter)

    p_k = [sum(row) / n_q for row in flip_matrix]
    rho = _spearman_rho(list(range(len(rungs))), p_k)
    perm_p = _spearman_exact_perm_p(rho, k=len(rungs))
    max_inv = max((p_k[k] - p_k[k + 1] for k in range(len(rungs) - 1)), default=0.0)
    f3_floor = p_k[-1] - p_k[0]

    b = sum(1 for i in range(n_q) if (not flip_matrix[0][i]) and flip_matrix[-1][i])
    c = sum(1 for i in range(n_q) if flip_matrix[0][i] and (not flip_matrix[-1][i]))
    mcnemar_p = _mcnemar_one_sided_p(b, c)

    return {
        "model": model_id,
        "p_k": p_k,
        "n_per_k": [n_q] * len(rungs),
        "spearman_rho_k_p": rho,
        "spearman_exact_perm_p": perm_p,
        "max_inversion_drop": max_inv,
        "f3_floor": f3_floor,
        "f3_floor_mcnemar_p_one_sided": mcnemar_p,
        "f3_floor_discordant_b_c": [b, c],
    }


def adjudicate(per_model: list[dict], page_L: float, page_p_norm: float) -> tuple[str, bool]:
    rhos = [m["spearman_rho_k_p"] for m in per_model]
    median_rho = sorted(rhos)[len(rhos) // 2]
    invs_above = sum(1 for m in per_model if m["max_inversion_drop"] > 0.10)

    pass_cond = (
        all(r >= 0.70 for r in rhos)
        and median_rho >= 0.80
        and invs_above < 2
        and page_p_norm <= 0.05
    )
    if pass_cond:
        decision = "pass"
    elif median_rho >= 0.50:
        decision = "partial"
    else:
        decision = "fail"
    if any(r < 0 for r in rhos):
        decision = "fail"

    floor_pass_models = sum(
        1
        for m in per_model
        if m["f3_floor"] >= 0.02 and m["f3_floor_mcnemar_p_one_sided"] <= 0.05
    )
    f3_usable = floor_pass_models >= 2
    return decision, f3_usable


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(PILOT_MODELS_FROZEN))
    p.add_argument("--output", default=str(REPO_ROOT / "abc_exp/results/B_pilot_monotonicity.json"))
    p.add_argument("--dry-run", action="store_true", help="No GPU; emit zero p_k for harness wiring test")
    args = p.parse_args()

    if tuple(args.models) != PILOT_MODELS_FROZEN:
        raise RuntimeError(
            f"Pilot model list is FROZEN by A5 §1: {PILOT_MODELS_FROZEN}. Got: {tuple(args.models)}"
        )

    ladder, qids = _verify_artifacts()

    questions = load_benchmark("mmlu_pro")
    questions_by_id = {q.id: q for q in questions}
    missing = [qid for qid in qids if qid not in questions_by_id]
    if missing:
        raise RuntimeError(f"Missing question IDs in mmlu_pro: {missing[:5]} ...")

    per_model = []
    t_start = time.time()
    # B-pilot 2026-04-26: per-model partial-write checkpoint after PID 560694
    # crashed on model 3/3 with KV-budget bust, losing models 1+2 in-memory data.
    partial_path = Path(args.output).with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("")
    for model_id in args.models:
        m_result = run_model(model_id, ladder, qids, questions_by_id, dry_run=args.dry_run)
        per_model.append(m_result)
        with open(partial_path, "a") as pf:
            pf.write(json.dumps(m_result) + "\n")
            pf.flush()

    matrix = [m["p_k"] for m in per_model]
    page_L, page_p_norm, page_p_exact = _page_L_and_p(matrix)
    decision, f3_usable = adjudicate(per_model, page_L, page_p_norm)
    median_rho = sorted([m["spearman_rho_k_p"] for m in per_model])[len(per_model) // 2]

    out = {
        "pilot_pre_reg_sha": A5_SHA,
        "pre_reg_sha_a4": A4_SHA,
        "pre_reg_sha_a6": A6_SHA,
        "probe_ladder_sha256": LADDER_SHA256,
        "pool_sha256": POOL_SHA256,
        "models": list(args.models),
        "pool": "abc_exp/results/high_fr_question_ids.json[:50]",  # D6 repoint
        "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
        "per_model": per_model,
        "median_rho": median_rho,
        "page_L": page_L,
        "page_L_p_one_sided": page_p_norm,
        "page_L_p_one_sided_mc": page_p_exact,
        "decision": decision,
        "f3_usable": f3_usable,
        "wall_clock_s": time.time() - t_start,
        "dry_run": args.dry_run,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
