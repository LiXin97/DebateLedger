"""Recompute the headline ρ on the N=17 cohort per Deviation 5 (2026-04-26) + Deviation 3 + Deviation 2 + A6 §1 + A4 §2.5/§3/§4.

Deviation 5 supersedes D3's cohort spec (N=18 → N=17): qwen3-32b excluded
post-cohort-build as a sealed-paper-vintage artifact on the wrong question
pool (599 qids vs the post-A6 high-FR N=200 pool, 7-qid overlap). Cohort-
spec defect, not wrapper-row-loss. Qwen family transitions 7-member →
6-member; G stays = 8 (count of clusters unchanged, DoF G-1=7 preserved).
Gate thresholds, subgroup operationalization, ladder sha256, and A5
untouched. Filename retained as `recompute_headline_n18.py` for git
history continuity; the realized cohort is N=17.

Deviation 3 (deepseek-v4-pro excluded post-launch due to OpenRouter
throughput ceiling, ≈0.11 r/min sustained at c=8) and Deviation 2
(kimi-k2.6 excluded) and A6 §1 chain unchanged otherwise. n<5
directional-only treatment for MoE n=2 and reasoning n=2 still applies
per D3.

Schema-aligned with `abc_exp/results/cohort_n18_headline.json` (filename
retained; payload reflects N=17).

Inputs (all required, fail-fast if missing):
  --traces-dir   directory containing 17 `debate_traces_*.jsonl` files
  --alpha-dir    directory containing 17 α-probe JSONLs (or a single alpha_table.json)
  --cohort-yaml  abc_exp/scripts/cohort_n20_models.yaml (A6 N=20 spec; D2 drops kimi-k2.6 + D3 drops deepseek-v4-pro + D5 drops qwen3-32b at FAMILY_BY_MODEL filter — yaml supplies access paths/snapshots only, n17→n20→n19→n18→n17 chain is intentional)
  --pre-reg-sha  SHA of Deviation 5 commit (default: see PRE_REG_SHA_DEFAULT — D5 supersedes D3 § cohort + A6 §1.2/§3.1)
  --out          output JSON path (default: abc_exp/results/cohort_n18_headline.json — filename retained)

Stat-rigor scope: this script is the SINGLE adjudication run for §3. Re-running
on a corrected pipeline requires a further Amendment per §3 of A4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy import stats


PRE_REG_SHA_DEFAULT = "d89781d"  # Deviation 4 (omnibus partial-row, 5 stage-2 alpha-probe lanes, -61/9000 wrapper-row-loss; supersedes D3 9f285ba). D7 (abfd632) drops 3 AWQ lanes (N=17→14, G=8→7) — N/G enumerated in load_cohort, not via SHA. A6 baseline = 256ad21.
AMENDMENT_VERSION = "AMENDMENT_6+DEVIATION_2_v1+DEVIATION_3_v1+DEVIATION_4_v1+DEVIATION_5_v1+DEVIATION_7_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

# §1.3 LOCKED family partition (A4 §1.3 + A6 §1.3 + Deviation 2 2026-04-26: kimi-k2.6 excluded, Moonshot family dropped, G=9→8 + Deviation 3 2026-04-26: deepseek-v4-pro excluded, DeepSeek family transitions 2-member→singleton, G=8 unchanged + Deviation 5 2026-04-26: qwen3-32b excluded as wrong-pool sealed-paper-vintage artifact, Qwen family 7→6, G=8 unchanged, N=18→17 + Deviation 7 2026-04-26: 3 AWQ-packaged lanes (llama-3.3-70b-awq + mistral-large-awq + mistral-small-4-119b-2603) dropped re-affirming D1's AWQ kernel exclusion on A6000 hardware, Mistral family fully dropped, Meta family 2→1 singleton, N=17→14, G=8→7). DO NOT mutate post-hoc.
FAMILY_BY_MODEL: dict[str, str] = {
    # Original N=9
    "sonnet-4.5": "Anthropic",
    "gpt-4o-mini": "OpenAI",
    "gpt-5.4-mini": "OpenAI",
    "gemini-3-flash": "Google",
    "phi-4-mini": "Phi",
    "qwen3-4b": "Qwen",
    "qwen3-8b": "Qwen",
    # qwen3-32b EXCLUDED per Deviation 5 (2026-04-26) — sealed-paper-vintage artifact on wrong question pool (599 qids vs post-A6 N=200 high-FR pool, 7-qid overlap); cohort-spec defect, not wrapper-row-loss. Cohort: N=18 → N=17. Qwen family: 7 → 6. G=8 unchanged.
    "llama-3.1-8b": "Meta",
    # A4 §1.2 additions (V3.2 dropped per A6, V4-Flash restored)
    "qwen3.6-27b-fp8": "Qwen",
    "qwen3.6-35b-a3b-fp8": "Qwen",
    "gemma-4-31b-it-awq": "Google",
    # llama-3.3-70b-awq EXCLUDED per Deviation 1 (2026-04-19; AWQ kernel degenerate output on A6000 hardware) and re-affirmed by Deviation 7 (2026-04-26). Meta family transitions 2 → 1 (singleton llama-3.1-8b only).
    # mistral-large-awq EXCLUDED per Deviation 1 (2026-04-19; AWQ kernel degenerate output on A6000 hardware) and re-affirmed by Deviation 7 (2026-04-26). With mistral-small-4-119b-2603 also excluded (next), Mistral family is fully dropped.
    # deepseek-v4-pro EXCLUDED per Deviation 3 (2026-04-26) — OpenRouter throughput ceiling, DeepSeek family transitions to singleton
    "deepseek-v4-flash": "DeepSeek",  # A6 §1.2 restoration (HF createdAt 2026-04-22)
    # kimi-k2.6 EXCLUDED per Deviation 2 (2026-04-26) — throughput ceiling, Moonshot family dropped entirely
    # A6 §1.2 size-matched 2026 additions (N=17 → N=20 → N=19 after D2 → N=18 after D3 → N=17 after D5 → N=14 after D7)
    "qwen3.5-4b": "Qwen",
    "qwen3.5-9b": "Qwen",
    # mistral-small-4-119b-2603 EXCLUDED per Deviation 7 (2026-04-26) — extends D1 by analogy: same cyankiwi AWQ-4bit packaging family that failed on A6000 hardware; never load-tested, pre-flight load-test precluded by hardware constraints in D5.1 window. Mistral family fully dropped (G: 8 → 7).
}

QWEN_MODELS = {m for m, f in FAMILY_BY_MODEL.items() if f == "Qwen"}
MOE_MODELS = {  # §4.2 locked subgroup (A6 §2.2: V3.2 → V4-Flash; D2: kimi removed; D3: V4-Pro removed → n=2)
    "deepseek-v4-flash", "qwen3.6-35b-a3b-fp8",
}
REASONING_MODELS = {  # §4.5 locked subgroup (A6 §2.2: V3.2 dropped, V4-Flash added; D2: kimi removed; D3: V4-Pro removed → n=2)
    "deepseek-v4-flash", "gpt-5.4-mini",
}


@dataclass
class CohortPoint:
    model: str
    family: str
    alpha_tot: float
    c_cond: float
    c_cond_wilson_halfwidth: float


def seed_from_sha(pre_reg_sha: str) -> int:
    """§2.5: bootstrap RNG seed = int(pre_reg_sha[:8], 16)."""
    return int(pre_reg_sha[:8], 16)


def fingerprint_inputs(paths: Iterable[Path]) -> str:
    """SHA-256 over concatenated trace + α JSON files, sorted by filename. §3."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


SEALED_N9_NAME_MAP: dict[str, str] = {
    # alpha_decomp_headline_n9.json uses pretty-printed names; FAMILY_BY_MODEL
    # uses lowercase keys. Qwen3-32B intentionally NOT mapped (D5-EXCLUDED).
    "Sonnet 4.5": "sonnet-4.5",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-5.4-mini": "gpt-5.4-mini",
    "Gemini 3-flash": "gemini-3-flash",
    "Phi-4-mini": "phi-4-mini",
    "Qwen3-4B": "qwen3-4b",
    "Llama-3.1-8B": "llama-3.1-8b",
    "Qwen3-8B": "qwen3-8b",
}

# conditional_collapse_table.json (sealed N=11 reference) uses Anthropic/Google
# vendor-prefixed names for the 2 sealed models we re-use; map for Wilson HW.
CCT_NAME_MAP: dict[str, str] = {
    "Sonnet 4.5": "Claude Sonnet 4.5",
    # other 7 sealed models use the same key in both files
}

NEW_LANE_FILES: dict[str, tuple[str, str]] = {
    # model_key -> (sa_causal_*.jsonl basename, debate_traces_*.jsonl basename)
    # Naming asymmetry on gemma is intentional — debate trace was collected
    # before the AWQ suffix landed in the harness lane name; α-probe ran on
    # the AWQ build itself.
    "qwen3.6-27b-fp8":      ("sa_causal_vllm_qwen3.6-27b-fp8.jsonl",     "debate_traces_vllm_qwen3.6-27b-fp8.jsonl"),
    "qwen3.6-35b-a3b-fp8":  ("sa_causal_vllm_qwen3.6-35b-a3b-fp8.jsonl", "debate_traces_vllm_qwen3.6-35b-a3b-fp8.jsonl"),
    "gemma-4-31b-it-awq":   ("sa_causal_vllm_gemma-4-31b-it-awq.jsonl",  "debate_traces_vllm_gemma-4-31b-it.jsonl"),
    "deepseek-v4-flash":    ("sa_causal_router_deepseek-v4-flash.jsonl", "debate_traces_openrouter_deepseek-v4-flash.jsonl"),
    "qwen3.5-4b":           ("sa_causal_vllm_qwen3.5-4b.jsonl",          "debate_traces_vllm_qwen3.5-4b.jsonl"),
    "qwen3.5-9b":           ("sa_causal_vllm_qwen3.5-9b.jsonl",          "debate_traces_vllm_qwen3.5-9b.jsonl"),
}

C_COND_WILSON_HALFWIDTH_GATE = 0.20  # parent §2 — drop any model whose Wilson 95% half-width on C^cond exceeds this on the proportion scale.


def _mode(letters: list[str]) -> str:
    """Plurality mode over a 3-agent answer list, ignoring None entries (parse failures).
    Ties broken lexicographically (deterministic). Returns "" if no non-None letters."""
    clean = [x for x in (letters or []) if x is not None]
    if not clean:
        return ""
    counts: dict[str, int] = {}
    for x in clean:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    return sorted(k for k, v in counts.items() if v == max_count)[0]


def _conditional_collapse_from_debates(path: Path, endpoint: str = "stored") -> tuple[int, int]:
    """Per sealed paper §3.2: C^cond = #{init_majority_correct AND final_majority != init_majority} / #{init_majority_correct}.

    Returns (n_init_correct, n_cond_collapse). The per-debate JSONL stores
    `initial_answers` (list of 3 letters), `final_answers` or `majority_answer`
    (final majority), and `correct_label`. We derive the initial majority from
    `initial_answers` rather than trusting any cached field, so this is robust
    across new-vintage and sealed-vintage harness variants. `outcome.collapse`
    in the new-vintage row is per-AGENT collapse (any agent flipped from
    correct to wrong), NOT per-debate conditional collapse — DO NOT substitute.

    None entries in `initial_answers` / `final_answers` (parse failures from
    the agent harness) are filtered before computing the per-round majority;
    debates with no non-None initial answer are skipped entirely.

    endpoint="stored" (default, original behaviour): stored `majority_answer` (Round 4 of the
    post-A6 harness, ties by first-seen answer), falling back to the mode of `final_answers`.
    endpoint="r3_alpha" (the paper's endpoint): alphabetic-tie mode of the Round 3 agent
    answers (`round_traces[2]`), the convention of the sealed rows and of the paper text.
    """
    n_init_correct = 0
    n_cond_coll = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            init_ans = d.get("initial_answers")
            correct = d.get("correct_label")
            if not init_ans or not correct:
                continue
            init_maj = _mode(init_ans)
            if not init_maj or init_maj != correct:
                continue
            n_init_correct += 1
            if endpoint == "r3_alpha":
                final_maj = _mode([x.get("answer") for x in d["round_traces"][2]])
            else:
                final_maj = d.get("majority_answer") or _mode(d.get("final_answers") or [])
            if final_maj and final_maj != init_maj:
                n_cond_coll += 1
    return n_init_correct, n_cond_coll


def _wilson_halfwidth_proportion(p: float, n: int) -> float:
    """Two-sided 95% normal-approximation half-width on a binomial proportion (proportion scale 0..1)."""
    if n <= 0:
        return float("nan")
    return float(1.96 * np.sqrt(p * (1 - p) / n))


def _sealed_n9_wilson_hw_from_cct(cct_by_pretty: dict, pretty: str) -> float | None:
    """Pull (cond_lo, cond_hi) for a sealed model from conditional_collapse_table.json
    and return Wilson half-width on the PROPORTION scale (the table stores percent)."""
    key = CCT_NAME_MAP.get(pretty, pretty)
    row = cct_by_pretty.get(key)
    if row is None:
        return None
    lo = float(row["cond_lo"]) / 100.0
    hi = float(row["cond_hi"]) / 100.0
    return (hi - lo) / 2.0


def load_cohort(traces_dir: Path, alpha_dir: Path, cohort_yaml: Path, ccond_endpoint: str = "stored") -> list[CohortPoint]:
    """Load N=14 (alpha_tot, C^cond) cohort points from dual-vintage artifacts.

    SEALED N=9 vintage (8 models post-D5; Q3-32B excluded at SEALED_N9_NAME_MAP):
      α_tot + C^cond pre-aggregated in `alpha_dir/alpha_decomp_headline_n9.json`
      (sealed paper artifact). Per-model Wilson HW pulled from
      `alpha_dir/conditional_collapse_table.json` if available, otherwise
      computed from the sealed C^cond proportion against a conservative n=200
      lower bound. The sealed values are authoritative for the sealed lanes:
      do NOT recompute from per-row sealed-vintage block0_alpha.jsonl, which
      uses a different per-question-per-agent schema (no `condition` axis).

    NEW POST-A6 vintage (6 models post-D7; D2/D3/D7 lanes not in NEW_LANE_FILES):
      α_tot re-aggregated from `alpha_dir/sa_causal_<vllm|router>_<model>.jsonl`
      as `mean(flip_rate)` over (qid, condition, agent_idx) rows. C^cond
      re-aggregated from `traces_dir/debate_traces_<vllm|openrouter>_<model>.jsonl`
      per-debate rows: n_cond_coll / n_init_correct, where init_majority is
      the mode-of-3 over `initial_answers` and final_majority is `majority_answer`
      (or mode-of-3 over `final_answers` as fallback).

    α_tot equivalence assumption (sa_causal_*.jsonl):
      The new-vintage row stores `flip_rate ∈ [0,1]` per (qid, cond, agent_idx),
      which is the mean of `revised` over the 8 (strength × social) probes for
      that triple. Therefore mean(flip_rate) over triples ≡ mean(flip) over
      individual probe trials, conditional on every triple having the same
      probe count (8). This is invariance-by-construction of the alpha-probe
      harness, not a numerical fit; if any future lane lands with non-uniform
      probes-per-triple the equivalence breaks and α_tot must be recomputed
      from the per-trial `probe_results` array. Surfaced as soft warning.

    EXCLUSIONS (FAMILY_BY_MODEL drops these at the §1.3 partition):
      qwen3-32b              ← D5 (sealed-paper-vintage, wrong question pool)
      llama-3.3-70b-awq      ← D1+D7 (AWQ kernel on A6000; D7 re-affirms D1)
      mistral-large-awq      ← D1+D7 (AWQ kernel on A6000; D7 re-affirms D1)
      mistral-small-4-119b-2603 ← D7 (extends D1 by analogy: same cyankiwi
                                  AWQ-4bit packaging, never load-tested)
      kimi-k2.6              ← D2 (Moonshot throughput ceiling)
      deepseek-v4-pro        ← D3 (OpenRouter throughput ceiling)

    `cohort_yaml` is read for completeness verification only (every realized
    model must appear in the manifest); FAMILY_BY_MODEL is the source of
    truth for the partition (§1.3 LOCKED). Wilson HW > 0.20 drops the model
    per parent §2 (logged to stderr, not silent).

    Wired 2026-04-26 (D4 commit-batch follow-on, post-D7-codebase-reflection-correction).
    """
    points: list[CohortPoint] = []

    # ---- Sealed N=9 vintage (8 models post-D5) ----
    sealed_path = alpha_dir / "alpha_decomp_headline_n9.json"
    if not sealed_path.exists():
        raise FileNotFoundError(
            f"sealed N=9 source not found: {sealed_path}. Required for the 8 "
            "sealed models (sonnet-4.5, gpt-{4o,5.4}-mini, gemini-3-flash, "
            "phi-4-mini, qwen3-{4b,8b}, llama-3.1-8b)."
        )
    sealed_n9 = json.loads(sealed_path.read_text())
    cct_path = alpha_dir / "conditional_collapse_table.json"
    cct_by_pretty: dict = {}
    if cct_path.exists():
        cct_by_pretty = {r["model"]: r for r in json.loads(cct_path.read_text())["table"]}

    for i, pretty in enumerate(sealed_n9["models"]):
        model_key = SEALED_N9_NAME_MAP.get(pretty)
        if model_key is None:
            # Qwen3-32B falls here (D5-EXCLUDED) — silent skip is correct.
            continue
        if model_key not in FAMILY_BY_MODEL:
            print(f"WARNING: sealed model {pretty}->{model_key} not in FAMILY_BY_MODEL; skipping.", file=sys.stderr)
            continue
        alpha_tot = float(sealed_n9["alpha_total"][i])
        c_cond_pct = float(sealed_n9["C_cond"][i])  # PERCENT scale
        c_cond_prop = c_cond_pct / 100.0
        hw_prop = _sealed_n9_wilson_hw_from_cct(cct_by_pretty, pretty)
        if hw_prop is None:
            # Fallback: assume sealed-N9 n_init_correct=100 (conservative lower bound for n_deb=200 lanes)
            hw_prop = _wilson_halfwidth_proportion(c_cond_prop, 100)
        if hw_prop > C_COND_WILSON_HALFWIDTH_GATE:
            print(f"WARNING: dropping sealed {model_key}: Wilson HW {hw_prop:.3f} > {C_COND_WILSON_HALFWIDTH_GATE}.", file=sys.stderr)
            continue
        points.append(CohortPoint(
            model=model_key,
            family=FAMILY_BY_MODEL[model_key],
            alpha_tot=alpha_tot,
            c_cond=c_cond_pct,
            c_cond_wilson_halfwidth=hw_prop,
        ))

    # ---- New post-A6 vintage (6 models post-D7) ----
    for model_key, (alpha_fn, debate_fn) in NEW_LANE_FILES.items():
        if model_key not in FAMILY_BY_MODEL:
            print(f"WARNING: new-lane {model_key} not in FAMILY_BY_MODEL; skipping.", file=sys.stderr)
            continue
        alpha_path = alpha_dir / alpha_fn
        debate_path = traces_dir / debate_fn
        if not alpha_path.exists():
            print(f"WARNING: missing α-probe file for {model_key}: {alpha_path}", file=sys.stderr)
            continue
        if not debate_path.exists():
            print(f"WARNING: missing debate-trace file for {model_key}: {debate_path}", file=sys.stderr)
            continue
        rows = _load_alpha_rows(alpha_path)
        if not rows:
            print(f"WARNING: empty α-probe rows for {model_key}; skipping.", file=sys.stderr)
            continue
        alpha_tot = float(np.mean([r[3] for r in rows]))
        n_init_correct, n_cond_coll = _conditional_collapse_from_debates(debate_path, ccond_endpoint)
        if n_init_correct == 0:
            print(f"WARNING: 0 init-majority-correct debates for {model_key}; cannot compute C^cond. Skipping.", file=sys.stderr)
            continue
        c_cond_prop = n_cond_coll / n_init_correct
        c_cond_pct = c_cond_prop * 100.0  # match sealed-N9 scale
        hw_prop = _wilson_halfwidth_proportion(c_cond_prop, n_init_correct)
        if hw_prop > C_COND_WILSON_HALFWIDTH_GATE:
            print(f"WARNING: dropping new-lane {model_key}: Wilson HW {hw_prop:.3f} > {C_COND_WILSON_HALFWIDTH_GATE} (n_init_correct={n_init_correct}, n_cond_coll={n_cond_coll}).", file=sys.stderr)
            continue
        points.append(CohortPoint(
            model=model_key,
            family=FAMILY_BY_MODEL[model_key],
            alpha_tot=alpha_tot,
            c_cond=c_cond_pct,
            c_cond_wilson_halfwidth=hw_prop,
        ))

    return points


def spearman_with_fisher_z(alphas: np.ndarray, ccond: np.ndarray) -> dict:
    """One-sided Spearman + Fisher-z 95% CI (parent §1, demoted to comparator by §2.5)."""
    rho, p_two = stats.spearmanr(alphas, ccond)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    n = len(alphas)
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    return {
        "rho": float(rho),
        "p_one_sided": float(p_one),
        "fisher_z_ci_low": float(np.tanh(z_lo)),
        "fisher_z_ci_high": float(np.tanh(z_hi)),
    }


def cluster_bootstrap_ci(
    alphas: np.ndarray,
    ccond: np.ndarray,
    families: list[str],
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> dict:
    """§2.5: family-clustered bootstrap, all-or-none within cluster, 10k resamples."""
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(families):
        by_family[f].append(i)
    family_keys = sorted(by_family.keys())
    boot_rhos = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        chosen = rng.choice(len(family_keys), size=len(family_keys), replace=True)
        idx: list[int] = []
        for k in chosen:
            idx.extend(by_family[family_keys[k]])
        if len({families[i] for i in idx}) < 2 or len(set(alphas[idx])) < 2:
            boot_rhos[b] = np.nan
            continue
        boot_rhos[b], _ = stats.spearmanr(alphas[idx], ccond[idx])
    valid = boot_rhos[~np.isnan(boot_rhos)]
    return {
        "n_resamples": n_resamples,
        "n_valid": int(len(valid)),
        "ci_low": float(np.percentile(valid, 2.5)),
        "ci_high": float(np.percentile(valid, 97.5)),
    }


def wild_cluster_bootstrap_t(
    alphas: np.ndarray,
    ccond: np.ndarray,
    families: list[str],
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> dict:
    """§2.5 secondary CI: wild-cluster bootstrap-t (Rademacher), small-G coverage.

    Implementation note: full Cameron–Gelbach–Miller wild-cluster bootstrap-t
    on a Spearman ρ is non-standard; we operationalize as Rademacher-flipped
    rank-residual resampling at the family level on the Fisher-z transform.
    Documented as such in the gate-decision doc (Task #6).
    """
    rng = np.random.default_rng(seed + 1 if seed is not None else None)
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(families):
        by_family[f].append(i)
    family_keys = sorted(by_family.keys())
    rho0, _ = stats.spearmanr(alphas, ccond)
    z0 = np.arctanh(rho0)
    boot_z = np.empty(n_resamples, dtype=float)
    a_ranks = stats.rankdata(alphas)
    c_ranks = stats.rankdata(ccond)
    a_resid = a_ranks - a_ranks.mean()
    c_resid = c_ranks - c_ranks.mean()
    for b in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=len(family_keys))
        a_b = a_ranks.copy()
        c_b = c_ranks.copy()
        for s, k in zip(signs, family_keys):
            for i in by_family[k]:
                a_b[i] = a_ranks.mean() + s * a_resid[i]
                c_b[i] = c_ranks.mean() + s * c_resid[i]
        rho_b, _ = stats.spearmanr(a_b, c_b)
        boot_z[b] = np.arctanh(np.clip(rho_b, -0.999, 0.999))
    z_lo, z_hi = np.percentile(boot_z, [2.5, 97.5])
    return {
        "n_resamples": n_resamples,
        "ci_low": float(np.tanh(z_lo)),
        "ci_high": float(np.tanh(z_hi)),
        "method": "rademacher_rank_residual_fisher_z",
    }


def adjudicate_gate(rho: float, ci_low_conservative: float) -> dict:
    """§3 three-tier gate. Conservative CI_low binds (§2.5)."""
    if rho >= 0.55 and ci_low_conservative > 0:
        tier = "Upgrade"
    elif rho < 0.40:
        tier = "Reframe"
    else:
        tier = "Robustness"
    return {
        "tier": tier,
        "rho_observed": rho,
        "ci_low_conservative": ci_low_conservative,
        "thresholds": {"upgrade_rho_min": 0.55, "reframe_rho_max": 0.40},
        "ci_low_required_at_upgrade": "> 0",
    }


def headline(cohort: list[CohortPoint], pre_reg_sha: str) -> dict:
    alphas = np.array([p.alpha_tot for p in cohort])
    ccond = np.array([p.c_cond for p in cohort])
    families = [p.family for p in cohort]
    seed = seed_from_sha(pre_reg_sha)

    primary = spearman_with_fisher_z(alphas, ccond)
    cluster_ci = cluster_bootstrap_ci(alphas, ccond, families, seed=seed)
    wild_ci = wild_cluster_bootstrap_t(alphas, ccond, families, seed=seed)

    # §2.5: conservative CI_low binds the §3 gate.
    ci_low_conservative = min(cluster_ci["ci_low"], wild_ci["ci_low"])
    gate = adjudicate_gate(primary["rho"], ci_low_conservative)

    return {
        "headline_rho": primary["rho"],
        "p_one_sided": primary["p_one_sided"],
        "fisher_z_ci": [primary["fisher_z_ci_low"], primary["fisher_z_ci_high"]],
        "family_cluster_bootstrap_ci": cluster_ci,
        "wild_cluster_bootstrap_t_ci": wild_ci,
        "ci_low_conservative": ci_low_conservative,
        "gate": gate,
    }


def subgroup(cohort: list[CohortPoint], member_set: set[str], pre_reg_sha: str) -> dict:
    sub = [p for p in cohort if p.model in member_set]
    if len(sub) < 5:
        # Deviation 2 (2026-04-26): n<5 precludes formal inference.
        # n=3 Spearman ranges over {-1,-0.5,0,0.5,1} only; min one-sided
        # exact-perm p = 1/6 ≈ 0.167. Bootstrap on n<5 is meaningless.
        # Report directional point estimate + per-model listing only.
        if len(sub) < 2:
            return {"n": len(sub), "skipped": "subgroup_too_small"}
        alphas_pe = np.array([p.alpha_tot for p in sub])
        ccond_pe = np.array([p.c_cond for p in sub])
        rho_pe, _ = stats.spearmanr(alphas_pe, ccond_pe)
        return {
            "n": len(sub),
            "models": sorted(p.model for p in sub),
            "rho_point_estimate": float(rho_pe),
            "per_model_alpha_ccond": [
                {"model": p.model, "alpha_tot": p.alpha_tot, "c_cond": p.c_cond}
                for p in sorted(sub, key=lambda x: x.model)
            ],
            "skipped": "n_below_formal_test_threshold",
        }
    alphas = np.array([p.alpha_tot for p in sub])
    ccond = np.array([p.c_cond for p in sub])
    families = [p.family for p in sub]
    seed = seed_from_sha(pre_reg_sha)
    return {
        "n": len(sub),
        "models": sorted(p.model for p in sub),
        "primary": spearman_with_fisher_z(alphas, ccond),
        "family_cluster_bootstrap_ci": cluster_bootstrap_ci(
            alphas, ccond, families, seed=seed
        ),
    }


def lofo(cohort: list[CohortPoint], pre_reg_sha: str) -> list[dict]:
    """§4.4 leave-one-family-out on the §1.3 partition (9 folds)."""
    families = sorted({p.family for p in cohort})
    out = []
    for held_out in families:
        sub = [p for p in cohort if p.family != held_out]
        alphas = np.array([p.alpha_tot for p in sub])
        ccond = np.array([p.c_cond for p in sub])
        sub_families = [p.family for p in sub]
        rho, p2 = stats.spearmanr(alphas, ccond)
        out.append({
            "held_out_family": held_out,
            "n_remaining": len(sub),
            "rho": float(rho),
            "family_cluster_bootstrap_ci": cluster_bootstrap_ci(
                alphas, ccond, sub_families,
                seed=seed_from_sha(pre_reg_sha),
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Deviation 4 (omnibus partial-row sensitivity, 2026-04-26).
# Wrapper-level row drops affected ≥3 cohort lanes (Q3.5-9B 1773/1800,
# Q3.5-4B 1790/1800, etc.). Uniform sensitivity protocol below; one call
# per partial lane in main(). No code path runs unless realized rows < 1800.
# ---------------------------------------------------------------------------

ALPHA_FILENAME_PREFIXES = ("sa_causal_vllm_", "sa_causal_router_", "sa_causal_")
DEFAULT_CONDITIONS = ("default", "anti_social", "anti_argument")
DEFAULT_AGENT_INDICES = (0, 1, 2)


def _wilson_halfwidth_normal_approx(p: float, n: int, z: float = 1.96) -> float:
    """Two-sided 95% normal-approximation half-width on a binomial proportion."""
    if n <= 0:
        return float("nan")
    return float(z * np.sqrt(p * (1 - p) / n))


def _model_name_from_alpha_path(path: Path) -> str:
    name = path.stem
    for prefix in ALPHA_FILENAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _load_alpha_rows(alpha_jsonl: Path) -> list[tuple[str, str, int, float]]:
    """Load (qid, condition, agent_idx, flip_rate) tuples from an α-probe JSONL.

    Schema: per-(qid, condition, agent_idx) row with `flip_rate ∈ [0, 1]`
    aggregating the 8 probe families for that triple.
    """
    rows: list[tuple[str, str, int, float]] = []
    with open(alpha_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append((
                d["question_id"], d["condition"],
                int(d["agent_idx"]), float(d["flip_rate"]),
            ))
    return rows


def _expected_triples(high_fr_qids_json: Path) -> set[tuple[str, str, int]]:
    """Canonical (qid, cond, agent_idx) triple set for the post-A6 N=200 high-FR pool.

    200 qids × 3 conditions × 3 agents = 1800 expected rows per cohort lane.
    """
    with open(high_fr_qids_json) as f:
        qids = json.load(f)
    return {
        (qid, cond, ai)
        for qid in qids
        for cond in DEFAULT_CONDITIONS
        for ai in DEFAULT_AGENT_INDICES
    }


def _spearman_with_alpha_replaced(
    cohort: list[CohortPoint],
    target_model: str,
    alpha_replacement: float,
) -> float:
    """Headline Spearman ρ when target_model's α_tot is overwritten in-cohort."""
    alphas = np.array(
        [alpha_replacement if p.model == target_model else p.alpha_tot for p in cohort]
    )
    ccond = np.array([p.c_cond for p in cohort])
    rho, _ = stats.spearmanr(alphas, ccond)
    return float(rho)


def partial_lane_sensitivity(
    target_model: str,
    target_alpha_jsonl: Path,
    comparator_alpha_jsonls: list[Path],
    expected_triples: set[tuple[str, str, int]],
    cohort: list[CohortPoint],
    n_resamples: int = 1000,
    seed: int | None = None,
) -> dict:
    """D4 omnibus per-lane sensitivity. Three reports on headline ρ_αtot:

      (a) **Parametric bootstrap** on realized rows (B=n_resamples). Resample
          the realized N target rows with replacement, recompute α_tot,
          replace target's α_tot in cohort, recompute ρ. Returns 2.5/97.5
          percentiles + max |Δρ|. Principled measurement-noise contribution.

      (b) **Worst-case Wilson perturbation** at p=0.5 (most conservative
          binomial variance). Reports ρ under ±half-width perturbation. One-
          shot back-of-envelope; weaker than (a) but transparent.

      (c) **Donor-pool imputation** for missing triples. For each missing
          (qid, cond, agent_idx) triple, impute flip_rate from the mean across
          comparator lanes on that exact triple. Falls back to qid-level mean
          if no comparator has the exact triple. Reports imputed α_tot and
          headline ρ under imputation. Defensible under MAR; matches survey-
          stats donor-pool methodology.

    All three sensitivities reported; narrative emphasis (esp. of (c)) is
    governed by the MCAR-vs-MAR audit |diff| value documented in the D4 entry.
    """
    rng = np.random.default_rng(seed)
    target_rows = _load_alpha_rows(target_alpha_jsonl)
    realized_n = len(target_rows)
    realized_triples = {(r[0], r[1], r[2]) for r in target_rows}
    realized_flip = np.array([r[3] for r in target_rows])
    realized_alpha_tot = float(realized_flip.mean())

    headline_rho_realized = _spearman_with_alpha_replaced(
        cohort, target_model, realized_alpha_tot
    )

    # (a) Parametric bootstrap on realized rows.
    boot_rhos = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, realized_n, size=realized_n)
        alpha_b = float(realized_flip[idx].mean())
        boot_rhos[b] = _spearman_with_alpha_replaced(cohort, target_model, alpha_b)
    bootstrap_block = {
        "n_resamples": int(n_resamples),
        "headline_rho_mean": float(boot_rhos.mean()),
        "headline_rho_std": float(boot_rhos.std(ddof=1)),
        "headline_rho_ci_low": float(np.percentile(boot_rhos, 2.5)),
        "headline_rho_ci_high": float(np.percentile(boot_rhos, 97.5)),
        "delta_rho_max_abs": float(np.max(np.abs(boot_rhos - headline_rho_realized))),
    }

    # (b) Worst-case Wilson perturbation (p=0.5 conservative + observed-p).
    wilson_hw_p05 = _wilson_halfwidth_normal_approx(0.5, realized_n)
    wilson_hw_obs = _wilson_halfwidth_normal_approx(realized_alpha_tot, realized_n)
    wilson_block = {
        "wilson_halfwidth_p05_conservative": wilson_hw_p05,
        "wilson_halfwidth_at_observed_p": wilson_hw_obs,
        "headline_rho_under_plus_p05_perturb": _spearman_with_alpha_replaced(
            cohort, target_model, min(1.0, realized_alpha_tot + wilson_hw_p05)
        ),
        "headline_rho_under_minus_p05_perturb": _spearman_with_alpha_replaced(
            cohort, target_model, max(0.0, realized_alpha_tot - wilson_hw_p05)
        ),
    }

    # (c) Donor-pool imputation for missing triples.
    missing_triples = expected_triples - realized_triples
    n_missing = len(missing_triples)
    if n_missing == 0:
        donor_block = {"applicable": False, "reason": "no_missing_triples"}
    else:
        donor_rows: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        donor_qid_pool: dict[str, list[float]] = defaultdict(list)
        for path in comparator_alpha_jsonls:
            for qid, cond, ai, flip in _load_alpha_rows(path):
                donor_rows[(qid, cond, ai)].append(flip)
                donor_qid_pool[qid].append(flip)
        imputed_flips: list[float] = []
        n_exact = n_qid_fallback = n_unimputable = 0
        for triple in missing_triples:
            if donor_rows[triple]:
                imputed_flips.append(float(np.mean(donor_rows[triple])))
                n_exact += 1
            elif donor_qid_pool[triple[0]]:
                imputed_flips.append(float(np.mean(donor_qid_pool[triple[0]])))
                n_qid_fallback += 1
            else:
                n_unimputable += 1
        if imputed_flips:
            n_total = realized_n + len(imputed_flips)
            sum_flip = float(realized_flip.sum()) + float(sum(imputed_flips))
            imputed_alpha_tot = sum_flip / n_total
            headline_rho_imputed = _spearman_with_alpha_replaced(
                cohort, target_model, imputed_alpha_tot
            )
            donor_block = {
                "applicable": True,
                "n_missing_triples": n_missing,
                "n_exact_triple_donor": n_exact,
                "n_qid_level_fallback": n_qid_fallback,
                "n_unimputable": n_unimputable,
                "n_comparator_lanes": len(comparator_alpha_jsonls),
                "imputed_alpha_tot": imputed_alpha_tot,
                "delta_alpha_tot_imputed_minus_realized":
                    imputed_alpha_tot - realized_alpha_tot,
                "headline_rho_under_imputation": headline_rho_imputed,
                "delta_rho_imputed_minus_realized":
                    headline_rho_imputed - headline_rho_realized,
            }
        else:
            donor_block = {
                "applicable": False,
                "n_missing_triples": n_missing,
                "n_unimputable": n_unimputable,
                "reason": "no_donor_data_for_any_missing_triple",
            }

    return {
        "target_model": target_model,
        "realized_n_rows": realized_n,
        "expected_n_rows": len(expected_triples),
        "n_missing_rows": len(expected_triples) - realized_n,
        "realized_alpha_tot": realized_alpha_tot,
        "headline_rho_realized": headline_rho_realized,
        "parametric_bootstrap": bootstrap_block,
        "worst_case_wilson_perturbation": wilson_block,
        "donor_pool_imputation": donor_block,
    }


def partial_lanes_sensitivity_block(
    cohort: list[CohortPoint],
    alpha_dir: Path,
    high_fr_qids_json: Path,
    pre_reg_sha: str,
    n_resamples: int = 1000,
) -> list[dict]:
    """D4 omnibus driver. Iterate alpha_dir, run partial_lane_sensitivity for
    every cohort-resident lane with realized rows < expected.

    Comparators for a target lane = all OTHER cohort-resident α-probe JSONLs in
    alpha_dir (smoke files and out-of-cohort artifacts excluded). The caller is
    responsible for ensuring all comparator JSONLs are on the same expected
    triple set (post-A6 N=200 high-FR pool); the `_expected_triples` reference
    is derived from `high_fr_qids_json` and shared across calls.
    """
    expected = _expected_triples(high_fr_qids_json)
    expected_n = len(expected)
    cohort_models = {p.model for p in cohort}
    candidate_paths = [
        p for p in sorted(alpha_dir.glob("sa_causal_*.jsonl"))
        if "_smoke" not in p.name
    ]
    cohort_paths = [
        p for p in candidate_paths
        if _model_name_from_alpha_path(p) in cohort_models
    ]
    sensitivities: list[dict] = []
    seed = seed_from_sha(pre_reg_sha)
    for path in cohort_paths:
        n_rows = sum(1 for line in path.open() if line.strip())
        if n_rows >= expected_n:
            continue
        target_model = _model_name_from_alpha_path(path)
        comparators = [p for p in cohort_paths if p != path]
        sensitivities.append(partial_lane_sensitivity(
            target_model=target_model,
            target_alpha_jsonl=path,
            comparator_alpha_jsonls=comparators,
            expected_triples=expected,
            cohort=cohort,
            n_resamples=n_resamples,
            seed=seed,
        ))
    return sensitivities


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-dir", type=Path, required=True)
    ap.add_argument("--alpha-dir", type=Path, required=True)
    ap.add_argument(
        "--cohort-yaml",
        type=Path,
        default=Path("abc_exp/scripts/cohort_n20_models.yaml"),
    )
    ap.add_argument("--pre-reg-sha", default=PRE_REG_SHA_DEFAULT)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("abc_exp/results/cohort_n18_headline.json"),
    )
    ap.add_argument(
        "--high-fr-qids",
        type=Path,
        default=REPO_ROOT / "abc_exp/results/high_fr_question_ids.json",
        help="200-qid JSON list defining the post-A6 high-FR pool (D4 expected triples). "
             "Default points to the canonical pool now tracked in this worktree at "
             "abc_exp/results/high_fr_question_ids.json (D6 follow-up 1422b6f); "
             "sha256 must equal the B-pilot POOL_SHA256 constant.",
    )
    ap.add_argument(
        "--partial-lane-bootstrap-resamples",
        type=int,
        default=1000,
        help="B for parametric-bootstrap-on-realized-rows (D4 omnibus). 1000 = ~1 min total.",
    )
    ap.add_argument(
        "--ccond-endpoint",
        choices=("stored", "r3_alpha"),
        default="stored",
        help="Debate endpoint for the post-A6 rows' C^cond. stored = Round 4 majority_answer "
             "(original); r3_alpha = Round 3, alphabetic ties (paper convention, 2026-10-30).",
    )
    args = ap.parse_args(argv)

    cohort = load_cohort(args.traces_dir, args.alpha_dir, args.cohort_yaml, args.ccond_endpoint)

    expected = set(FAMILY_BY_MODEL)
    realized = {p.model for p in cohort}
    missing = sorted(expected - realized)
    if missing:
        print(
            f"WARNING: cohort missing {len(missing)} model(s): {missing}. "
            "Per A4 §0.3, no replacement; recompute on realized N.",
            file=sys.stderr,
        )

    input_paths = sorted(args.traces_dir.glob("*.jsonl")) + sorted(
        args.alpha_dir.glob("*.json*")
    )
    fingerprint = fingerprint_inputs(input_paths)

    headline_block = headline(cohort, args.pre_reg_sha)
    qwen_blind = subgroup(cohort, set(FAMILY_BY_MODEL) - QWEN_MODELS, args.pre_reg_sha)
    moe_block = subgroup(cohort, MOE_MODELS, args.pre_reg_sha)
    dense_block = subgroup(
        cohort, set(FAMILY_BY_MODEL) - MOE_MODELS, args.pre_reg_sha
    )
    reasoning_block = subgroup(cohort, REASONING_MODELS, args.pre_reg_sha)
    nonreasoning_block = subgroup(
        cohort, set(FAMILY_BY_MODEL) - REASONING_MODELS, args.pre_reg_sha
    )
    lofo_block = lofo(cohort, args.pre_reg_sha)
    partial_lanes_block = partial_lanes_sensitivity_block(
        cohort, args.alpha_dir, args.high_fr_qids, args.pre_reg_sha,
        n_resamples=args.partial_lane_bootstrap_resamples,
    )

    out = {
        "amendment_version": AMENDMENT_VERSION,
        "pre_reg_sha": args.pre_reg_sha,
        "input_fingerprint_sha256": fingerprint,
        "bootstrap_seed": seed_from_sha(args.pre_reg_sha),
        "ccond_endpoint_post_a6": args.ccond_endpoint,
        "n_realized": len(cohort),
        "n_expected": len(FAMILY_BY_MODEL),
        "missing_models": missing,
        "headline": headline_block,
        "sensitivities": {
            "qwen_blind": qwen_blind,
            "moe_subgroup": moe_block,
            "dense_subgroup": dense_block,
            "reasoning_subgroup": reasoning_block,
            "nonreasoning_subgroup": nonreasoning_block,
            "lofo": lofo_block,
            "partial_lanes": partial_lanes_block,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
