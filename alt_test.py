"""Alternative Annotator Test (Calderon et al., 2025).

Port of the official procedure in https://github.com/nitaytech/AltTest
(``alt_test_example.ipynb``). Average advantage probability ρ is the
primary score for comparing judges. Winning rate ω depends on ε and is
secondary.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import ttest_1samp

DEFAULT_EPSILON = 0.15
DEFAULT_Q_FDR = 0.05
MIN_HUMANS_PER_INSTANCE = 3
MIN_INSTANCES_PER_HUMAN = 30
PASS_THRESHOLD = 0.5

InstanceId = Hashable
AnnotatorId = Hashable
Annotation = Any
ScoringFn = Callable[[Any, list[Any]], float]


def majority_label(ratings: Sequence[bool]) -> bool | None:
    """Strict majority of boolean ratings, or None on a tie / empty list."""
    if not ratings:
        return None
    n_true = sum(1 for value in ratings if value)
    n_false = len(ratings) - n_true
    if n_true == n_false:
        return None
    return n_true > n_false


def scoring_gold(ratings: Sequence[bool], *, min_humans: int = MIN_HUMANS_PER_INSTANCE) -> bool | None:
    """Gold used for per-shot diagnostics: majority when ≥ ``min_humans`` ratings."""
    values = [bool(item) for item in ratings]
    if len(values) < min_humans:
        return bool(values[0]) if values else None
    majority = majority_label(values)
    if majority is not None:
        return majority
    return bool(values[0])


def accuracy(pred: Any, annotations: Sequence[Any]) -> float:
    if not annotations:
        return 0.0
    return float(np.mean([pred == ann for ann in annotations]))


def by_procedure(p_values: Sequence[float], q: float) -> list[int]:
    """Benjamini–Yekutieli FDR: indices of rejected nulls (original order)."""
    if not p_values:
        return []
    p_arr = np.array(p_values, dtype=float)
    m = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_pvals = p_arr[sorted_indices]
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = (np.arange(1, m + 1) / m) * (q / harmonic)
    max_i = -1
    for i in range(m):
        if sorted_pvals[i] <= thresholds[i]:
            max_i = i
    if max_i == -1:
        return []
    return [int(idx) for idx in sorted_indices[: max_i + 1]]


def ttest(indicators: Sequence[float], epsilon: float) -> float:
    arr = np.asarray(indicators, dtype=float)
    if len(arr) < 2:
        return 1.0
    if np.allclose(arr, arr[0]):
        return 0.0 if float(arr[0]) < epsilon else 1.0
    pvalue = ttest_1samp(arr, epsilon, alternative="less").pvalue
    if pvalue is None or not np.isfinite(pvalue):
        return 1.0
    return float(pvalue)


def _resolve_scoring_function(scoring_function: str | ScoringFn) -> ScoringFn:
    if isinstance(scoring_function, str):
        if scoring_function == "accuracy":
            return accuracy
        raise ValueError(f"Unknown scoring function: {scoring_function}")
    return scoring_function


def leave_one_out_alignments(
    llm_annotations: Mapping[InstanceId, Annotation],
    humans_annotations: Mapping[AnnotatorId, Mapping[InstanceId, Annotation]],
    scoring_function: str | ScoringFn = "accuracy",
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
) -> tuple[dict[str, list[tuple[float, float]]], list[str], int]:
    """Per excluded annotator, ``(S_llm, S_human)`` vs the remaining labelers.

    ``S`` is the Alt-Test alignment score (ACC for discrete labels): agreement
    with the annotators who were not left out. Advantage indicators are
    ``S_llm >= S_human`` / ``S_human >= S_llm``.
    """
    score_fn = _resolve_scoring_function(scoring_function)

    i_set: dict[AnnotatorId, list[InstanceId]] = {}
    h_set: dict[InstanceId, list[AnnotatorId]] = {}
    for human, anns in humans_annotations.items():
        i_set[human] = list(anns.keys())
        for instance_id, _ann in anns.items():
            h_set.setdefault(instance_id, []).append(human)

    instances_to_keep = {
        instance_id
        for instance_id, annotators in h_set.items()
        if len(annotators) >= min_humans_per_instance and instance_id in llm_annotations
    }
    i_set = {
        human: [instance_id for instance_id in ids if instance_id in instances_to_keep]
        for human, ids in i_set.items()
    }
    h_set = {
        instance_id: annotators
        for instance_id, annotators in h_set.items()
        if instance_id in instances_to_keep
    }

    alignments: dict[str, list[tuple[float, float]]] = {}
    skipped_annotators: list[str] = []
    for excluded_h in humans_annotations:
        instances = [i for i in i_set.get(excluded_h, []) if i in llm_annotations]
        if len(instances) < min_instances_per_human:
            skipped_annotators.append(str(excluded_h))
            continue
        pairs: list[tuple[float, float]] = []
        for instance_id in instances:
            human_ann = humans_annotations[excluded_h][instance_id]
            llm_ann = llm_annotations[instance_id]
            remaining = [
                humans_annotations[other][instance_id]
                for other in h_set[instance_id]
                if other != excluded_h
            ]
            pairs.append(
                (
                    score_fn(llm_ann, remaining),
                    score_fn(human_ann, remaining),
                )
            )
        alignments[str(excluded_h)] = pairs
    return alignments, skipped_annotators, len(instances_to_keep)


def alt_test(
    llm_annotations: Mapping[InstanceId, Annotation],
    humans_annotations: Mapping[AnnotatorId, Mapping[InstanceId, Annotation]],
    scoring_function: str | ScoringFn = "accuracy",
    epsilon: float = DEFAULT_EPSILON,
    q_fdr: float = DEFAULT_Q_FDR,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
) -> dict[str, Any]:
    """Leave-one-out Alt-Test. ρ does not depend on ``epsilon``; ω does."""
    alignments, skipped_annotators, n_kept = leave_one_out_alignments(
        llm_annotations,
        humans_annotations,
        scoring_function=scoring_function,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )

    p_values: list[float] = []
    advantage_probs: list[float] = []
    agree_llm: list[float] = []
    agree_human: list[float] = []
    humans: list[str] = []
    per_annotator: dict[str, dict[str, Any]] = {}

    for excluded_h, pairs in alignments.items():
        llm_scores = [llm_score for llm_score, _human in pairs]
        human_scores = [human_score for _llm, human_score in pairs]
        llm_indicators = [1 if llm_score >= human_score else 0 for llm_score, human_score in pairs]
        excluded_indicators = [
            1 if human_score >= llm_score else 0 for llm_score, human_score in pairs
        ]
        diffs = [
            float(exc_ind - llm_ind)
            for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)
        ]
        rho_f = float(np.mean(llm_indicators)) if llm_indicators else 0.0
        rho_h = float(np.mean(excluded_indicators)) if excluded_indicators else 0.0
        alignment_f = float(np.mean(llm_scores)) if llm_scores else 0.0
        alignment_h = float(np.mean(human_scores)) if human_scores else 0.0
        pvalue = ttest(diffs, epsilon)
        p_values.append(pvalue)
        advantage_probs.append(rho_f)
        agree_llm.append(alignment_f)
        agree_human.append(alignment_h)
        humans.append(excluded_h)
        per_annotator[excluded_h] = {
            "rho_f": rho_f,
            "rho_h": rho_h,
            "alignment_f": alignment_f,
            "alignment_h": alignment_h,
            "p_value": pvalue,
            "n": len(pairs),
            "rejected": False,
        }

    rejected = set(by_procedure(p_values, q_fdr))
    for index in rejected:
        key = humans[index]
        if key in per_annotator:
            per_annotator[key]["rejected"] = True

    n_humans = len(humans)
    winning_rate = (len(rejected) / n_humans) if n_humans else None
    advantage_prob = float(np.mean(advantage_probs)) if advantage_probs else None
    return {
        "advantage_prob": advantage_prob,
        "winning_rate": winning_rate,
        "passed": (winning_rate >= PASS_THRESHOLD) if winning_rate is not None else None,
        "loo_agree_judge": float(np.mean(agree_llm)) if agree_llm else None,
        "loo_agree_human": float(np.mean(agree_human)) if agree_human else None,
        "n_annotators": n_humans,
        "epsilon": float(epsilon),
        "n": n_kept,
        "per_annotator": per_annotator,
        "skipped_annotators": skipped_annotators,
    }


def score_binary_judge(
    instances: Sequence[tuple[InstanceId, Sequence[bool], bool | None]],
    *,
    epsilon: float = DEFAULT_EPSILON,
    min_humans_per_instance: int = MIN_HUMANS_PER_INSTANCE,
    min_instances_per_human: int = MIN_INSTANCES_PER_HUMAN,
    q_fdr: float = DEFAULT_Q_FDR,
) -> dict[str, Any]:
    """Score one judge against boolean human ratings.

    Each item is ``(instance_id, ratings, llm_verdict)``. ``llm_verdict`` is
    ``None`` when the judge did not produce a parsed 0/1. Only instances with
    at least ``min_humans_per_instance`` ratings and a verdict enter the test.
    Rating index is the annotator id.
    """
    n_total = len(instances)
    n_missing = 0
    n_skipped_lt3 = 0
    humans: dict[str, dict[InstanceId, bool]] = {}
    llm: dict[InstanceId, bool] = {}

    for instance_id, ratings, pred in instances:
        values = [bool(item) for item in ratings]
        if len(values) < min_humans_per_instance:
            n_skipped_lt3 += 1
            continue
        if pred is None:
            n_missing += 1
            continue
        llm[instance_id] = bool(pred)
        for index, rating in enumerate(values):
            humans.setdefault(str(index), {})[instance_id] = bool(rating)

    if not llm or not humans:
        return {
            "advantage_prob": None,
            "winning_rate": None,
            "passed": None,
            "loo_agree_judge": None,
            "loo_agree_human": None,
            "n": 0,
            "n_missing": n_missing,
            "n_skipped_lt3": n_skipped_lt3,
            "n_label_rows": n_total,
            "n_annotators": 0,
            "epsilon": float(epsilon),
            "per_annotator": {},
            "skipped_annotators": [],
        }

    result = alt_test(
        llm,
        humans,
        scoring_function="accuracy",
        epsilon=epsilon,
        q_fdr=q_fdr,
        min_humans_per_instance=min_humans_per_instance,
        min_instances_per_human=min_instances_per_human,
    )
    result["n"] = len(llm)
    result["n_missing"] = n_missing
    result["n_skipped_lt3"] = n_skipped_lt3
    result["n_label_rows"] = n_total
    return result
