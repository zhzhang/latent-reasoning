"""Score LALM-no-GT judges against the LLM-with-GT first-shot pack.

Treats ``llm-judge-gt`` majority votes as gold. Reports agreement
(accuracy), precision/recall/F1, Cohen's κ, and — when the GT entry
stores three sample votes — Alt-Test ρ as in ``view_judges.py``.

Usage::

    uv run modal run judge-quality/download_judge_quality.py
    uv run python judge-quality/score_lalm_vs_gt.py
    uv run python judge-quality/view_judge_quality.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aggregate import order_model_labels
from alt_test import DEFAULT_EPSILON, score_binary_judge
from grader import parse_judge_key
from mmar_common import load_jsonl, write_json

GT_PACK_NAME = "llm-judge-gt"
LALM_PACK_NAME = "lalm-judge-no-gt"
DEFAULT_LOCAL_DIR = _REPO_ROOT / "outputs" / "judge-quality"
GT_PROMPT = "neutral_with_gt_no_audio"
LALM_PROMPT = "neutral_no_gt"
ACCURACY_JSON_NAME = "lalm_vs_gt.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def first_shot(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    shots = list(record.get("shots") or [])
    shots.sort(key=_shot_index)
    chosen = next((shot for shot in shots if _shot_index(shot) == 0), None)
    if chosen is None and shots:
        chosen = shots[0]
    return chosen if isinstance(chosen, dict) else None


def overlay_sidecars(model_dir: Path, by_id: dict[str, dict[str, Any]]) -> None:
    """Fill missing ``shots[].judges`` keys from ``judge_partials/*.jsonl``."""
    partials = model_dir / "judge_partials"
    if not partials.is_dir():
        return
    for path in sorted(partials.glob("*.jsonl")):
        for row in load_jsonl(path):
            if not isinstance(row, dict):
                continue
            qid = str(row.get("id") or "").strip()
            record = by_id.get(qid)
            key = str(row.get("judge_key") or "")
            entry = row.get("entry")
            if record is None or not key or not isinstance(entry, dict):
                continue
            shot = first_shot(record)
            if shot is None:
                continue
            try:
                if int(row.get("shot_index", 0)) != _shot_index(shot):
                    continue
            except (TypeError, ValueError):
                continue
            judges = shot.setdefault("judges", {})
            if not isinstance(judges, dict):
                judges = {}
                shot["judges"] = judges
            if key not in judges:
                judges[key] = entry


def load_pack_predictions(pack_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    models_root = pack_dir / "models"
    if not models_root.is_dir():
        return predictions
    for child in sorted(models_root.iterdir()):
        pred_path = child / "predictions.jsonl"
        if not child.is_dir() or not pred_path.is_file():
            continue
        by_id: dict[str, dict[str, Any]] = {}
        for record in load_jsonl(pred_path):
            if not isinstance(record, dict):
                continue
            qid = str(record.get("id") or "").strip()
            if qid:
                by_id[qid] = record
        overlay_sidecars(child, by_id)
        predictions[child.name] = by_id
    return predictions


def collect_judge_keys(
    predictions: dict[str, dict[str, dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in manifest.get("judges") or []:
        if isinstance(raw, dict) and raw.get("label"):
            key = str(raw["label"])
        elif isinstance(raw, str) and raw:
            key = raw
        else:
            continue
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    primary = manifest.get("primary_judge")
    if primary and str(primary) not in seen:
        ordered.insert(0, str(primary))
        seen.add(str(primary))
    for by_id in predictions.values():
        for record in by_id.values():
            for shot in record.get("shots") or []:
                judges = shot.get("judges") or {}
                if not isinstance(judges, dict):
                    continue
                for key in judges:
                    if key and key not in seen:
                        ordered.append(str(key))
                        seen.add(str(key))
    return ordered


def short_judge_name(key: str) -> str:
    parsed = parse_judge_key(key)
    return parsed.get("model") or key


def entry_correct(entry: Any) -> bool | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("correct") is not None:
        return bool(entry["correct"])
    verdict = str(entry.get("verdict") or "").strip().lower()
    if verdict == "pass":
        return True
    if verdict == "fail":
        return False
    return None


def gt_sample_ratings(entry: Any) -> list[bool]:
    """Boolean votes from the GT judge's 3 samples, if stored."""
    if not isinstance(entry, dict):
        return []
    ratings: list[bool] = []
    for sample in entry.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        verdict = str(sample.get("verdict") or "").strip().lower()
        if verdict in {"pass", "fail"}:
            ratings.append(verdict == "pass")
            continue
        if sample.get("correct") is not None:
            ratings.append(bool(sample["correct"]))
    return ratings


def pick_gt_judge_key(keys: list[str], manifest: dict[str, Any]) -> str:
    primary = str(manifest.get("primary_judge") or "")
    if primary and primary in keys:
        return primary
    for key in keys:
        parsed = parse_judge_key(key)
        if parsed.get("prompt") == GT_PROMPT:
            return key
    for key in keys:
        if parse_judge_key(key).get("gold_tag") == "gold":
            return key
    if keys:
        return keys[0]
    raise SystemExit("No GT judge key found in llm-judge-gt pack")


def pick_lalm_judge_keys(keys: list[str]) -> list[str]:
    matched = [
        key for key in keys if parse_judge_key(key).get("prompt") == LALM_PROMPT
    ]
    return matched or list(keys)


def _load_question_ids(pack_dir: Path) -> list[str]:
    payload = _load_json(pack_dir / "question_ids.json")
    ids = payload.get("ids") if isinstance(payload, dict) else None
    if isinstance(ids, list) and ids:
        return [str(qid) for qid in ids if str(qid).strip()]
    return []


def confusion_stats(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """``pairs`` are ``(gt_correct, pred_correct)``."""
    n = len(pairs)
    empty = {
        "n": 0,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "kappa": None,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "gt_pass_rate": None,
        "pred_pass_rate": None,
    }
    if n == 0:
        return empty
    tp = tn = fp = fn = 0
    for gold, pred in pairs:
        if gold and pred:
            tp += 1
        elif (not gold) and (not pred):
            tn += 1
        elif (not gold) and pred:
            fp += 1
        else:
            fn += 1
    acc = (tp + tn) / n
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    p_gt = (tp + fn) / n
    p_pred = (tp + fp) / n
    pe = p_gt * p_pred + (1 - p_gt) * (1 - p_pred)
    if pe >= 1:
        kappa = 1.0 if acc >= 1 else 0.0
    else:
        kappa = (acc - pe) / (1 - pe)
    return {
        "n": n,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "gt_pass_rate": p_gt,
        "pred_pass_rate": p_pred,
    }


def _fmt_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.3f}"


def print_score_table(payload: dict[str, Any]) -> None:
    judges = payload.get("judges") or {}
    print(
        f"\n=== LALM-no-GT vs LLM-GT ({payload.get('gt_judge_key')}) ==="
    )
    print(
        f"{'judge':<42} {'n':>7} {'miss':>6} {'acc':>8} {'κ':>8} "
        f"{'F1':>8} {'ρ':>8} {'P':>8} {'R':>8}"
    )
    for key in judges:
        row = judges[key]
        stats = row.get("overall") or row
        alt = row.get("alt_test") or {}
        print(
            f"{short_judge_name(key):<42} {stats.get('n', 0):>7} "
            f"{row.get('n_missing', 0):>6} "
            f"{_fmt_rate(stats.get('accuracy')):>8} "
            f"{_fmt_rate(stats.get('kappa')):>8} "
            f"{_fmt_rate(stats.get('f1')):>8} "
            f"{_fmt_rate(alt.get('advantage_prob')):>8} "
            f"{_fmt_rate(stats.get('precision')):>8} "
            f"{_fmt_rate(stats.get('recall')):>8}"
        )
    print()
    print(
        "acc = agreement with LLM-GT majority. κ = Cohen's kappa. "
        "ρ = Alt-Test average advantage vs the three GT samples."
    )
    print(
        f"pairs={payload.get('n_pairs')} questions={payload.get('n_questions')} "
        f"models={len(payload.get('models') or [])}"
    )


def score_experiment(
    gt_dir: Path,
    lalm_dir: Path,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    if not gt_dir.is_dir():
        raise SystemExit(
            f"GT pack not found: {gt_dir}\n"
            "Download with: uv run modal run judge-quality/download_judge_quality.py"
        )
    if not lalm_dir.is_dir():
        raise SystemExit(
            f"LALM pack not found: {lalm_dir}\n"
            "Download with: uv run modal run judge-quality/download_judge_quality.py"
        )

    gt_manifest = _load_json(gt_dir / "manifest.json")
    lalm_manifest = _load_json(lalm_dir / "manifest.json")
    gt_preds = load_pack_predictions(gt_dir)
    lalm_preds = load_pack_predictions(lalm_dir)
    gt_key = pick_gt_judge_key(
        collect_judge_keys(gt_preds, gt_manifest), gt_manifest
    )
    lalm_keys = pick_lalm_judge_keys(collect_judge_keys(lalm_preds, lalm_manifest))
    if not lalm_keys:
        raise SystemExit(f"No LALM judge keys in {lalm_dir}")

    question_ids = _load_question_ids(gt_dir) or _load_question_ids(lalm_dir)
    model_labels = order_model_labels(
        list(dict.fromkeys([*gt_preds, *lalm_preds]))
    )
    if not question_ids:
        seen: list[str] = []
        found: set[str] = set()
        for label in model_labels:
            for qid in gt_preds.get(label, {}):
                if qid not in found:
                    found.add(qid)
                    seen.append(qid)
        question_ids = seen

    # (qid, model) -> payloads used by the viewer
    pairs_by_judge: dict[str, list[tuple[bool, bool]]] = {key: [] for key in lalm_keys}
    alt_instances: dict[str, list[tuple[str, list[bool], bool | None]]] = {
        key: [] for key in lalm_keys
    }
    missing: dict[str, int] = {key: 0 for key in lalm_keys}
    by_model_pairs: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        key: {label: [] for label in model_labels} for key in lalm_keys
    }
    by_id: dict[str, dict[str, Any]] = {}
    n_pairs = 0
    n_gt_missing = 0

    for qid in question_ids:
        sample_record = None
        for label in model_labels:
            rec = lalm_preds.get(label, {}).get(qid) or gt_preds.get(label, {}).get(qid)
            if rec is not None:
                sample_record = rec
                break
        q_row = {
            "id": qid,
            "question": (sample_record or {}).get("question"),
            "answer": (sample_record or {}).get("answer"),
            "audio_path": (sample_record or {}).get("audio_path"),
            "category": (sample_record or {}).get("category"),
            "language": (sample_record or {}).get("language"),
            "models": {},
        }
        for extra in ("source", "modality", "sub-category", "url"):
            if sample_record and sample_record.get(extra) is not None:
                q_row[extra] = sample_record.get(extra)
        for label in model_labels:
            gt_rec = gt_preds.get(label, {}).get(qid)
            lalm_rec = lalm_preds.get(label, {}).get(qid)
            if gt_rec is None and lalm_rec is None:
                continue
            gt_shot = first_shot(gt_rec)
            lalm_shot = first_shot(lalm_rec)
            shot = lalm_shot or gt_shot or {}
            gt_entry = (gt_shot or {}).get("judges", {}).get(gt_key) if gt_shot else None
            gt_correct = entry_correct(gt_entry)
            prediction = str(
                shot.get("answer_prediction")
                or (gt_shot or {}).get("answer_prediction")
                or ""
            )
            lalm_entries: dict[str, dict[str, Any]] = {}
            if isinstance((lalm_shot or {}).get("judges"), dict):
                for key in lalm_keys:
                    entry = (lalm_shot or {}).get("judges", {}).get(key)
                    if isinstance(entry, dict):
                        lalm_entries[key] = entry
            if gt_correct is None:
                n_gt_missing += 1
            else:
                n_pairs += 1
                ratings = gt_sample_ratings(gt_entry)
                instance_id = f"{qid}\t{label}"
                for key in lalm_keys:
                    pred = entry_correct(lalm_entries.get(key))
                    if pred is None:
                        missing[key] += 1
                        alt_instances[key].append((instance_id, ratings, None))
                        continue
                    pairs_by_judge[key].append((gt_correct, pred))
                    by_model_pairs[key][label].append((gt_correct, pred))
                    alt_instances[key].append((instance_id, ratings, pred))
            q_row["models"][label] = {
                "model_label": label,
                "answer_prediction": prediction,
                "thinking_prediction": shot.get("thinking_prediction"),
                "gt": {
                    "key": gt_key,
                    "correct": gt_correct,
                    "entry": gt_entry if isinstance(gt_entry, dict) else None,
                },
                "lalms": {
                    key: {
                        "correct": entry_correct(entry),
                        "entry": entry,
                    }
                    for key, entry in lalm_entries.items()
                },
            }
        by_id[qid] = q_row

    judges_out: dict[str, dict[str, Any]] = {}
    for key in lalm_keys:
        overall = confusion_stats(pairs_by_judge[key])
        by_model = {
            label: confusion_stats(pairs)
            for label, pairs in by_model_pairs[key].items()
            if pairs
        }
        alt = score_binary_judge(alt_instances[key], epsilon=epsilon)
        judges_out[key] = {
            "label": key,
            "short": short_judge_name(key),
            "n_missing": missing[key],
            "overall": overall,
            "by_model": by_model,
            "alt_test": alt,
        }

    questions: list[dict[str, Any]] = []
    for qid in question_ids:
        q_row = by_id.get(qid)
        if not q_row:
            continue
        n_models = 0
        n_scored = 0
        n_gt_pass = 0
        n_disagree: dict[str, int] = {key: 0 for key in lalm_keys}
        n_agree: dict[str, int] = {key: 0 for key in lalm_keys}
        for model_entry in (q_row.get("models") or {}).values():
            n_models += 1
            gt_correct = (model_entry.get("gt") or {}).get("correct")
            if gt_correct is True:
                n_gt_pass += 1
            if gt_correct is None:
                continue
            n_scored += 1
            for key in lalm_keys:
                pred = ((model_entry.get("lalms") or {}).get(key) or {}).get("correct")
                if pred is None:
                    continue
                if pred == gt_correct:
                    n_agree[key] += 1
                else:
                    n_disagree[key] += 1
        n_disagree_any = sum(1 for key in lalm_keys if n_disagree[key])
        n_disagree_shots = max(n_disagree.values()) if n_disagree else 0
        total_disagree = sum(n_disagree.values())
        total_compared = sum(n_agree[k] + n_disagree[k] for k in lalm_keys)
        questions.append(
            {
                "id": qid,
                "question": q_row.get("question") or "",
                "category": q_row.get("category"),
                "models": list((q_row.get("models") or {}).keys()),
                "n_models": n_models,
                "n_scored": n_scored,
                "n_gt_pass": n_gt_pass,
                "n_disagree_any": n_disagree_any,
                "n_disagree_max": n_disagree_shots,
                "n_disagree": total_disagree,
                "n_compared": total_compared,
                "agree_rate": (
                    (total_compared - total_disagree) / total_compared
                    if total_compared
                    else None
                ),
                "disagree_by_judge": n_disagree,
                "agree_by_judge": n_agree,
            }
        )

    return {
        "gt_pack": str(gt_dir),
        "lalm_pack": str(lalm_dir),
        "gt_judge_key": gt_key,
        "lalm_judge_keys": lalm_keys,
        "n_questions": len(question_ids),
        "n_pairs": n_pairs,
        "n_gt_missing": n_gt_missing,
        "models": model_labels,
        "judges": judges_out,
        "epsilon": float(epsilon),
        "gt_manifest": {
            "name": gt_manifest.get("name"),
            "grade_prompt": gt_manifest.get("grade_prompt"),
            "n_samples": gt_manifest.get("n_samples"),
        },
        "lalm_manifest": {
            "name": lalm_manifest.get("name"),
            "grade_prompt": lalm_manifest.get("grade_prompt"),
            "n_samples": lalm_manifest.get("n_samples"),
        },
        "questions": questions,
        "by_id": by_id,
        "question_ids": question_ids,
    }


def scores_only(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable scores without the per-question ``by_id`` blob."""
    return {key: value for key, value in payload.items() if key != "by_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR / GT_PACK_NAME,
        help="Downloaded llm-judge-gt pack",
    )
    parser.add_argument(
        "--lalm-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR / LALM_PACK_NAME,
        help="Downloaded lalm-judge-no-gt pack",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_LOCAL_DIR / ACCURACY_JSON_NAME,
        help="Where to write lalm_vs_gt.json",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Alt-Test ε for winning rate ω (ρ does not use this)",
    )
    args = parser.parse_args()
    payload = score_experiment(
        args.gt_dir.expanduser().resolve(),
        args.lalm_dir.expanduser().resolve(),
        epsilon=args.epsilon,
    )
    print_score_table(payload)
    dest = args.out.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    slim = scores_only(payload)
    slim["questions"] = payload["questions"]
    write_json(dest, slim)
    print(f"Wrote {dest}")
    print("Browse:\n  uv run python judge-quality/view_judge_quality.py")


if __name__ == "__main__":
    main()
