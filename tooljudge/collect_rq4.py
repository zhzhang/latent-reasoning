"""Join the four RQ4 arms to human labels and emit the flat rows score.py reads.

Each arm lives in its own run directory (rq4-<condition>) because the runner reads a
model's predictions.jsonl whole and rewrites it at the model boundary, so two arms
sharing a directory overwrite each other. Verdicts live in shots[].judges[<label>];
human ratings live in the labels CSV. Nothing currently joins the two.

  uv run python tooljudge/collect_rq4.py \
      --root data/outputs/judge-quality \
      --labels-csv labels_audio_ok.csv \
      --meta mmar/MMAR-meta.json \
      --out rq4_rows.jsonl

Then:
  uv run python tooljudge/score.py rq4_rows.jsonl

Two corrections applied here rather than in score.py:

1. n_tool_calls is summed across samples by aggregate_samples, and the Claude arms run
   3 samples against Gemini's 1. Comparing the raw field across arms inflates Claude
   roughly 3x. tools_per_sample divides by the number of parsed samples. Likewise
   tool_calls is concatenated across samples, so distinct-tool counts taken from it are
   a union over samples, not a within-sample measurement; n_distinct_tools reads
   samples[] where present.

2. correct is None for an unparsed or errored verdict, never False. Those rows are kept
   with parsed=False so parse rate is measurable per condition; score.py drops them.
   Truncation clusters at high tool-call counts, so dropping them silently removes the
   tool-heavy shots. --common keeps only shots parsed in every arm, which is what the
   2x2 contrast needs.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

# condition -> (run directory suffix, judge label)
ARMS = {
    "text_only":   "claude-sonnet-5__neutral_no_gt_text_only",
    "tools_only":  "claude-sonnet-5__neutral_no_gt_tools_only",
    "audio_only":  "gemini-3.7-flash__neutral_no_gt_audio_only",
    "audio_tools": "gemini-3.7-flash__neutral_no_gt_audio_tools",
}
SELF_PREFERENCE_MODEL = "gemini-3.7-flash"  # judge model also under test


def load_meta(path):
    p = Path(path)
    txt = p.read_text()
    rows = (json.loads(txt) if txt.lstrip()[:1] == "["
            else [json.loads(l) for l in txt.splitlines() if l.strip()])
    return {str(r["id"]): r for r in rows}


def load_labels(path):
    """(question_id, model_label, shot_index) -> ratings list."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["question_id"], r["model_label"], int(r["shot_index"]))] = \
                json.loads(r["ratings"])
    return out


def distinct_tools(entry):
    """Mean distinct tool kinds within one sample.

    tool_calls at the top level is concatenated across samples, so its set() is a
    union. samples[] preserves the per-sample lists when n_samples > 1.
    """
    samples = entry.get("samples")
    if samples:
        per = [len(set(s.get("tool_calls") or [])) for s in samples]
        return sum(per) / len(per) if per else 0.0
    return float(len(set(entry.get("tool_calls") or [])))


def read_arm(run_dir, label, condition, labels, meta, unlabelled):
    rows = []
    mdir = Path(run_dir) / "models"
    if not mdir.is_dir():
        return rows
    for md in sorted(mdir.iterdir()):
        pf = md / "predictions.jsonl"
        if not pf.exists():
            continue
        for line in pf.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec.get("id") or rec.get("question_id"))
            for shot in rec.get("shots", []):
                e = (shot.get("judges") or {}).get(label)
                if e is None:
                    continue
                key = (qid, md.name, shot["shot_index"])
                ratings = labels.get(key)
                if ratings is None:
                    unlabelled[condition] += 1
                    continue
                m = meta.get(qid) or {}
                n_par = e.get("n_parsed_samples") or e.get("n_samples") or 1
                n_tc = e.get("n_tool_calls") or 0
                rows.append({
                    "question_id": qid,
                    "model_label": md.name,
                    "shot_index": shot["shot_index"],
                    "condition": condition,
                    "judge_label": label,
                    "self_preference": md.name == SELF_PREFERENCE_MODEL,
                    "category": m.get("category"),
                    "modality": m.get("modality"),
                    "ratings": ratings,
                    "human_pass": sum(ratings) > len(ratings) / 2,
                    "human_split": 0 < sum(ratings) < len(ratings),
                    "parsed": bool(e.get("parsed")),
                    "pass": e.get("correct"),
                    "confidence": e.get("confidence"),
                    "n_tool_calls": n_tc,
                    "tools_per_sample": n_tc / max(1, n_par),
                    "n_distinct_tools": distinct_tools(e),
                    "n_samples": e.get("n_samples", 1),
                    "n_parsed_samples": n_par,
                    "stop_reason": e.get("stop_reason"),
                    "input_tokens": e.get("input_tokens"),
                    "output_tokens": e.get("output_tokens"),
                    "audio_prompt_tokens": e.get("audio_prompt_tokens"),
                    "error": None,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory holding rq4-<condition> run dirs")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="rq4_rows.jsonl")
    ap.add_argument("--prefix", default="rq4-", help="run dir prefix")
    ap.add_argument("--common", action="store_true",
                    help="keep only shots parsed in every arm found")
    a = ap.parse_args()

    labels = load_labels(a.labels_csv)
    meta = load_meta(a.meta)
    unlabelled = defaultdict(int)

    rows, present = [], []
    for cond, label in ARMS.items():
        rd = Path(a.root) / f"{a.prefix}{cond}"
        if not rd.is_dir():
            print(f"  {cond:12s} run dir absent, skipping")
            continue
        r = read_arm(rd, label, cond, labels, meta, unlabelled)
        if not r:
            print(f"  {cond:12s} no verdicts found under {rd}")
            continue
        present.append(cond)
        rows.extend(r)
        n_par = sum(1 for x in r if x["parsed"])
        n_sp = sum(1 for x in r if x["self_preference"])
        tps = [x["tools_per_sample"] for x in r]
        print(f"  {cond:12s} n={len(r):5d}  parsed={n_par:5d} "
              f"({n_par/len(r):.3f})  self_pref={n_sp:4d}  "
              f"tools/sample={sum(tps)/len(tps):.2f}")

    if not rows:
        sys.exit("no verdicts found; check --root and --prefix")
    for c, n in sorted(unlabelled.items()):
        print(f"  note: {c} had {n} judged shots with no label row, dropped")

    if a.common and len(present) > 1:
        seen = defaultdict(set)
        for x in rows:
            if x["parsed"]:
                seen[(x["question_id"], x["model_label"], x["shot_index"])].add(
                    x["condition"])
        keep = {k for k, v in seen.items() if len(v) == len(present)}
        before = len(rows)
        rows = [x for x in rows
                if (x["question_id"], x["model_label"], x["shot_index"]) in keep]
        print(f"\ncommon-item filter: {len(keep)} shots parsed in all "
              f"{len(present)} arms; {before} -> {len(rows)} rows")

    with open(a.out, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")

    n3 = sum(1 for x in rows if len(x["ratings"]) == 3)
    print(f"\nwrote {a.out}: {len(rows)} rows, {n3} with 3 raters "
          f"(alt-test n per condition ~{n3 // max(1, len(present))})")
    print(f"arms present: {', '.join(present)}")


if __name__ == "__main__":
    main()
