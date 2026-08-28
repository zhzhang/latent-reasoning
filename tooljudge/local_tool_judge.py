"""Reference-free tool-calling judge for latent-reasoning freeform runs.

Local and CPU-only. No Modal, no GPU. Reads a downloaded run directory and
writes verdicts back into shots[].judges[<label>] so view_difficulty.py,
aggregate.py and the verdict grid pick them up with no changes.

  python local_tool_judge.py \
      --run-dir data/outputs/judge-quality/llm-judge-gt \
      --audio-dir mmar/audio \
      --meta mmar/MMAR-meta.json \
      --judge gemini --condition audio_tools

Conditions
  text_only    no audio, no tools    (reference-free floor)
  tools_only   no audio, tools       (RQ4: tools are the only route to the clip)
  audio_only   audio, no tools       (RQ1 reference-free number)
  audio_tools  audio and tools       (RQ4: does tool access add to hearing it)

Judges
  claude  claude-sonnet-5. The Messages API has no audio content block, so Claude
          runs text_only and tools_only only; the audio arms error out.
  gemini  gemini-3.7-flash via google-genai, GEMINI_API_KEY. Runs all four.

Idempotent. Re-running only fills shots missing this judge label unless
--force is passed. Writes predictions.jsonl atomically per model.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anthropic
from tqdm import tqdm
import judge as judge_claude
import judge_gemini
from judge import N_SAMPLES

# Naming follows the existing convention in this repo, e.g. the oracle run
# claude-sonnet-5__neutral_with_gt_no_audio__gold
CONDITIONS = {
    "text_only":   {"audio": False, "tools": False},
    "tools_only":  {"audio": False, "tools": True},
    "audio_only":  {"audio": True,  "tools": False},
    "audio_tools": {"audio": True,  "tools": True},
}

# The Anthropic Messages API has no audio content block, so Claude can only run the
# two no-audio arms. Gemini accepts audio and runs all four.
BACKENDS = {
    "claude": {"model": judge_claude.MODEL, "audio": False,
               "temperature": judge_claude.TEMPERATURE,   # None: sampling params rejected
               "max_tokens": judge_claude.MAX_TOKENS},
    "gemini": {"model": judge_gemini.MODEL, "audio": True,
               "temperature": judge_gemini.TEMPERATURE,
               "max_tokens": judge_gemini.MAX_OUTPUT_TOKENS},
}

SAMPLING_NOTE = ("model uses default sampling; claude-sonnet-5 rejects temperature, "
                 "so this arm is not deterministic")


def label_for(backend, condition):
    """e.g. gemini-3.7-flash__neutral_no_gt_audio_tools"""
    return f"{BACKENDS[backend]['model']}__neutral_no_gt_{condition}"


def judge_entry(label, backend, condition, n_samples=N_SAMPLES):
    """A manifest judges[] entry shaped like the ones grader.py writes.

    grader.JUDGE_FORMATS has no include_gold=False + audio_included=False entry, so
    every arm borrows neutral_no_gt for the prompt slot and records the truth in
    audio_included. All are include_gold=False, so judge_mode_bucket buckets them
    as "free" either way.
    """
    cond = CONDITIONS[condition]
    temp = BACKENDS[backend]["temperature"]
    entry = {
        "label": label,
        "model_id": BACKENDS[backend]["model"],
        "primary": False,
        "prompt": "neutral_no_gt",
        "include_gold": False,
        "audio_included": cond["audio"],
        "tools_enabled": cond["tools"],
        "n_samples": n_samples,
        "temperature": temp,
        # Caps differ per judge: Claude truncated at 4000, Gemini never did.
        "max_tokens": BACKENDS[backend]["max_tokens"],
    }
    if temp is None:
        entry["temperature_note"] = SAMPLING_NOTE
    return entry


def register_judge(man_path, entry):
    """Add or repair this label's judges[] entry. Never touches primary_judge.

    Replaces a bare-string entry written by an earlier version, and de-duplicates
    repeat labels. Written atomically so a crash cannot truncate the manifest.
    """
    m = json.loads(Path(man_path).read_text())
    out, seen, action = [], False, "added"
    for raw in m.get("judges") or []:
        raw_label = raw.get("label") if isinstance(raw, dict) else str(raw)
        if raw_label != entry["label"]:
            out.append(raw)
            continue
        if seen:
            action = "de-duplicated"
            continue
        seen = True
        action = "repaired" if not isinstance(raw, dict) else "updated"
        # Preserve an existing primary flag rather than silently demoting a judge.
        merged = dict(entry)
        if isinstance(raw, dict) and raw.get("primary"):
            merged["primary"] = True
        out.append(merged)
    if not seen:
        out.append(dict(entry))
    m["judges"] = out

    tmp = Path(man_path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2) + "\n")
    shutil.move(tmp, man_path)
    return action


def load_meta(path):
    p = Path(path)
    rows = ([json.loads(l) for l in p.open() if l.strip()]
            if p.suffix == ".jsonl" else json.load(p.open()))
    return {str(r["id"]): r for r in rows}


write_lock = threading.Lock()


def journal_path(run, label):
    """Per-shot crash journal, one file per judge label, beside the run dirs."""
    return Path(run) / f"{label}.journal.jsonl"


def append_journal(jf, model, qid, shot_index, entry):
    """One line per finished shot. Caller holds write_lock."""
    with jf.open("a") as fh:
        fh.write(json.dumps({"model": model, "question_id": qid,
                             "shot_index": shot_index, "entry": entry},
                            ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_journal(jf):
    """-> {(model, question_id, shot_index): entry} from a leftover journal."""
    if not jf.exists():
        return {}
    out = {}
    for line in jf.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue        # a torn final line from a hard kill
        out[(r["model"], r["question_id"], r["shot_index"])] = r["entry"]
    return out


def prune_journal(jf, done_model):
    """Drop lines for a model now durable in predictions.jsonl; keep the rest."""
    if not jf.exists():
        return
    keep = [l for l in jf.open()
            if l.strip() and json.loads(l).get("model") != done_model]
    if keep:
        tmp = jf.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(keep))
        shutil.move(tmp, jf)
    else:
        jf.unlink()


def load_label_keys(path):
    """(question_id, model_label, shot_index) for every row in labels.csv.

    Restricting to these keys keeps a run on the shots that actually have human
    ratings, and keeps it off the regenerated models whose text no longer matches
    those ratings. See CLAUDE.md, "Which models may be judged".
    """
    import csv
    with open(path) as fh:
        return {(r["question_id"], r["model_label"], int(r["shot_index"]))
                for r in csv.DictReader(fh)}


def index_audio(d):
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    return {p.stem: str(p) for p in Path(d).rglob("*") if p.suffix.lower() in exts}


def to_schema(res, label, model_id):
    """Match shots[].judges[<label>] as written by grader.py.

    correct is None, never False, when the request errored or the verdict did not
    parse. A request the API never answered must not be readable as a fail.
    view_judges.py counts correct=None as n_missing and drops it from the
    agreement denominator; correct=False would score it against the judge.
    """
    parsed = bool(res.get("parsed"))
    return {
        "correct": bool(res.get("pass")) if parsed else None,
        "verdict": (res.get("verdict") or "unparsed").lower(),
        "output": res.get("verdict") or "Unparsed",
        "generation": res.get("raw", ""),
        "model_id": model_id,
        "confidence": res.get("confidence"),
        "tool_calls": res.get("tool_calls", []),
        "n_tool_calls": res.get("n_tool_calls", 0),
        # stop_reason distinguishes a truncated turn from a genuinely empty one;
        # tokens are summed over the tool loop, so they are the billed cost per shot.
        "stop_reason": res.get("stop_reason"),
        "api_turns": res.get("api_turns"),
        "input_tokens": res.get("input_tokens"),
        "output_tokens": res.get("output_tokens"),
        # Gemini only; None on Claude, which never receives audio.
        "audio_prompt_tokens": res.get("audio_prompt_tokens"),
        "text_prompt_tokens": res.get("text_prompt_tokens"),
        "parsed": parsed,
        "judge_label": label,
    }


def _sum(samples, key):
    vals = [s.get(key) for s in samples if s.get(key) is not None]
    return sum(vals) if vals else None


def aggregate_samples(results, label, model_id):
    """Majority vote over N samples, in grader.py's shots[].judges[<label>] shape.

    An unparsed sample contributes no vote, never a fail vote. The shot is unparsed
    with correct=None when fewer than half the samples parse, or when the parsed
    votes tie -- a tie is no majority, and guessing would invent a verdict.
    """
    samples = [to_schema(r, label, model_id) for r in results]
    n = len(samples)
    votes = [s["correct"] for s in samples if s["parsed"]]
    n_pass, n_fail = sum(1 for v in votes if v), sum(1 for v in votes if not v)
    decided = len(votes) * 2 >= n and n_pass != n_fail

    correct = (n_pass > n_fail) if decided else None
    winner = next((s for s in samples if s["parsed"] and s["correct"] == correct),
                  samples[0])
    confs = [s["confidence"] for s in samples if s["parsed"] and s["confidence"] is not None]
    stops = sorted({str(s["stop_reason"]) for s in samples})

    entry = dict(winner)
    entry.update({
        "correct": correct,
        "verdict": ("pass" if correct else "fail") if decided else "unparsed",
        "output": ("PASS" if correct else "FAIL") if decided else "Unparsed",
        "parsed": decided,
        "confidence": (sum(confs) / len(confs)) if confs else None,
        "stop_reason": stops[0] if len(stops) == 1 else ",".join(stops),
        "tool_calls": [t for s in samples for t in (s["tool_calls"] or [])],
        "n_tool_calls": _sum(samples, "n_tool_calls") or 0,
        "api_turns": _sum(samples, "api_turns"),
        "input_tokens": _sum(samples, "input_tokens"),
        "output_tokens": _sum(samples, "output_tokens"),
        "audio_prompt_tokens": _sum(samples, "audio_prompt_tokens"),
        "text_prompt_tokens": _sum(samples, "text_prompt_tokens"),
        "n_samples": n,
        "n_parsed_samples": len(votes),
        "votes": {"pass": n_pass, "fail": n_fail},
        "samples": samples,
    })
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--judge", default="claude", choices=list(BACKENDS),
                    help="which judge backend to run; default claude")
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--label", help="override the judge label")
    ap.add_argument("--models", help="comma-separated model labels; default all")
    ap.add_argument("--labels-csv",
                    help="restrict judging to shots present in this labels file, "
                         "joined on (question_id, model_label, shot_index); "
                         "unlabelled shots are skipped, not judged")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--limit", type=int,
                   help="max shots per model; may return many shots of one question")
    g.add_argument("--limit-questions", type=int,
                   help="judge every shot of the first N distinct question_ids")
    ap.add_argument("--shots-per-question", type=int,
                    help="cap shots taken from each question; 1 with --limit 5 gives "
                         "5 distinct questions instead of 5 shots of one")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="judgements per shot; >1 writes a samples[] array and a "
                         "majority-vote verdict. Default 1.")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent judge calls within one model. Default 1.")
    ap.add_argument("--force", action="store_true", help="re-judge shots that already have this label")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cond = CONDITIONS[a.condition]
    use_tools, send_audio = cond["tools"], cond["audio"]

    if send_audio and not BACKENDS[a.judge]["audio"]:
        sys.exit(
            f"--judge {a.judge} cannot run --condition {a.condition}: the Anthropic "
            f"Messages API has no audio content block, so {BACKENDS[a.judge]['model']} "
            f"never receives the clip. Audio arms need --judge gemini; Claude supports "
            f"{sorted(c for c, v in CONDITIONS.items() if not v['audio'])}.")

    key = "ANTHROPIC_API_KEY" if a.judge == "claude" else "GEMINI_API_KEY"
    if not a.dry_run and not os.environ.get(key):
        sys.exit(f"set {key}")

    label = a.label or label_for(a.judge, a.condition)
    model_id = BACKENDS[a.judge]["model"]

    run = Path(a.run_dir)
    meta = load_meta(a.meta)
    audio = index_audio(a.audio_dir)
    label_keys = load_label_keys(a.labels_csv) if a.labels_csv else None
    if label_keys is not None:
        print(f"restricting to {len(label_keys)} labelled shots from {a.labels_csv}")
    client = None
    if not a.dry_run:
        # Default is 2. At --workers 8 x --n-samples 3 a 429 that exhausts retries
        # becomes an unparsed sample that casts no vote, so rate-limit pressure would
        # show up as missing data clustered when the pool is hottest.
        client = (anthropic.Anthropic(max_retries=8) if a.judge == "claude"
                  else judge_gemini.make_client())

    jf = journal_path(run, label)
    resume = load_journal(jf)
    if resume:
        print(f"resuming: {len(resume)} shots recovered from {jf.name}")

    model_dirs = sorted((run / "models").iterdir())
    if a.models:
        keep = set(a.models.split(","))
        model_dirs = [d for d in model_dirs if d.name in keep]

    for mdir in model_dirs:
        pf = mdir / "predictions.jsonl"
        if not pf.exists():
            continue
        records = [json.loads(l) for l in pf.open() if l.strip()]

        pending = []
        for rec in records:
            qid = str(rec.get("id") or rec.get("question_id"))
            taken = 0
            for shot in rec.get("shots", []):
                if (label_keys is not None
                        and (qid, mdir.name, shot["shot_index"]) not in label_keys):
                    continue
                # Journalled shots are already judged: replay them into the record
                # so the model-boundary write persists them, and do not re-judge.
                jkey = (mdir.name, qid, shot["shot_index"])
                if jkey in resume:
                    shot.setdefault("judges", {})[label] = resume[jkey]
                    continue
                if not a.force and label in shot.get("judges", {}):
                    continue
                if a.shots_per_question and taken >= a.shots_per_question:
                    break
                pending.append((qid, rec, shot))
                taken += 1
        if a.limit:
            pending = pending[:a.limit]
        elif a.limit_questions:
            keep = set(list(dict.fromkeys(q for q, _, _ in pending))[:a.limit_questions])
            pending = [p for p in pending if p[0] in keep]

        print(f"{mdir.name}: {len(pending)} shots to judge as '{label}'")
        if a.dry_run:
            nq = len({q for q, _, _ in pending})
            print(f"  {nq} distinct questions; first 10 shots:")
            for qid, _, shot in pending[:10]:
                print(f"    {qid}  shot {shot['shot_index']}")
            continue
        if not pending:
            continue

        errors = [0]

        def judge_one(item):
            """Judge one shot. Runs on a worker thread; touches no shared state."""
            qid, rec, shot = item
            path = audio.get(qid)
            question = (meta.get(qid) or {}).get("question") or rec.get("question")
            answer = shot.get("answer_prediction", "")
            # The clip is needed to attach it and to run the tools on it; the
            # text_only arm never touches it.
            missing = ("question" if not question
                       else "audio" if (use_tools or send_audio) and not path else None)
            if missing:
                return to_schema({"parsed": False, "verdict": "Error",
                                  "raw": f"missing {missing} for {qid}"},
                                 label, model_id), 1
            results, n_err = [], 0
            for _ in range(a.n_samples):
                try:
                    # judge.judge() has no send_audio parameter; Claude never gets audio.
                    results.append(
                        judge_gemini.judge(client, path, question, answer,
                                           use_tools, send_audio)
                        if a.judge == "gemini"
                        else judge_claude.judge(client, path, question, answer, use_tools))
                except Exception as exc:
                    results.append({"parsed": False,
                                    "raw": f"{type(exc).__name__}: {exc}"})
                    n_err += 1
            entry = (to_schema(results[0], label, model_id) if a.n_samples == 1
                     else aggregate_samples(results, label, model_id))
            return entry, n_err

        def run_shot(item):
            """Judge, then commit under the lock. The journal is the only
            concurrent write; predictions.jsonl is still written once, later."""
            qid, _, shot = item
            entry, n_err = judge_one(item)
            with write_lock:
                shot.setdefault("judges", {})[label] = entry
                errors[0] += n_err
                append_journal(jf, mdir.name, qid, shot["shot_index"], entry)
            return entry

        if a.workers > 1:
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for _ in tqdm(ex.map(run_shot, pending), total=len(pending),
                              desc=f"{mdir.name} x{a.workers}"):
                    pass
        else:
            for item in tqdm(pending, desc=mdir.name):
                run_shot(item)

        # Model boundary: the single atomic write, unchanged.
        tmp = pf.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shutil.move(tmp, pf)
        # Everything for this model is now durable in predictions.jsonl, so its
        # journal lines are redundant. Lines for other models (from an earlier
        # crashed run) are preserved.
        with write_lock:
            prune_journal(jf, mdir.name)
        print(f"  wrote {pf}  ({errors[0]} errors)")

    man = run / "manifest.json"
    if man.exists() and not a.dry_run:
        action = register_judge(man, judge_entry(label, a.judge, a.condition, a.n_samples))
        print(f"{action} '{label}' in manifest.json (primary_judge unchanged)")

    print("\nNow re-aggregate and view:")
    print("  uv run python aggregate.py --run-id <run_id>   # if it takes one")
    print("  uv run python view_difficulty.py")


if __name__ == "__main__":
    main()
