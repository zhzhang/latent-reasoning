"""Validity checks for the Claude+GT oracle before it stands in for human labels.

Usage:
  python check_oracle.py --labels labels.csv --generations generations.csv \
      --new-generations data/outputs/mmar-freeform-thinking \
      --oracle data/outputs/judge-quality/llm-judge-gt \
      --meta mmar/MMAR-meta.json
Any argument may be omitted; the checks that need it are skipped.

--new-generations accepts a run directory (models/<label>/predictions.jsonl), a
single predictions.jsonl, or a CSV.

--oracle accepts either
  * a run directory holding models/<label>/predictions.jsonl with verdicts already
    materialised under shots[].judges[<label>] by grader.py  (preferred), or
  * an _anthropic_batch/<judge_key> shard holding jobs.jsonl + output.jsonl, which
    is re-parsed here and needs its own sample aggregation.
"""
from __future__ import annotations
import argparse, json, glob, os, collections
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency

GOLD_LABEL = "claude-sonnet-5__neutral_with_gt_no_audio__gold"
KEY = ["question_id", "model_label", "shot_index"]


def boot_ci(x, n=5000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def read_json_records(path):
    """A .json array or a .jsonl stream, whichever the file actually is."""
    with open(path) as fh:
        head = fh.read(4096).lstrip()
        fh.seek(0)
        if head.startswith("["):
            return json.load(fh)
        return [json.loads(l) for l in fh if l.strip()]


def resolve_meta(path):
    """MMAR ships MMAR-meta.json; older notes call it .jsonl. Accept either."""
    if os.path.exists(path):
        return path
    for a, b in ((".jsonl", ".json"), (".json", ".jsonl")):
        if path.endswith(a):
            alt = path[: -len(a)] + b
            if os.path.exists(alt):
                print(f"[meta] {path} not found, using {alt}")
                return alt
    raise FileNotFoundError(path)


def _prediction_files(path):
    """Run dir -> its per-model predictions.jsonl; a file -> itself."""
    if os.path.isfile(path):
        return [path]
    hits = sorted(glob.glob(os.path.join(path, "models", "*", "predictions.jsonl")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(path, "**", "predictions.jsonl"),
                                recursive=True))
    return hits


def load_generation_frame(path):
    """-> question_id, model_label, shot_index, answer_prediction [, generation_id].

    predictions.jsonl has no generation_id and keeps per-shot text under shots[];
    the model label is the directory name, not a field.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        print(f"[gen] {path}: {len(df)} rows (csv)")
        return df
    rows = []
    files = _prediction_files(path)
    for f in files:
        model = os.path.basename(os.path.dirname(f))
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            for s in r.get("shots") or []:
                rows.append({"question_id": r["id"], "model_label": model,
                             "shot_index": s["shot_index"],
                             "answer_prediction": s.get("answer_prediction")})
    df = pd.DataFrame(rows)
    print(f"[gen] {path}: {len(df)} shot rows from {len(files)} predictions.jsonl "
          f"({df.model_label.nunique() if len(df) else 0} models), no generation_id")
    return df


def _verdict_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    return str(v).strip().upper() in {"PASS", "CORRECT", "TRUE", "YES", "1"}


def _oracle_from_run_dir(path, label):
    """Verdicts already written into shots[].judges[<label>] by grader.py."""
    files = _prediction_files(path)
    seen = collections.Counter()
    rows = []
    for f in files:
        model = os.path.basename(os.path.dirname(f))
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            for s in r.get("shots") or []:
                for lb in (s.get("judges") or {}):
                    seen[lb] += 1
                j = (s.get("judges") or {}).get(label)
                if not j or j.get("correct") is None:
                    continue
                rows.append({"question_id": r["id"], "model_label": model,
                             "shot_index": s["shot_index"],
                             "oracle": _verdict_bool(j["correct"])})
    print(f"[oracle] run dir, {len(files)} predictions.jsonl")
    print(f"[oracle] judge labels present: {dict(seen)}")
    df = pd.DataFrame(rows)
    print(f"[oracle] label '{label}': {len(df)} shot verdicts, "
          f"{df.model_label.nunique() if len(df) else 0} models")
    return df


def _parse_batch_generation(text):
    """The judge is told to end on a single word. Returns True/False/None."""
    if not text:
        return None
    last = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not last:
        return None
    tail = last[-1].strip().strip('".*').upper()
    if "INCORRECT" in tail:
        return False
    if "CORRECT" in tail:
        return True
    return None


def _oracle_from_batch_dir(path, label):
    """Re-parse an _anthropic_batch shard: jobs.jsonl owners x output.jsonl text."""
    jobs_p, out_p = os.path.join(path, "jobs.jsonl"), os.path.join(path, "output.jsonl")
    state_p = os.path.join(path, "state.json")
    if os.path.exists(state_p):
        st = json.load(open(state_p))
        print(f"[oracle] batch {st.get('judge_key')} status={st.get('status')!r} "
              f"n_requests={st.get('n_requests')}")
        if st.get("status") not in ("ended", "completed", "succeeded"):
            print(f"  *** state.json says status={st.get('status')!r}; this shard was "
                  "not confirmed complete ***")

    jobs = [json.loads(l) for l in open(jobs_p) if l.strip()]
    sample_to_key, defined = {}, set()
    for j in jobs:
        owners = j.get("owners") or []
        for sid in j.get("sample_custom_ids") or []:
            defined.add(sid)
            for o in owners:
                sample_to_key[sid] = (o["qid"], o["model"], o["shot_index"])

    texts, types = {}, collections.Counter()
    present = set()
    for line in open(out_p):
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r["custom_id"]
        present.add(cid)
        if cid not in defined:
            continue
        res = r.get("result") or {}
        types[res.get("type")] += 1
        if res.get("type") != "succeeded":
            continue
        content = (res.get("message") or {}).get("content") or []
        texts[cid] = "".join(c.get("text", "") for c in content if c.get("type") == "text")

    print(f"[oracle] jobs.jsonl defines {len(jobs)} jobs / {len(defined)} sample ids")
    print(f"[oracle] output.jsonl holds {len(present)} custom_ids "
          f"({len(present - defined)} of them belong to other shards of this run)")
    print(f"[oracle] result types for this shard's ids: {dict(types)}")
    missing = defined - present
    if missing:
        print(f"  *** {len(missing)} of this shard's sample ids have no output row ***")

    per_key = collections.defaultdict(list)
    n_unparsed = 0
    for sid, key in sample_to_key.items():
        if sid not in texts:
            continue
        v = _parse_batch_generation(texts[sid])
        if v is None:
            n_unparsed += 1
            continue
        per_key[key].append(v)
    total = sum(len(v) for v in per_key.values()) + n_unparsed
    rate = (total - n_unparsed) / total if total else 0.0
    print(f"[oracle] parse rate {rate:.4f} ({total - n_unparsed}/{total} samples)")
    print("[oracle] NOTE: samples aggregated here by majority vote. The run dir "
          "carries grader.py's own aggregation and does not need this step.")

    rows = [{"question_id": q, "model_label": m, "shot_index": s,
             "oracle": sum(v) > len(v) / 2}
            for (q, m, s), v in per_key.items() if v]
    df = pd.DataFrame(rows)
    print(f"[oracle] {len(df)} shot verdicts, "
          f"{df.model_label.nunique() if len(df) else 0} models: "
          f"{sorted(df.model_label.unique()) if len(df) else []}")
    return df


def load_oracle(path, label):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "jobs.jsonl")):
        return _oracle_from_batch_dir(path, label)
    return _oracle_from_run_dir(path, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--generations", default="generations.csv")
    ap.add_argument("--new-generations")
    ap.add_argument("--oracle")
    ap.add_argument("--oracle-label", default=GOLD_LABEL)
    ap.add_argument("--meta")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels); lab["ratings"] = lab.ratings.apply(json.loads)
    gen = pd.read_csv(a.generations)
    full = lab[lab.ratings.apply(len) == 3].copy()
    M = np.array(full.ratings.tolist()).astype(int)

    print("=" * 70)
    print("CHECK 1  human baseline")
    lo = []
    for j in range(3):
        others = [i for i in range(3) if i != j]
        lo.append((M[:, [j]] == M[:, others]).mean())
    print(f"  each annotator vs other two: {[round(x,4) for x in lo]}")
    print(f"  mean = {np.mean(lo):.4f}   <- the 0.89 to beat")
    print(f"  n items = {len(M)}, n questions = {full.question_id.nunique()}")

    print("=" * 70)
    print("CHECK 2  is the labelled subset representative of all generations?")
    labelled = set(lab.question_id)
    allq = set(gen.question_id)
    print(f"  labelled {len(labelled)} of {len(allq)} questions "
          f"({len(labelled)/len(allq):.1%})")
    if a.meta:
        recs = read_json_records(resolve_meta(a.meta))
        meta = pd.DataFrame(recs)[["id", "category", "modality"]]
        meta["labelled"] = meta.id.isin(labelled)
        meta = meta[meta.id.isin(allq)]
        for key in ("category", "modality"):
            ct = pd.crosstab(meta[key], meta.labelled)
            if ct.shape[1] == 2 and (ct.values.min() >= 1):
                chi2, p, _, _ = chi2_contingency(ct)
                flag = "NOT REPRESENTATIVE" if p < 0.05 else "ok"
                print(f"\n  {key}: chi2 p={p:.4f}  {flag}")
                print((ct[True] / ct.sum(1)).round(3).to_string())

    print("=" * 70)
    print("CHECK 3  do human labels still match the new generation text?")
    if a.new_generations:
        new = load_generation_frame(a.new_generations)
        if "generation_id" in new.columns:
            old = gen.set_index("generation_id").answer_prediction
            nw = new.set_index("generation_id").answer_prediction
            shared = old.index.intersection(nw.index)
            labelled_ids = set(lab.generation_id) & set(shared)
        else:
            labelled_ids = set()
            print("  new file has no generation_id column")
        if labelled_ids:
            ids = sorted(labelled_ids)
            same = (old.loc[ids].fillna("") == nw.loc[ids].fillna("")).mean()
            print(f"  {len(ids)} labelled generation_ids appear in both files")
            print(f"  text identical for {same:.1%}")
            if same < 0.99:
                print("  *** LABELS DO NOT MATCH THE NEW TEXT. "
                      "Re-key on (question_id, model_label, shot_index) or relabel. ***")
        else:
            print("  no labelled generation_ids overlap; ids were reassigned")
            print("  *** join on (question_id, model_label, shot_index) instead ***")
            # same statistic, composite key
            o = gen.merge(lab[KEY], on=KEY).drop_duplicates(KEY).set_index(KEY).answer_prediction
            n = new.drop_duplicates(KEY).set_index(KEY).answer_prediction
            sh = o.index.intersection(n.index)
            if len(sh):
                same = (o.loc[sh].fillna("") == n.loc[sh].fillna("")).mean()
                print(f"  on the composite key: {len(sh)} labelled shots in both files")
                print(f"  text identical for {same:.1%}")
                if same < 0.99:
                    print("  *** LABELS DO NOT MATCH THE NEW TEXT; relabel before use ***")
            else:
                print("  composite key does not overlap either")
        print(f"  models in new file: {sorted(new.model_label.unique())}")
    else:
        print("  skipped, pass --new-generations")

    print("=" * 70)
    print("CHECK 4  oracle vs humans, with a confidence interval")
    if a.oracle:
        orc = load_oracle(a.oracle, a.oracle_label)
        if not len(orc):
            print("  no oracle verdicts loaded")
        else:
            mg = full.merge(orc, on=KEY, how="inner")
            print(f"  joined on {KEY}")
            if len(mg):
                Mm = np.array(mg.ratings.tolist()).astype(int)
                ov = mg.oracle.values.astype(int)
                agree = np.concatenate([(ov[:, None] == Mm[:, [i for i in range(3) if i != j]]).mean(1)
                                        for j in range(3)])
                ci = boot_ci(agree)
                print(f"  n matched = {len(mg)}  "
                      f"({mg.model_label.nunique()} models: {sorted(mg.model_label.unique())})")
                print(f"  oracle vs other two: {agree.mean():.4f}  "
                      f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
                print(f"  human mean          : {np.mean(lo):.4f}")
                print(f"  margin              : {agree.mean()-np.mean(lo):+.4f}")
                if ci[0] < np.mean(lo):
                    print("  *** CI overlaps the human baseline; margin is not resolved ***")
                sp = mg[mg.ratings.apply(lambda r: 0 < sum(r) < 3)]
                if len(sp):
                    Ms = np.array(sp.ratings.tolist()).astype(int)
                    print(f"\n  on the {len(sp)} items where humans split: "
                          f"oracle matches majority {(sp.oracle.values == (Ms.sum(1) > 1.5)).mean():.3f}")
                print(f"\n  oracle pass rate {ov.mean():.3f} vs human {Mm.mean():.3f}")
                if abs(ov.mean() - Mm.mean()) > 0.05:
                    print("  *** systematic leniency or strictness; report this ***")
            else:
                print("  no overlap between oracle and labels on the composite key")
    else:
        print("  skipped, pass --oracle")


if __name__ == "__main__":
    main()
