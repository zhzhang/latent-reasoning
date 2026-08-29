"""Batch runner. Resumable, one JSONL row per judged generation."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import anthropic
from judge import judge


def load(labels_csv, generations_csv, audio_dir, min_raters=3):
    lab = pd.read_csv(labels_csv)
    gen = pd.read_csv(generations_csv)
    lab["ratings"] = lab["ratings"].apply(json.loads)
    lab = lab[lab["ratings"].apply(len) >= min_raters].copy()
    df = lab.merge(gen[["generation_id", "answer_prediction"]], on="generation_id", how="left")

    meta = {}
    for name in ("MMAR-meta.jsonl", "MMAR-meta.json"):
        p = Path(audio_dir).parent / name
        if p.exists():
            rows = ([json.loads(l) for l in p.open() if l.strip()]
                    if name.endswith("jsonl") else json.load(p.open()))
            meta = {str(r["id"]): r for r in rows}
            break
    if not meta:
        sys.exit(f"MMAR metadata not found next to {audio_dir}")

    index = {p.stem: str(p) for p in Path(audio_dir).rglob("*")
             if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}}

    df["question"] = df.question_id.map(lambda q: meta.get(q, {}).get("question"))
    df["category"] = df.question_id.map(lambda q: meta.get(q, {}).get("category"))
    df["modality"] = df.question_id.map(lambda q: meta.get(q, {}).get("modality"))
    df["audio_path"] = df.question_id.map(index.get)
    df["human_pass"] = df.ratings.apply(lambda r: sum(r) > len(r) / 2)
    df["human_split"] = df.ratings.apply(lambda r: 0 < sum(r) < len(r))

    bad = df.audio_path.isna() | df.question.isna() | df.answer_prediction.isna()
    if bad.any():
        print(f"dropping {bad.sum()} rows with missing audio, question, or answer")
    return df[~bad].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--generations", default="generations.csv")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["tools", "no_tools"],
                    help="both text-only; tools adds the four analysis tools as the "
                         "judge's only route to the clip")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY")

    df = load(args.labels, args.generations, args.audio_dir)
    if args.limit:
        df = df.head(args.limit)

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["generation_id"] for l in out.open() if l.strip()}
        print(f"resuming, {len(done)} already done")

    use_tools = args.condition == "tools"
    client = anthropic.Anthropic()

    with out.open("a") as f:
        for row in tqdm(df.itertuples(), total=len(df)):
            if row.generation_id in done:
                continue
            try:
                res = judge(client, row.audio_path, row.question,
                            row.answer_prediction, use_tools)
                err = None
            except Exception as exc:
                res, err = {}, f"{type(exc).__name__}: {exc}"
            f.write(json.dumps({
                "generation_id": int(row.generation_id),
                "question_id": row.question_id,
                "model_label": row.model_label,
                "category": row.category, "modality": row.modality,
                "condition": args.condition,
                "human_pass": bool(row.human_pass),
                "human_split": bool(row.human_split),
                "ratings": row.ratings,
                **res, "error": err,
            }) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
