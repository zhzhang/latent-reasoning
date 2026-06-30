import argparse
import json
import os
import random
import re
import string
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DATASET_ID = "ahabench/AHa-Bench"
DATASET_CONFIG = "default"
DATASET_SPLIT = "sample_with_audio"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"

YES_VALUES = {"yes", "y", "true", "是"}
NO_VALUES = {"no", "n", "false", "否"}
BINARY_RE = re.compile(r"\b(yes|no|true|false|y|n)\b|[是否]", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run NVIDIA Audio Flamingo 3 on AHa-Bench and write per-example "
            "predictions plus aggregate scores."
        )
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument(
        "--output_dir",
        default="output/audio_flamingo3_aha",
        help="Directory for predictions.jsonl and scores.json.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of examples to evaluate. Use -1 for the full split.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Dataset row offset to start from.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=("sdpa", "flash_attention_2"),
        help="Optional attention implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Device map passed to from_pretrained, e.g. "auto" or "cuda:0".',
    )
    parser.add_argument(
        "--question_types",
        default=None,
        help="Comma-separated AHa-Bench type filter, e.g. source_number,distance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when --shuffle is set.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print progress every N evaluated examples.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Answer the question about the audio. Respond with only the short "
            "answer, no explanation. For yes/no questions, answer only yes or no."
        ),
        help="Instruction prepended before each benchmark question.",
    )
    return parser.parse_args()


def request_json(url, params, timeout=120):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}")
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_audio_url(audio_cell):
    if isinstance(audio_cell, list) and audio_cell:
        first = audio_cell[0]
        if isinstance(first, dict):
            return first.get("src") or first.get("path")
    if isinstance(audio_cell, dict):
        return audio_cell.get("src") or audio_cell.get("path")
    if isinstance(audio_cell, str):
        return audio_cell
    return None


def iter_dataset_rows(args):
    wanted_types = None
    if args.question_types:
        wanted_types = {item.strip() for item in args.question_types.split(",") if item.strip()}

    yielded = 0
    offset = args.start
    page_size = 100
    target = None if args.num_samples < 0 else args.num_samples

    while target is None or yielded < target:
        length = page_size if target is None else min(page_size, target - yielded)
        data = request_json(
            DATASET_ROWS_URL,
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "offset": offset,
                "length": length,
            },
        )
        rows = data.get("rows", [])
        if not rows:
            break

        for item in rows:
            row = item["row"]
            offset = item["row_idx"] + 1
            if wanted_types and row.get("type") not in wanted_types:
                continue
            audio_url = extract_audio_url(row.get("audio"))
            if not audio_url:
                continue
            yield {
                "row_idx": item["row_idx"],
                "question_id": row.get("question_id"),
                "type": row.get("type"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "answer_details": row.get("answer_details"),
                "label": row.get("label"),
                "index": row.get("index"),
                "audio_text": row.get("audio_text"),
                "audio_url": audio_url,
            }
            yielded += 1
            if target is not None and yielded >= target:
                return

        total = data.get("num_rows_total")
        if total is not None and offset >= total:
            break


def normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def binary_label(text):
    normalized = normalize_text(text)
    if normalized in YES_VALUES:
        return "yes"
    if normalized in NO_VALUES:
        return "no"
    return None


def extract_binary_prediction(text):
    match = BINARY_RE.search(text or "")
    if not match:
        return None
    return binary_label(match.group(0))


def score_prediction(prediction, gold):
    gold_binary = binary_label(gold)
    if gold_binary is not None:
        predicted_binary = extract_binary_prediction(prediction)
        return predicted_binary == gold_binary, predicted_binary or ""

    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    if not normalized_gold:
        return False, normalized_prediction

    # ASR rows are often returned with wrapper text such as "The audio says ...".
    return normalized_gold in normalized_prediction, normalized_prediction


def load_model(model_id, torch_dtype, device_map, attn_implementation):
    try:
        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Audio Flamingo 3 requires a recent Transformers install. Try:\n"
            "  pip install --upgrade git+https://github.com/huggingface/transformers accelerate\n"
            f"Original import error: {exc}"
        ) from exc

    dtype_lookup = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs = {
        "device_map": device_map,
        "torch_dtype": dtype_lookup[torch_dtype],
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, processor


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def build_conversation(sample, instruction):
    question = f"{instruction}\n\nQuestion: {sample['question']}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio", "path": sample["audio_url"]},
            ],
        }
    ]


def generate_batch(model, processor, samples, args):
    import torch

    conversations = [build_conversation(sample, args.prompt) for sample in samples]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model_input_device(model))

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs.input_ids.shape[1]
    return processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )


def write_jsonl(path, records):
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    scores_path = output_dir / "scores.json"
    predictions_path.unlink(missing_ok=True)

    samples = list(iter_dataset_rows(args))
    if args.shuffle:
        random.shuffle(samples)
    if not samples:
        raise SystemExit("No AHa-Bench samples matched the requested arguments.")

    print(f"Loaded {len(samples)} AHa-Bench samples.")
    print(f"Loading {args.model_id} ...")
    model, processor = load_model(
        args.model_id,
        args.torch_dtype,
        args.device_map,
        args.attn_implementation,
    )

    correct = 0
    results_by_type = {}
    start_time = time.time()

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        try:
            predictions = generate_batch(model, processor, batch, args)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Skipping batch at offset {start}: failed to fetch/process audio: {exc}")
            continue

        records = []
        for sample, prediction in zip(batch, predictions):
            is_correct, normalized_prediction = score_prediction(prediction, sample["answer"])
            correct += int(is_correct)
            bucket = results_by_type.setdefault(sample["type"] or "unknown", {"correct": 0, "total": 0})
            bucket["correct"] += int(is_correct)
            bucket["total"] += 1

            record = {
                **sample,
                "prediction": prediction,
                "normalized_prediction": normalized_prediction,
                "correct": is_correct,
            }
            records.append(record)

        write_jsonl(predictions_path, records)

        evaluated = sum(bucket["total"] for bucket in results_by_type.values())
        if args.print_every > 0 and evaluated % args.print_every == 0:
            accuracy = correct / evaluated if evaluated else 0.0
            print(f"{evaluated}/{len(samples)} accuracy={accuracy:.4f}")

    total = sum(bucket["total"] for bucket in results_by_type.values())
    summary = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "elapsed_seconds": time.time() - start_time,
        "by_type": {
            key: {
                **value,
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(results_by_type.items())
        },
    }
    scores_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote scores to {scores_path}")


if __name__ == "__main__":
    main()
