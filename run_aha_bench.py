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
THINKING_END_TAGS = (
    "</" + "redacted_thinking" + ">",
    "</thinking>",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an audio-language model on AHa-Bench and write per-example "
            "predictions plus aggregate scores."
        )
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model_family",
        default="auto",
        choices=("auto", "audio_flamingo3", "qwen3_omni"),
        help=(
            "Model backend to use. auto infers from --model_id "
            "(audio-flamingo* -> audio_flamingo3, qwen3-omni* -> qwen3_omni)."
        ),
    )
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument(
        "--output_dir",
        default="output/aha_bench",
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
        default="sdpa",
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


def detect_model_family(model_id):
    lowered = model_id.lower()
    if "audio-flamingo" in lowered or "audio_flamingo" in lowered:
        return "audio_flamingo3"
    if "qwen3-omni" in lowered or "qwen3_omni" in lowered:
        return "qwen3_omni"
    raise SystemExit(
        "Could not infer --model_family from "
        f"{model_id!r}. Pass --model_family explicitly."
    )


def resolve_model_family(args):
    if args.model_family == "auto":
        return detect_model_family(args.model_id)
    return args.model_family


def is_thinking_model(model_id):
    return "thinking" in model_id.lower()


def apply_model_defaults(args, model_family):
    if model_family != "qwen3_omni" or not is_thinking_model(args.model_id):
        return

    # Qwen thinking checkpoints expect sampling from generation_config, not greedy.
    if args.temperature == 0.0 and args.top_p == 1.0:
        args.temperature = 0.6
        args.top_p = 0.95
    if args.max_new_tokens == 64:
        args.max_new_tokens = 2048


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


def strip_thinking_content(text):
    if not text:
        return ""
    text = str(text)
    for end_tag in THINKING_END_TAGS:
        if end_tag in text:
            text = text.split(end_tag, 1)[-1]
    text = re.sub(
        r"<" + "redacted_thinking" + ">.*",
        "",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


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


def load_model(model_id, model_family, torch_dtype, device_map, attn_implementation):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required: {exc}") from exc

    dtype_lookup = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_lookup[torch_dtype]
    kwargs = {"device_map": device_map}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if model_family == "audio_flamingo3":
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise SystemExit(
                "Audio Flamingo 3 requires a recent Transformers install. Try:\n"
                "  pip install --upgrade git+https://github.com/huggingface/transformers accelerate\n"
                f"Original import error: {exc}"
            ) from exc

        kwargs["torch_dtype"] = resolved_dtype
        processor = AutoProcessor.from_pretrained(model_id)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
        model.eval()
        return model, processor

    if model_family == "qwen3_omni":
        try:
            from qwen_omni_utils import process_mm_info
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as exc:
            raise SystemExit(
                "Qwen3-Omni requires qwen-omni-utils and a recent Transformers install. Try:\n"
                "  pip install qwen-omni-utils -U\n"
                "  pip install --upgrade git+https://github.com/huggingface/transformers accelerate\n"
                f"Original import error: {exc}"
            ) from exc

        kwargs["dtype"] = resolved_dtype
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        model.eval()
        return model, processor, process_mm_info

    raise SystemExit(f"Unsupported model family: {model_family}")


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def build_question_text(sample, instruction):
    return f"{instruction}\n\nQuestion: {sample['question']}"


def build_conversation(sample, instruction, model_family):
    question = build_question_text(sample, instruction)
    if model_family == "qwen3_omni":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": sample["audio_url"]},
                    {"type": "text", "text": question},
                ],
            }
        ]

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio", "path": sample["audio_url"]},
            ],
        }
    ]


def build_generation_kwargs(args, model=None, model_family=None):
    if model_family == "qwen3_omni":
        generation_kwargs = {
            "thinker_max_new_tokens": args.max_new_tokens,
            "thinker_do_sample": args.temperature > 0,
            "thinker_return_dict_in_generate": True,
            "return_audio": False,
            "use_audio_in_video": False,
        }
        if args.temperature > 0:
            generation_kwargs["thinker_temperature"] = args.temperature
            generation_kwargs["thinker_top_p"] = args.top_p
            if model is not None and hasattr(model, "generation_config"):
                config = model.generation_config
                if getattr(config, "top_k", None) not in (None, 0):
                    generation_kwargs["thinker_top_k"] = config.top_k
        return generation_kwargs

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    if model is not None and hasattr(model, "generation_config"):
        config = model.generation_config
        if args.temperature > 0 and getattr(config, "top_k", None) not in (None, 0):
            generation_kwargs["top_k"] = config.top_k
    return {key: value for key, value in generation_kwargs.items() if value is not None}


def decode_qwen3_omni_generate_output(generated, processor, prompt_len):
    import torch

    if isinstance(generated, tuple):
        generated = generated[0]

    if hasattr(generated, "sequences"):
        sequences = generated.sequences
    elif isinstance(generated, torch.Tensor):
        sequences = generated
    else:
        raise TypeError(f"Unexpected Qwen3-Omni generate output type: {type(generated)!r}")

    return processor.batch_decode(
        sequences[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_batch_audio_flamingo3(model, processor, samples, args):
    import torch

    conversations = [build_conversation(sample, args.prompt, "audio_flamingo3") for sample in samples]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model_input_device(model))

    generation_kwargs = build_generation_kwargs(args)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs.input_ids.shape[1]
    return processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )


def generate_batch_qwen3_omni(model, processor, process_mm_info, samples, args):
    import torch

    conversations = [build_conversation(sample, args.prompt, "qwen3_omni") for sample in samples]
    text = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model_input_device(model)).to(getattr(model, "dtype", torch.bfloat16))

    generation_kwargs = build_generation_kwargs(args, model=model, model_family="qwen3_omni")

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    return decode_qwen3_omni_generate_output(generated, processor, prompt_len)


def generate_batch(model, processor, samples, args, model_family, process_mm_info=None):
    if model_family == "audio_flamingo3":
        return generate_batch_audio_flamingo3(model, processor, samples, args)
    if model_family == "qwen3_omni":
        if process_mm_info is None:
            raise ValueError("process_mm_info is required for qwen3_omni generation.")
        return generate_batch_qwen3_omni(model, processor, process_mm_info, samples, args)
    raise ValueError(f"Unsupported model family: {model_family}")


def write_jsonl(path, records):
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    model_family = resolve_model_family(args)
    apply_model_defaults(args, model_family)

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
    print(f"Loading {args.model_id} ({model_family}) ...")
    loaded = load_model(
        args.model_id,
        model_family,
        args.torch_dtype,
        args.device_map,
        args.attn_implementation,
    )
    process_mm_info = None
    if model_family == "qwen3_omni":
        model, processor, process_mm_info = loaded
    else:
        model, processor = loaded

    correct = 0
    results_by_type = {}
    start_time = time.time()

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        try:
            predictions = generate_batch(
                model,
                processor,
                batch,
                args,
                model_family,
                process_mm_info=process_mm_info,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Skipping batch at offset {start}: failed to fetch/process audio: {exc}")
            continue

        records = []
        for sample, prediction in zip(batch, predictions):
            answer_text = strip_thinking_content(prediction)
            is_correct, normalized_prediction = score_prediction(answer_text, sample["answer"])
            correct += int(is_correct)
            bucket = results_by_type.setdefault(sample["type"] or "unknown", {"correct": 0, "total": 0})
            bucket["correct"] += int(is_correct)
            bucket["total"] += 1

            record = {
                **sample,
                "prediction": prediction,
                "answer_text": answer_text,
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
        "model_family": model_family,
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
