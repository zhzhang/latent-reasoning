from utils import (
    ensure_cuda_runtime_on_path,
    ensure_libcuda_on_path,
    ensure_mmar_audio,
    resolve_attn_implementation,
)

ensure_libcuda_on_path()
ensure_cuda_runtime_on_path()

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_META = SCRIPT_DIR / "MMAR-meta.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "audio_flamingo3_mmar"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THINK_SUFFIX = "Please think and reason about the input audio before you respond."
ANSWER_MARKERS = (
    r"therefore[, ]+the answer is[:\s]*",
    r"the answer is[:\s]*",
    r"final answer[:\s]*",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Audio Flamingo 3 on MMAR with think-mode CoT, write predictions "
            "for MMAR-Rubrics evaluation, and optionally score them."
        )
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--local_model_dir",
        default=None,
        help="Optional local snapshot directory for the AF3 checkpoint.",
    )
    parser.add_argument(
        "--meta",
        default=str(DEFAULT_META),
        help="Path to MMAR-meta.jsonl with rubrics and ground truth.",
    )
    parser.add_argument(
        "--data_root",
        default=str(SCRIPT_DIR),
        help="Repo root used to resolve ./audio paths and cache the MMAR archive.",
    )
    parser.add_argument(
        "--audio_dir",
        default=None,
        help="Directory containing MMAR wav files. Defaults to <data_root>/audio.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for predictions.jsonl and evaluated output.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of MMAR items to evaluate. Use -1 for all.",
    )
    parser.add_argument("--start", type=int, default=0, help="Offset into the meta file.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="Disable the AF-Think PEFT adapter and think suffix.",
    )
    parser.add_argument(
        "--force_download_audio",
        action="store_true",
        help="Re-download and extract the MMAR audio archive.",
    )
    parser.add_argument(
        "--skip_audio_download",
        action="store_true",
        help="Do not download MMAR audio; fail if clips are missing.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run evaluation_rubrics.py after inference completes.",
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="Skip inference and only run evaluation_rubrics.py on existing preds.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every N completed examples.",
    )
    return parser.parse_args()


def torch_dtype_value(torch_module, dtype_name):
    return {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def generation_kwargs(args):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, records, mode="a"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(data_root) / path).resolve())


def format_choices(choices):
    lines = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        lines.append(f"({label}) {choice}")
    return "\n".join(lines)


def build_mmar_prompt(item, use_think):
    question = str(item["question"]).strip()
    choices_block = format_choices(item["choices"])
    prompt = (
        f"{question}\n"
        "Choose the correct option from the following options:\n"
        f"{choices_block}"
    )
    if use_think:
        prompt += f"\n{THINK_SUFFIX}"
    return prompt


def build_conversation(audio_path, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]


def _normalize_for_match(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _match_choice_in_text(text, choices):
    normalized_text = _normalize_for_match(text)
    matched = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        patterns = (
            rf"\({label}\)\s*{re.escape(choice)}",
            rf"\b{re.escape(choice)}\b",
            rf"\({label}\)",
        )
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matched.append((index, choice))
                break

    if not matched:
        return None

    # Prefer the last-mentioned choice in the output tail.
    return matched[-1][1]


def parse_af3_output(raw_text, choices):
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    search_regions = []
    if lines:
        search_regions.append(lines[-1])
        search_regions.append("\n".join(lines[-3:]))
    search_regions.append(text)

    answer_prediction = None
    for region in search_regions:
        answer_prediction = _match_choice_in_text(region, choices)
        if answer_prediction:
            break

    if not answer_prediction:
        for marker in ANSWER_MARKERS:
            match = re.search(marker + r"(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            candidate = match.group(1).strip().splitlines()[0].strip()
            answer_prediction = _match_choice_in_text(candidate, choices) or candidate
            break

    if not answer_prediction and lines:
        answer_prediction = _match_choice_in_text(lines[-1], choices) or lines[-1]

    if not answer_prediction:
        answer_prediction = text
        return "", answer_prediction

    answer_index = text.lower().rfind(answer_prediction.lower())
    if answer_index > 0:
        thinking_prediction = text[:answer_index].strip()
    elif len(lines) > 1:
        thinking_prediction = "\n".join(lines[:-1]).strip()
    else:
        thinking_prediction = ""

    return thinking_prediction, answer_prediction


def load_audio_flamingo3(args):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    kwargs = {
        "device_map": args.device_map,
        "torch_dtype": torch_dtype_value(torch, args.torch_dtype),
    }
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    local_id = args.local_model_dir
    if not local_id:
        print(f"Downloading / resolving snapshot for {args.model_id} ...")
        local_id = snapshot_download(repo_id=args.model_id)
    else:
        local_id = str(Path(local_id).expanduser().resolve())

    processor = AutoProcessor.from_pretrained(local_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, **kwargs)

    if not args.no_think:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit(
                "AF-Think mode requires peft. Try:\n"
                "  pip install peft\n"
                f"Original import error: {exc}"
            ) from exc

        non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
        if not os.path.exists(non_lora_path):
            raise SystemExit(f"Think adapter weights not found at {non_lora_path}")

        print("Loading AF-Think PEFT adapter ...")
        non_lora_trainables = torch.load(
            non_lora_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(model, local_id, subfolder="think")

    model.eval()
    return model, processor


def generate_batch(model, processor, samples, args):
    import torch

    use_think = not args.no_think
    conversations = []
    for sample in samples:
        prompt_text = build_mmar_prompt(sample, use_think=use_think)
        conversations.append(build_conversation(sample["audio_path"], prompt_text))

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model_input_device(model))

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs(args))

    prompt_len = inputs.input_ids.shape[1]
    decoded = processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )

    results = []
    for sample, raw_output in zip(samples, decoded):
        thinking_prediction, answer_prediction = parse_af3_output(
            raw_output,
            sample["choices"],
        )
        results.append(
            {
                "model_output": raw_output,
                "thinking_prediction": thinking_prediction,
                "answer_prediction": answer_prediction,
            }
        )
    return results


def load_completed_ids(predictions_path):
    if not predictions_path.exists():
        return set()
    completed = set()
    for item in load_jsonl(predictions_path):
        record_id = item.get("id")
        if record_id:
            completed.add(record_id)
    return completed


def run_rubrics_scoring(predictions_path, meta_path, evaluated_path):
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping rubrics scoring.")
        print(
            "Run manually:\n"
            f"  export OPENAI_API_KEY=...\n"
            f"  python {SCRIPT_DIR / 'evaluation_rubrics.py'} "
            f"--input {predictions_path} --meta {meta_path} --output {evaluated_path}"
        )
        return False

    rubrics_script = SCRIPT_DIR / "evaluation_rubrics.py"
    cmd = [
        sys.executable,
        str(rubrics_script),
        "--input",
        str(predictions_path),
        "--meta",
        str(meta_path),
        "--output",
        str(evaluated_path),
    ]
    print("Running MMAR-Rubrics evaluation ...")
    subprocess.run(cmd, check=True)
    return True


def main():
    args = parse_args()
    random.seed(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    meta_path = Path(args.meta).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    evaluated_path = output_dir / "predictions.evaluated.jsonl"

    if args.score_only:
        if not predictions_path.exists():
            raise SystemExit(f"No predictions found at {predictions_path}")
        run_rubrics_scoring(predictions_path, meta_path, evaluated_path)
        return

    if not meta_path.exists():
        raise SystemExit(f"MMAR metadata not found: {meta_path}")

    meta_items = load_jsonl(meta_path)
    expected_wavs = len(meta_items)
    audio_dir = args.audio_dir or str(data_root / "audio")

    if args.skip_audio_download:
        wav_count = sum(
            1
            for path in Path(audio_dir).glob("*.wav")
            if path.is_file()
        ) if Path(audio_dir).exists() else 0
        if wav_count < expected_wavs:
            raise SystemExit(
                f"MMAR audio missing in {audio_dir} ({wav_count}/{expected_wavs} wav files). "
                "Re-run without --skip_audio_download."
            )
    else:
        ensure_mmar_audio(
            data_root=data_root,
            audio_dir=audio_dir,
            min_wav_files=expected_wavs,
            force_download=args.force_download_audio,
        )

    end = None if args.num_samples < 0 else args.start + args.num_samples
    selected_items = meta_items[args.start : end]
    completed_ids = load_completed_ids(predictions_path)
    pending_items = []
    for item in selected_items:
        audio_path = resolve_path(data_root, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {item['id']}: missing audio at {audio_path}")
            continue
        if item["id"] in completed_ids:
            continue
        pending_items.append({**item, "audio_path": audio_path})

    if not pending_items:
        print("No pending MMAR items to evaluate.")
        if args.score and predictions_path.exists():
            run_rubrics_scoring(predictions_path, meta_path, evaluated_path)
        return

    print(f"Evaluating {len(pending_items)} MMAR items with {args.model_id} ...")
    model, processor = load_audio_flamingo3(args)

    start_time = time.time()
    completed = 0
    for start in range(0, len(pending_items), args.batch_size):
        batch = pending_items[start : start + args.batch_size]
        try:
            outputs = generate_batch(model, processor, batch, args)
        except (OSError, ValueError, RuntimeError) as exc:
            ids = ", ".join(item["id"] for item in batch)
            print(f"Skipping batch ({ids}): {exc}")
            continue

        records = []
        for item, output in zip(batch, outputs):
            record = {
                **item,
                **output,
            }
            records.append(record)
            completed += 1
            if args.print_every > 0 and completed % args.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{completed}/{len(pending_items)}] "
                    f"id={item['id']} answer={output['answer_prediction']!r} "
                    f"({elapsed:.1f}s elapsed)"
                )

        write_jsonl(predictions_path, records, mode="a")

    print(f"Wrote predictions to {predictions_path}")
    if args.score:
        run_rubrics_scoring(predictions_path, meta_path, evaluated_path)


if __name__ == "__main__":
    main()
