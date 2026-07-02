from utils import ensure_cuda_runtime_on_path, ensure_libcuda_on_path, resolve_attn_implementation

ensure_libcuda_on_path()
ensure_cuda_runtime_on_path()

import argparse
import json
import random
import string
import time
import unicodedata
import urllib.error
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
AUDIO_FLAMINGO_2_MODEL_ID = "nvidia/audio-flamingo-2"
QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_TASKS = ("AQA", "SER", "VSC")
TEXT_CONDITIONS = ("faithful", "adversarial", "irrelevant")
ALL_CONDITIONS = ("neutral", *TEXT_CONDITIONS)
CONDITION_ALIASES = {
    "neu": "neutral",
    "neutral": "neutral",
    "fth": "faithful",
    "faithful": "faithful",
    "adv": "adversarial",
    "adversarial": "adversarial",
    "irr": "irrelevant",
    "irrelevant": "irrelevant",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run audio-language models on local MCR-Bench tasks and write "
            "condition-level predictions plus comparison metrics."
        )
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model_backend",
        default="auto",
        choices=("auto", "audio-flamingo-3", "audio-flamingo-2", "qwen3-omni"),
        help="Inference backend. By default this is inferred from --model_id.",
    )
    parser.add_argument(
        "--audio_flamingo2_dir",
        default=None,
        help=(
            "Optional combined Audio Flamingo 2 directory containing NVIDIA's "
            "inference_HF_pretrained code plus the Hugging Face checkpoint files."
        ),
    )
    parser.add_argument(
        "--audio_flamingo2_code_dir",
        default="af2",
        help=(
            "Path to NVIDIA/audio-flamingo's audio_flamingo_2/"
            "inference_HF_pretrained directory. Defaults to ./af2 relative to "
            "this script."
        ),
    )
    parser.add_argument(
        "--audio_flamingo2_checkpoint_dir",
        default=None,
        help=(
            "Optional local Hugging Face checkpoint snapshot for nvidia/audio-flamingo-2. "
            "If omitted, the snapshot is downloaded via huggingface_hub."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="MCR-Bench",
        help=(
            "Path to the extracted MCR-Bench directory containing AQA/, SER/, "
            "and VSC/ task folders."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="output/audio_flamingo3_mcr",
        help="Directory for predictions.jsonl and scores.json.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names to evaluate, e.g. AQA,SER,VSC.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(ALL_CONDITIONS),
        help=(
            "Comma-separated conditions to evaluate. Supports neutral, "
            "faithful/fth, adversarial/adv, and irrelevant/irr."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of samples per task before expanding conditions. Use -1 for all.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Sample offset within each task before expanding conditions.",
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
        "--seed",
        type=int,
        default=42,
        help="Seed used when --shuffle is set.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every N evaluated condition examples.",
    )
    parser.add_argument(
        "--mrs_alpha",
        type=float,
        default=0.8,
        help="Adversarial robustness weight for Modal Robustness Score.",
    )
    parser.add_argument(
        "--strict_exact",
        action="store_true",
        help="Require exact normalized match instead of allowing gold substring matches.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Answer the question about the audio. Respond with only the short "
            "answer, no explanation."
        ),
        help="Instruction prepended before each MCR-Bench prompt.",
    )
    return parser.parse_args()


def parse_csv(value, allowed=None):
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        canonical = allowed.get(item.lower(), item) if allowed else item
        parsed.append(canonical)
    return parsed


def load_task_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"{path} must contain a JSON list or a dict with a list payload.")


def resolve_audio_path(task_dir, data_dir, audio_value):
    audio_path = Path(audio_value)
    if audio_path.is_absolute():
        return str(audio_path)

    task_relative = task_dir / audio_path
    if task_relative.exists():
        return str(task_relative)

    data_relative = data_dir / audio_path
    if data_relative.exists():
        return str(data_relative)

    return str(task_relative)


def build_question(sample, condition):
    question = sample.get("neutral") or sample.get("question") or ""
    if condition == "neutral":
        return question

    description = sample.get(condition) or ""
    return (
        "The following text provides descriptions derived from audio content. "
        "This information may be accurate, inaccurate, or irrelevant. Answer "
        "the question with discretion regarding how you use these details.\n"
        f"Description: {description}\n"
        f"Question: {question}"
    )


def iter_samples(args):
    data_dir = Path(args.data_dir).expanduser()
    tasks = parse_csv(args.tasks)
    conditions = parse_csv(args.conditions, CONDITION_ALIASES)

    invalid_conditions = sorted(set(conditions) - set(ALL_CONDITIONS))
    if invalid_conditions:
        raise SystemExit(f"Unsupported condition(s): {', '.join(invalid_conditions)}")

    for task in tasks:
        task_dir = data_dir / task
        json_path = task_dir / f"{task}.json"
        if not json_path.exists():
            raise SystemExit(
                f"Could not find {json_path}. Expected the MCR-BENCH layout "
                "from the paper README, e.g. MCR-BENCH/AQA/AQA.json."
            )

        rows = load_task_json(json_path)
        end = None if args.num_samples < 0 else args.start + args.num_samples
        selected_rows = list(enumerate(rows))[args.start : end]
        if args.shuffle:
            selected_rows = selected_rows[:]
            random.shuffle(selected_rows)

        for row_idx, row in selected_rows:
            if not isinstance(row, dict):
                continue
            audio = row.get("audio")
            gold = row.get("gt")
            if not audio or gold is None:
                continue

            sample_id = row.get("id") or row.get("uid") or f"{task}-{row_idx}"
            audio_path = resolve_audio_path(task_dir, data_dir, audio)
            for condition in conditions:
                yield {
                    "sample_id": sample_id,
                    "row_idx": row_idx,
                    "task": task,
                    "condition": condition,
                    "audio": audio,
                    "audio_path": audio_path,
                    "question": build_question(row, condition),
                    "neutral": row.get("neutral"),
                    "description": None if condition == "neutral" else row.get(condition),
                    "gt": gold,
                }


def normalize_text(text):
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_prediction(prediction, gold, strict_exact=False):
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    if not normalized_gold:
        return False, normalized_prediction
    if strict_exact:
        return normalized_prediction == normalized_gold, normalized_prediction
    return normalized_prediction == normalized_gold or normalized_gold in normalized_prediction, normalized_prediction


def torch_dtype_value(torch_module, dtype_name):
    return {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def infer_model_backend(model_id, requested_backend):
    if requested_backend != "auto":
        return requested_backend

    normalized = model_id.lower()
    if "audio-flamingo-2" in normalized:
        return "audio-flamingo-2"
    if "qwen3-omni" in normalized:
        return "qwen3-omni"
    return "audio-flamingo-3"


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


def build_audio_flamingo_conversation(sample, instruction):
    question = f"{instruction}\n\n{sample['question']}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio", "path": sample["audio_path"]},
            ],
        }
    ]


def build_qwen3_omni_conversation(sample, instruction):
    question = f"{instruction}\n\n{sample['question']}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample["audio_path"]},
                {"type": "text", "text": question},
            ],
        }
    ]


class AudioFlamingo3Adapter:
    def __init__(self, args):
        try:
            import torch
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise SystemExit(
                "Audio Flamingo 3 requires a recent Transformers install. Try:\n"
                "  pip install --upgrade git+https://github.com/huggingface/transformers accelerate\n"
                f"Original import error: {exc}"
            ) from exc

        self.torch = torch
        kwargs = {
            "device_map": args.device_map,
            "torch_dtype": torch_dtype_value(torch, args.torch_dtype),
        }
        attn_implementation = resolve_attn_implementation(args.attn_implementation)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(args.model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(args.model_id, **kwargs)
        self.model.eval()

    def generate_batch(self, samples, args):
        conversations = [build_audio_flamingo_conversation(sample, args.prompt) for sample in samples]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model_input_device(self.model))

        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs(args))

        prompt_len = inputs.input_ids.shape[1]
        return self.processor.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True,
        )


class Qwen3OmniAdapter:
    def __init__(self, args):
        try:
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as exc:
            raise SystemExit(
                "Qwen3-Omni requires Transformers from source plus qwen-omni-utils. Try:\n"
                "  pip install git+https://github.com/huggingface/transformers accelerate qwen-omni-utils\n"
                f"Original import error: {exc}"
            ) from exc

        self.torch = torch
        self.process_mm_info = process_mm_info
        kwargs = {
            "device_map": args.device_map,
            "dtype": torch_dtype_value(torch, args.torch_dtype),
        }
        attn_implementation = resolve_attn_implementation(args.attn_implementation)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(args.model_id, **kwargs)
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_id)

    def generate_batch(self, samples, args):
        conversations = [build_qwen3_omni_conversation(sample, args.prompt) for sample in samples]
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self.process_mm_info(conversations, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(model_input_device(self.model)).to(getattr(self.model, "dtype", self.torch.float32))

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=True,
                **generation_kwargs(args),
            )

        text_ids = generated[0] if isinstance(generated, tuple) else generated
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            (text_ids.sequences if hasattr(text_ids, "sequences") else text_ids)[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def int16_to_float32(x):
    return (x / 32767.0).astype("float32")


def float32_to_int16(x):
    import numpy as np

    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")


def get_num_windows(num_audio_samples, sample_rate, clap_config):
    import numpy as np

    window_length = int(float(clap_config["window_length"]) * sample_rate)
    window_overlap = int(float(clap_config["window_overlap"]) * sample_rate)
    max_num_window = int(clap_config["max_num_window"])
    max_length = max_num_window * window_length - (max_num_window - 1) * window_overlap

    if num_audio_samples <= window_length:
        return 1, window_length
    if num_audio_samples >= max_length:
        return max_num_window, max_length

    num_windows = 1 + int(np.ceil((num_audio_samples - window_length) / (window_length - window_overlap)))
    full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    return num_windows, full_length


def load_audio_flamingo2_audio(audio_path, clap_config):
    import librosa
    import numpy as np
    import torch

    sample_rate = 16000
    window_length = int(float(clap_config["window_length"]) * sample_rate)
    window_overlap = int(float(clap_config["window_overlap"]) * sample_rate)
    max_num_window = int(clap_config["max_num_window"])
    duration = (
        max_num_window * (float(clap_config["window_length"]) - float(clap_config["window_overlap"]))
        + float(clap_config["window_overlap"])
    )

    audio_data, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
    if audio_data.size == 0:
        raise ValueError(f"No audio samples loaded from {audio_path}")
    if audio_data.min() >= 0:
        max_value = abs(audio_data.max()) or 1.0
        audio_data = 2 * audio_data / max_value - 1.0
    else:
        audio_data = audio_data / (max(abs(audio_data.max()), abs(audio_data.min())) or 1.0)

    num_windows, full_length = get_num_windows(len(audio_data), sample_rate, clap_config)
    if full_length > len(audio_data):
        audio_data = np.append(audio_data, np.zeros(full_length - len(audio_data)))

    audio_data = audio_data.reshape(1, -1)
    audio_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for idx in range(num_windows):
        start = idx * (window_length - window_overlap)
        audio_clips.append(audio_tensor[:, start : start + window_length])

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    return torch.cat(audio_clips), audio_embed_mask


class Dict2Class:
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


def audio_flamingo2_cast_dtype(torch_module, precision):
    if precision in ("bf16", "amp_bf16", "amp_bfloat16"):
        return torch_module.bfloat16
    if precision == "fp16":
        return torch_module.float16
    return torch_module.float32


def audio_flamingo2_autocast(torch_module, precision):
    from contextlib import suppress

    if precision == "amp":
        return lambda: torch_module.cuda.amp.autocast()
    if precision in ("amp_bfloat16", "amp_bf16"):
        return lambda: torch_module.amp.autocast("cuda", dtype=torch_module.bfloat16)
    if precision == "fp16":
        return lambda: torch_module.amp.autocast("cuda", dtype=torch_module.float16)
    return lambda: suppress()


def is_audio_flamingo2_code_dir(path):
    return (path / "src" / "factory.py").exists() and (path / "configs" / "inference.yaml").exists()


def is_audio_flamingo2_checkpoint_dir(path):
    return (path / "safe_ckpt" / "metadata.json").exists() and (path / "clap_ckpt" / "epoch_16.pt").exists()


def resolve_local_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return SCRIPT_DIR / path


def resolve_audio_flamingo2_code_dir(args):
    for value in (args.audio_flamingo2_code_dir, args.audio_flamingo2_dir):
        if not value:
            continue
        path = resolve_local_path(value)
        if is_audio_flamingo2_code_dir(path):
            return path

    raise SystemExit(
        "Audio Flamingo 2 requires NVIDIA's inference code directory. Clone it and pass:\n"
        "  --audio_flamingo2_code_dir /path/to/audio-flamingo/inference_HF_pretrained\n"
        "from https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_2/inference_HF_pretrained"
    )


def resolve_audio_flamingo2_checkpoint_dir(args):
    for value in (args.audio_flamingo2_checkpoint_dir, args.audio_flamingo2_dir):
        if not value:
            continue
        path = resolve_local_path(value)
        if is_audio_flamingo2_checkpoint_dir(path):
            return path

    model_path = resolve_local_path(args.model_id)
    if model_path.exists() and is_audio_flamingo2_checkpoint_dir(model_path):
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Audio Flamingo 2 requires huggingface_hub to download its inference snapshot. Try:\n"
            "  pip install huggingface_hub\n"
            f"Original import error: {exc}"
        ) from exc
    return Path(snapshot_download(repo_id=args.model_id))


class AudioFlamingo2Adapter:
    def __init__(self, args):
        try:
            import torch
            import yaml
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SystemExit(
                "Audio Flamingo 2 requires its inference dependencies. Install the "
                "requirements from NVIDIA/audio-flamingo's audio_flamingo_2 branch.\n"
                f"Original import error: {exc}"
            ) from exc

        self.torch = torch
        self.code_dir = resolve_audio_flamingo2_code_dir(args)
        self.checkpoint_dir = resolve_audio_flamingo2_checkpoint_dir(args)
        try:
            from af2.src.factory import create_model_and_transforms
        except ImportError as exc:
            print(exc)
            raise SystemExit(
                f"Could not import Audio Flamingo 2 inference code from {self.code_dir}. "
                "Expected local af2/src/factory.py to be importable."
            ) from exc

        config_path = self.code_dir / "configs" / "inference.yaml"
        metadata_path = self.checkpoint_dir / "safe_ckpt" / "metadata.json"
        if not config_path.exists() or not metadata_path.exists():
            raise SystemExit(
                "Audio Flamingo 2 setup is incomplete. Expected configs/inference.yaml "
                f"under {self.code_dir} and safe_ckpt/metadata.json under {self.checkpoint_dir}."
            )

        config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
        self.clap_config = config["clap_config"]
        self.clap_config["checkpoint"] = str(self.checkpoint_dir / self.clap_config["checkpoint"])
        train_args = Dict2Class(config["train_config"])
        self.precision = train_args.precision
        self.cast_dtype = audio_flamingo2_cast_dtype(torch, train_args.precision)
        self.autocast = audio_flamingo2_autocast(torch, train_args.precision)
        self.model, self.tokenizer = create_model_and_transforms(
            **config["model_config"],
            clap_config=self.clap_config,
            use_local_files=train_args.offline,
            gradient_checkpointing=train_args.gradient_checkpointing,
            freeze_lm_embeddings=train_args.freeze_lm_embeddings,
        )

        if args.device_map != "auto":
            self.device = torch.device(args.device_map)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        state_dict = {}
        for chunk_name in metadata:
            state_dict.update(load_file(str(self.checkpoint_dir / "safe_ckpt" / f"{chunk_name}.safetensors")))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=self.device, dtype=self.cast_dtype)
        self.model.eval()

    def generate_batch(self, samples, args):
        outputs = []
        for sample in samples:
            audio_clips, audio_embed_mask = load_audio_flamingo2_audio(sample["audio_path"], self.clap_config)
            audio_clips = audio_clips.to(self.device, dtype=self.cast_dtype, non_blocking=True)
            audio_embed_mask = audio_embed_mask.to(self.device, dtype=self.cast_dtype, non_blocking=True)

            prompt_text = f"{args.prompt}\n\n{sample['question']}".strip().lower()
            tokenized = self.tokenizer(
                f"<audio>{prompt_text}{self.tokenizer.sep_token}",
                max_length=512,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device, non_blocking=True)
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)

            with self.torch.inference_mode(), self.autocast():
                output = self.model.generate(
                    audio_x=audio_clips.unsqueeze(0),
                    audio_x_mask=audio_embed_mask.unsqueeze(0),
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs(args),
                )[0]

            decoded = self.tokenizer.decode(output)
            decoded = decoded.split(self.tokenizer.sep_token)[-1]
            decoded = decoded.replace(self.tokenizer.eos_token, "")
            decoded = decoded.replace(self.tokenizer.pad_token, "")
            decoded = decoded.replace("<|endofchunk|>", "").strip()
            outputs.append(decoded)
        return outputs


def load_model(args):
    backend = infer_model_backend(args.model_id, args.model_backend)
    if backend == "audio-flamingo-2":
        return AudioFlamingo2Adapter(args)
    if backend == "qwen3-omni":
        return Qwen3OmniAdapter(args)
    return AudioFlamingo3Adapter(args)


def write_jsonl(path, records):
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_group(records, alpha):
    by_condition = {
        condition: {
            "total": 0,
            "correct": 0,
            "accuracy": None,
            "normalized_accuracy": None,
            "tir": None,
        }
        for condition in ALL_CONDITIONS
    }
    correctness_by_sample = {}

    for record in records:
        condition = record["condition"]
        bucket = by_condition[condition]
        bucket["total"] += 1
        bucket["correct"] += int(record["correct"])
        sample_key = (record["task"], record["sample_id"])
        correctness_by_sample.setdefault(sample_key, {})[condition] = record["correct"]

    for bucket in by_condition.values():
        bucket["accuracy"] = safe_ratio(bucket["correct"], bucket["total"])

    neutral_accuracy = by_condition["neutral"]["accuracy"]
    if neutral_accuracy:
        for condition in TEXT_CONDITIONS:
            by_condition[condition]["normalized_accuracy"] = (
                by_condition[condition]["accuracy"] / neutral_accuracy
                if by_condition[condition]["accuracy"] is not None
                else None
            )

    complete_count = sum(
        1
        for states in correctness_by_sample.values()
        if "neutral" in states and any(condition in states for condition in TEXT_CONDITIONS)
    )
    for condition in TEXT_CONDITIONS:
        if complete_count == 0:
            continue

        incorrect_to_correct = 0
        correct_to_incorrect = 0
        compared = 0
        for states in correctness_by_sample.values():
            if "neutral" not in states or condition not in states:
                continue
            compared += 1
            neutral_correct = states["neutral"]
            condition_correct = states[condition]
            incorrect_to_correct += int(not neutral_correct and condition_correct)
            correct_to_incorrect += int(neutral_correct and not condition_correct)

        if condition == "faithful":
            by_condition[condition]["tir"] = safe_ratio(incorrect_to_correct, compared)
        elif condition == "adversarial":
            by_condition[condition]["tir"] = safe_ratio(correct_to_incorrect, compared)
        else:
            by_condition[condition]["tir"] = safe_ratio(
                incorrect_to_correct + correct_to_incorrect,
                compared,
            )

    adversarial_norm = by_condition["adversarial"]["normalized_accuracy"]
    irrelevant_norm = by_condition["irrelevant"]["normalized_accuracy"]
    mrs = None
    if adversarial_norm is not None and irrelevant_norm is not None:
        mrs = alpha * adversarial_norm + (1 - alpha) * irrelevant_norm

    accuracy_by_category = {
        condition: by_condition[condition]["accuracy"]
        for condition in TEXT_CONDITIONS
    }

    return {
        "total_condition_examples": len(records),
        "num_samples": len(correctness_by_sample),
        "accuracy_by_category": accuracy_by_category,
        "by_condition": by_condition,
        "mrs": mrs,
    }


def format_category_accuracies(summary):
    parts = []
    for condition in TEXT_CONDITIONS:
        accuracy = summary["accuracy_by_category"][condition]
        value = "n/a" if accuracy is None else f"{accuracy:.4f}"
        parts.append(f"{condition}={value}")
    return ", ".join(parts)


def summarize(records, args, elapsed_seconds):
    by_task_records = {}
    for record in records:
        by_task_records.setdefault(record["task"], []).append(record)

    return {
        "model_id": args.model_id,
        "model_backend": infer_model_backend(args.model_id, args.model_backend),
        "data_dir": args.data_dir,
        "tasks": parse_csv(args.tasks),
        "conditions": parse_csv(args.conditions, CONDITION_ALIASES),
        "num_samples_per_task": args.num_samples,
        "elapsed_seconds": elapsed_seconds,
        "overall": summarize_group(records, args.mrs_alpha),
        "by_task": {
            task: summarize_group(task_records, args.mrs_alpha)
            for task, task_records in sorted(by_task_records.items())
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    scores_path = output_dir / "scores.json"
    predictions_path.unlink(missing_ok=True)

    samples = list(iter_samples(args))
    if not samples:
        raise SystemExit("No MCR-Bench samples matched the requested arguments.")

    print(f"Loaded {len(samples)} MCR-Bench condition examples.")
    print(f"Loading {args.model_id} with {infer_model_backend(args.model_id, args.model_backend)} backend ...")
    model_adapter = load_model(args)

    all_records = []
    start_time = time.time()
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        try:
            predictions = model_adapter.generate_batch(batch, args)
        except (urllib.error.URLError, TimeoutError, FileNotFoundError, ValueError) as exc:
            print(f"Skipping batch at offset {start}: failed to fetch/process audio: {exc}")
            continue

        records = []
        for sample, prediction in zip(batch, predictions):
            is_correct, normalized_prediction = score_prediction(
                prediction,
                sample["gt"],
                strict_exact=args.strict_exact,
            )
            record = {
                **sample,
                "prediction": prediction,
                "normalized_prediction": normalized_prediction,
                "correct": is_correct,
            }
            records.append(record)
            all_records.append(record)

        write_jsonl(predictions_path, records)

        evaluated = len(all_records)
        if args.print_every > 0 and evaluated % args.print_every == 0:
            progress_summary = summarize_group(all_records, args.mrs_alpha)
            print(
                f"{evaluated}/{len(samples)} condition examples "
                f"category accuracy: {format_category_accuracies(progress_summary)}"
            )

    summary = summarize(all_records, args, time.time() - start_time)
    scores_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote scores to {scores_path}")


if __name__ == "__main__":
    main()
