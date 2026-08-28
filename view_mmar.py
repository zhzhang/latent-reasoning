"""Local viewer for the MMAR freeform-thinking pack.

Shows questions from ``outputs/mmar-freeform-thinking`` with audio, gold
reference, and stored shots. Grades each generation with Claude Sonnet 5
majority-of-3 verdicts from the llm-judge-gt pack (the
``claude-sonnet-5__neutral_with_gt_no_audio__gold`` run). Only models that
have generations in the freeform pack are listed.

Usage::

    uv run python view_mmar.py
    uv run python view_mmar.py --port 7860
    uv run python view_mmar.py --pack-dir ./outputs/mmar-freeform-thinking
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import tarfile
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from aggregate import MODEL_LABEL_ORDER
from collate_mmar_freeform import ALL_API_LABELS
from mmar_common import (
    ASSISTANT_THINK_OPEN,
    MUSIC_FLAMINGO_THINK_SUFFIX,
    build_mmar_freeform_prompt,
    count_wavs,
    load_jsonl,
)
from mmar_models import MODEL_SPECS, resolve_sampling

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK_DIR = REPO_ROOT / "outputs" / "mmar-freeform-thinking"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mmar"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"
DEFAULT_JUDGE_KEY = "claude-sonnet-5__neutral_with_gt_no_audio__gold"
DEFAULT_JUDGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "judge-quality"
    / "llm-judge-gt"
    / "_anthropic_batch"
    / DEFAULT_JUDGE_KEY
)
BATCH_DIR_NAMES = frozenset({"_anthropic_batch", "_openai_batch", "_batch"})
MMAR_REPO = "BoJack/MMAR"
MMAR_AUDIO_ARCHIVE = "mmar-audio.tar.gz"
MIN_MMAR_WAVS = 1000
LABEL_ORDER = MODEL_LABEL_ORDER + ALL_API_LABELS

# Official MMAR taxonomy (evaluation.py / run_judges.py).
MMAR_CATEGORIES = (
    "Signal Layer",
    "Perception Layer",
    "Semantic Layer",
    "Cultural Layer",
)
MMAR_MODALITIES = (
    "sound",
    "music",
    "speech",
    "mix-sound-music",
    "mix-sound-speech",
    "mix-music-speech",
    "mix-sound-music-speech",
)

# API models are not in MODEL_SPECS (run_experiment_api.API_SPECS).
API_SAMPLING: dict[str, dict[str, Any]] = {
    "gemini-3.7-flash": {
        "model_id": "gemini-3.7-flash",
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
}

# Where the sampling values were taken from (HF card, GitHub, vendor docs).
SAMPLING_SOURCES: dict[str, dict[str, str]] = {
    "af-next-think": {
        "url": "https://huggingface.co/nvidia/audio-flamingo-next-think-hf",
        "label": "Hugging Face model card",
        "note": (
            "Card generate() example uses repetition_penalty=1.2. "
            "max_tokens=2048 matches generation_config; T=0.2 added for n-shot variance."
        ),
    },
    "music-flamingo": {
        "url": "https://huggingface.co/nvidia/music-flamingo-hf",
        "label": "Hugging Face model card",
        "note": (
            "generation_config.json is greedy (max_new_tokens=2048). "
            "T=0.2 added for n-shot variance; card optional example uses T=0.7, top_p=0.9."
        ),
    },
    "mimo-audio-7b": {
        "url": (
            "https://github.com/XiaomiMiMo/MiMo-Audio/blob/main/"
            "src/mimo_audio/mimo_audio.py#L131-L133"
        ),
        "label": "MiMo-Audio GitHub sampler",
        "note": (
            "Official audio_understanding global sampler: T=0.3, top_p=0.95. "
            "repetition_penalty=1.1 matches vLLM-Omni deploy defaults."
        ),
    },
    "interactive-omni-8b": {
        "url": "https://huggingface.co/sensenova/InteractiveOmni-8B",
        "label": "Hugging Face model card",
        "note": (
            "README: generation_config = dict(max_new_tokens=1024, do_sample=True). "
            "Transformers defaults fill in T=1.0, top_p=1.0."
        ),
    },
    "qwen3-omni": {
        "url": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "label": "Hugging Face model card",
        "note": (
            "Thinking-mode card / generation_config: T=0.6, top_p=0.95, top_k=20. "
            "max_tokens capped at 2048 for MMAR (card example uses 16384)."
        ),
    },
    "voxtral-small-24b": {
        "url": "https://huggingface.co/mistralai/Voxtral-Small-24B-2507",
        "label": "Hugging Face model card",
        "note": "Card recommends temperature=0.2 and top_p=0.95 for audio-understanding chat.",
    },
    "qwen2.5-omni-7b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-Omni-7B",
        "label": "Hugging Face model card",
        "note": (
            "Official eval is greedy (generation_config has no sampler). "
            "T=0.2 added for n-shot variance."
        ),
    },
    "phi-4-multimodal": {
        "url": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
        "label": "Hugging Face model card",
        "note": (
            "Card uses GenerationConfig.from_pretrained + max_new_tokens=1000 (greedy). "
            "T=0.2 added for n-shot variance."
        ),
    },
    "gemma-4-e4b": {
        "url": "https://huggingface.co/google/gemma-4-E4B-it#best-practices",
        "label": "Hugging Face Best Practices",
        "note": "Card Best Practices: temperature=1.0, top_p=0.95, top_k=64.",
    },
    "gemma-4-12b": {
        "url": "https://huggingface.co/google/gemma-4-12B-it#best-practices",
        "label": "Hugging Face Best Practices",
        "note": "Card Best Practices: temperature=1.0, top_p=0.95, top_k=64.",
    },
    "qwen3-omni-instruct": {
        "url": "https://huggingface.co/marksverdhei/Qwen3-Omni-30B-A3B-FP8",
        "label": "Hugging Face model card",
        "note": (
            "Block-wise FP8 of Qwen3-Omni Instruct. Official Instruct eval is greedy. "
            "T=0.2 added for n-shot variance."
        ),
    },
    "nemotron-3-nano-omni": {
        "url": (
            "https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
            "#best-practices"
        ),
        "label": "Hugging Face Best Practices",
        "note": (
            "Thinking-mode card: T=0.6, top_p=0.95, max_tokens=20480, "
            "reasoning_budget=16384, grace_period=1024, max_model_len=210000."
        ),
    },
    "gemini-3.7-flash": {
        "url": "https://ai.google.dev/api/generate-content",
        "label": "Gemini API docs",
        "note": "Gemini generateContent default temperature is 1.0.",
    },
    "gpt-4o-mini": {
        "url": "https://platform.openai.com/docs/models/gpt-4o-mini",
        "label": "OpenAI model docs",
        "note": "OpenAI Chat Completions default temperature is 1.0.",
    },
}

SAMPLING_KEY_ORDER = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
    "thinking_token_budget",
)

# Official reported MMAR / MMAU averages. ``score`` is accuracy (%). Missing
# keys or ``score: None`` render as none. Only first-party cards/papers or the
# official MMAR leaderboard; no third-party re-evals, and MMAU* speech-synth
# (Voxtral) is not MMAU.
BENCHMARK_SCORES: dict[str, dict[str, dict[str, Any]]] = {
    "af-next-think": {
        "mmar": {
            "score": 61.0,
            "split": "overall",
            "url": "https://huggingface.co/nvidia/audio-flamingo-next-think-hf",
            "label": "Hugging Face model card",
        },
        "mmau": {
            "score": 75.01,
            "split": "v05.15.25 avg",
            "url": "https://huggingface.co/nvidia/audio-flamingo-next-think-hf",
            "label": "Hugging Face model card",
        },
    },
    "music-flamingo": {
        "mmar": {
            "score": 48.66,
            "split": "music",
            "url": "https://arxiv.org/abs/2511.10289",
            "label": "Music Flamingo paper",
        },
        "mmau": {
            "score": 76.83,
            "split": "music full-test",
            "url": "https://arxiv.org/abs/2511.10289",
            "label": "Music Flamingo paper",
        },
    },
    "mimo-audio-7b": {
        "mmar": {
            "score": 63.60,
            "split": "overall",
            "url": "https://arxiv.org/abs/2512.23808",
            "label": "MiMo-Audio paper",
        },
        "mmau": {
            "score": 74.90,
            "split": "overall",
            "url": "https://arxiv.org/abs/2512.23808",
            "label": "MiMo-Audio paper",
        },
    },
    "interactive-omni-8b": {
        "mmau": {
            "score": 67.39,
            "split": "overall",
            "url": "https://huggingface.co/sensenova/InteractiveOmni-8B",
            "label": "Hugging Face model card",
        },
    },
    "qwen3-omni": {
        "mmar": {
            "score": 66.4,
            "split": "overall",
            "url": "https://github.com/ddlBoJack/MMAR",
            "label": "MMAR leaderboard",
        },
        "mmau": {
            "score": 75.4,
            "split": "v05.15.25",
            "url": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking",
            "label": "Hugging Face model card",
        },
    },
    "qwen3-omni-instruct": {
        "mmau": {
            "score": 77.5,
            "split": "v05.15.25",
            "url": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "label": "Hugging Face model card",
        },
    },
    "qwen2.5-omni-7b": {
        "mmar": {
            "score": 56.7,
            "split": "overall",
            "url": "https://github.com/ddlBoJack/MMAR",
            "label": "MMAR leaderboard",
        },
        "mmau": {
            "score": 65.60,
            "split": "avg",
            "url": "https://huggingface.co/Qwen/Qwen2.5-Omni-7B",
            "label": "Hugging Face model card",
        },
    },
    "phi-4-multimodal": {
        "mmau": {
            "score": 55.56,
            "split": "overall",
            "url": "https://arxiv.org/abs/2503.01743",
            "label": "Phi-4-Mini technical report",
        },
    },
    "nemotron-3-nano-omni": {
        "mmau": {
            "score": 74.56,
            "split": "FP8, reasoning off",
            "url": "https://arxiv.org/abs/2604.24954",
            "label": "Nemotron 3 Nano Omni paper",
        },
    },
}

QUESTION_KEYS = (
    "id",
    "question",
    "answer",
    "choices",
    "audio_path",
    "url",
    "source",
    "modality",
    "category",
    "sub-category",
    "language",
    "thinking",
    "rubric",
    "cue",
)
SHOT_KEYS = (
    "shot_index",
    "answer_prediction",
    "model_output",
    "thinking_prediction",
    "correct",
)
JUDGE_ENTRY_KEYS = (
    "correct",
    "verdict",
    "output",
    "generation",
    "reasoning",
    "model_id",
    "prompt",
    "include_gold",
    "n_samples",
)

CONFIG: dict[str, Any] = {}

AF_NEXT_SYSTEM = (
    "You are Audio Flamingo-Next, a multimodal assistant for language and "
    "audio. On each turn you receive an optional audio clip which may contain "
    "speech, music, or ambient sounds and optional text, you will receive at "
    "least one or both; use your world knowledge and reasoning to help the "
    "user with any task. Interpret the entirety of the content of any input "
    "audio—regardless of whether the user calls it audio, speech, music, or "
    "sound."
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def ensure_mmar_audio(audio_dir: Path, *, force: bool = False) -> Path:
    """Download and extract the MMAR wav archive into ``audio_dir`` if needed."""
    audio_dir = audio_dir.expanduser().resolve()
    wav_count = count_wavs(audio_dir)
    if wav_count >= MIN_MMAR_WAVS and not force:
        print(f"MMAR audio ready: {wav_count} wav files in {audio_dir}", flush=True)
        return audio_dir

    from huggingface_hub import hf_hub_download

    cache_root = audio_dir.parent
    archive_cache = cache_root / MMAR_AUDIO_ARCHIVE
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"MMAR audio missing or incomplete ({wav_count} wavs); "
        f"downloading {MMAR_AUDIO_ARCHIVE} from {MMAR_REPO} ...",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="mmar-audio-") as tmp:
        tmp_root = Path(tmp)
        archive_tmp = Path(
            hf_hub_download(
                repo_id=MMAR_REPO,
                filename=MMAR_AUDIO_ARCHIVE,
                repo_type="dataset",
                local_dir=str(tmp_root / "download"),
            )
        )
        print(f"Extracting {archive_tmp.name} ...", flush=True)
        extract_dir = tmp_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_tmp, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        candidate_audio = extract_dir / "audio"
        if not candidate_audio.is_dir():
            matches = [path for path in extract_dir.rglob("audio") if path.is_dir()]
            if not matches:
                raise SystemExit(
                    f"No audio/ directory found after extracting {archive_tmp}"
                )
            candidate_audio = matches[0]

        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        shutil.copytree(candidate_audio, audio_dir)
        shutil.copy2(archive_tmp, archive_cache)

    wav_count = count_wavs(audio_dir)
    if wav_count < MIN_MMAR_WAVS:
        raise SystemExit(
            f"Expected at least {MIN_MMAR_WAVS} wav files in {audio_dir}, "
            f"found {wav_count}."
        )
    print(f"MMAR audio ready: {wav_count} wav files in {audio_dir}", flush=True)
    return audio_dir


def order_model_labels(labels: list[str]) -> list[str]:
    found = list(dict.fromkeys(str(x) for x in labels if x))
    known = [label for label in LABEL_ORDER if label in found]
    rest = [label for label in found if label not in LABEL_ORDER]
    return known + rest


def _ordered_unique(values: set[str], canonical: tuple[str, ...] = ()) -> list[str]:
    extras = sorted(name for name in values if name and name not in canonical)
    return [name for name in canonical if name in values] + extras


def _ordered_sampling(sampling: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in SAMPLING_KEY_ORDER:
        if key in sampling:
            out[key] = sampling[key]
    for key, value in sampling.items():
        if key not in out:
            out[key] = value
    return out


def _benchmark_entry(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    score = raw.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    return {
        "score": score_f,
        "split": raw.get("split") or "",
        "url": raw.get("url") or "",
        "label": raw.get("label") or "",
    }


def sampling_entries(model_labels: list[str]) -> list[dict[str, Any]]:
    """Sampling, MMAR/MMAU scores, and public source URLs for each pack model."""
    entries: list[dict[str, Any]] = []
    for label in model_labels:
        spec = MODEL_SPECS.get(label) or API_SAMPLING.get(label) or {}
        source = SAMPLING_SOURCES.get(label) or {}
        benches = BENCHMARK_SCORES.get(label) or {}
        model_id = str(spec.get("model_id") or label)
        if label in MODEL_SPECS:
            sampling = _ordered_sampling(resolve_sampling(label))
        else:
            sampling = _ordered_sampling(dict(spec.get("sampling") or {}))
        url = source.get("url") or ""
        if not url and "/" in model_id and not model_id.startswith("gpt-"):
            url = f"https://huggingface.co/{model_id}"
        entries.append(
            {
                "label": label,
                "model_id": model_id,
                "sampling": sampling,
                "source_url": url,
                "source_label": source.get("label") or "Source",
                "note": source.get("note") or "",
                "mmar": _benchmark_entry(benches.get("mmar")),
                "mmau": _benchmark_entry(benches.get("mmau")),
            }
        )
    return entries


def discover_model_labels(pack_dir: Path, manifest: dict[str, Any] | None = None) -> list[str]:
    models_root = pack_dir / "models"
    found: list[str] = []
    if models_root.is_dir():
        for child in sorted(models_root.iterdir()):
            pred = child / "predictions.jsonl"
            if child.is_dir() and pred.is_file() and pred.stat().st_size > 0:
                found.append(child.name)
    if not found and manifest:
        found = [str(label) for label in (manifest.get("models") or []) if label]
    return order_model_labels(found)


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _compact_judge_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    out = {key: entry.get(key) for key in JUDGE_ENTRY_KEYS if key in entry}
    if "correct" not in out and entry.get("correct") is not None:
        out["correct"] = entry.get("correct")
    if "verdict" not in out and out.get("correct") is not None:
        out["verdict"] = "pass" if out.get("correct") else "fail"
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        compacted_samples: list[dict[str, Any]] = []
        for item in samples:
            if not isinstance(item, dict):
                continue
            sample: dict[str, Any] = {}
            for key in ("correct", "verdict", "output", "generation", "reasoning"):
                if key in item:
                    sample[key] = item.get(key)
            if sample:
                compacted_samples.append(sample)
        if compacted_samples:
            out["samples"] = compacted_samples
            out.setdefault("n_samples", len(compacted_samples))
    return out


def _judge_key_from_dir(judge_dir: Path) -> str:
    name = judge_dir.name
    if name.count("__") >= 2:
        return name
    return DEFAULT_JUDGE_KEY


def _judge_pack_dir(judge_dir: Path) -> Path | None:
    """Pack that holds applied verdicts for a judge work dir or pack path."""
    if (judge_dir / "models").is_dir():
        return judge_dir
    if judge_dir.parent.name in BATCH_DIR_NAMES:
        pack = judge_dir.parent.parent
        if (pack / "models").is_dir():
            return pack
    return None


def _anthropic_output_text(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    result = row.get("result")
    if not isinstance(result, dict):
        return ""
    if str(result.get("type") or "") != "succeeded":
        return ""
    message = result.get("message")
    if not isinstance(message, dict):
        return ""
    parts: list[str] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _majority_entry_from_texts(
    texts: list[str],
    *,
    model_id: str,
    prompt_name: str,
    include_gold: bool,
) -> dict[str, Any] | None:
    from grader import _verdict_fields, format_grade_output, majority_grade_verdict

    if not any(texts):
        return None
    sample_fields = [_verdict_fields(text) for text in texts]
    raw = [item["grader_verdict_raw"] for item in sample_fields]
    majority = majority_grade_verdict(raw)
    generation = ""
    reasoning = ""
    for item, verdict in zip(sample_fields, raw):
        if majority is not None and verdict is majority:
            generation = item["generation"]
            reasoning = item.get("reasoning") or ""
            break
    if not generation and sample_fields:
        generation = sample_fields[0]["generation"]
        reasoning = sample_fields[0].get("reasoning") or ""
    return {
        "correct": bool(majority) if majority is not None else False,
        "verdict": (
            "pass" if majority is True else "fail" if majority is False else None
        ),
        "output": format_grade_output(majority),
        "generation": generation,
        "reasoning": reasoning,
        "model_id": model_id,
        "prompt": prompt_name,
        "include_gold": include_gold,
        "samples": [
            {
                "correct": item["correct"],
                "verdict": item["verdict"],
                "generation": item["generation"],
                "reasoning": item.get("reasoning") or "",
                "output": item["grader_output"],
            }
            for item in sample_fields
        ],
        "n_samples": len(sample_fields),
    }


def _load_grades_from_judge_pack(
    pack_dir: Path, judge_key: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    overlay: dict[tuple[str, str, int], dict[str, Any]] = {}
    models_root = pack_dir / "models"
    if not models_root.is_dir():
        return overlay
    for child in sorted(models_root.iterdir()):
        pred = child / "predictions.jsonl"
        if not child.is_dir() or not pred.is_file():
            continue
        label = child.name
        for record in load_jsonl(pred):
            qid = str(record.get("id") or "")
            if not qid:
                continue
            for shot in record.get("shots") or []:
                entry = _shot_judge_entry(shot, judge_key)
                compact = _compact_judge_entry(entry) if entry else None
                if compact is None:
                    continue
                overlay[(label, qid, _shot_index(shot))] = compact
    return overlay


def _load_grades_from_batch_dir(
    batch_dir: Path, judge_key: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    from grader import parse_judge_key

    jobs_path = batch_dir / "jobs.jsonl"
    output_path = batch_dir / "output.jsonl"
    if not jobs_path.is_file() or not output_path.is_file():
        return {}
    parsed = parse_judge_key(judge_key)
    model_id = parsed.get("model") or "claude-sonnet-5"
    prompt_name = parsed.get("prompt") or "neutral_with_gt_no_audio"
    include_gold = parsed.get("gold_tag") != "nongold"
    by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(output_path):
        cid = str(row.get("custom_id") or "")
        if cid:
            by_id[cid] = row
    overlay: dict[tuple[str, str, int], dict[str, Any]] = {}
    for job in load_jsonl(jobs_path):
        sample_ids = [
            str(cid)
            for cid in (job.get("sample_custom_ids") or [job.get("custom_id")])
            if cid
        ]
        if not sample_ids:
            continue
        texts = [_anthropic_output_text(by_id.get(cid)) for cid in sample_ids]
        entry = _majority_entry_from_texts(
            texts,
            model_id=model_id,
            prompt_name=prompt_name,
            include_gold=include_gold,
        )
        compact = _compact_judge_entry(entry) if entry else None
        if compact is None:
            continue
        for owner in job.get("owners") or []:
            if isinstance(owner, dict):
                gradee = str(owner.get("model") or "")
                qid = str(owner.get("qid") or "")
                shot_index = int(owner.get("shot_index", 0))
            else:
                gradee, qid, shot_index = owner[0], owner[1], int(owner[2])
            if not gradee or not qid:
                continue
            overlay[(gradee, qid, shot_index)] = compact
    return overlay


def load_judge_overlay(
    judge_dir: Path | None,
) -> tuple[str | None, dict[tuple[str, str, int], dict[str, Any]]]:
    """Load Claude (or other) majority-of-n verdicts keyed by model/qid/shot."""
    if judge_dir is None:
        return None, {}
    judge_dir = judge_dir.expanduser()
    if not judge_dir.exists():
        return None, {}
    judge_key = _judge_key_from_dir(judge_dir)
    overlay: dict[tuple[str, str, int], dict[str, Any]] = {}
    pack_dir = _judge_pack_dir(judge_dir)
    if pack_dir is not None:
        overlay = _load_grades_from_judge_pack(pack_dir, judge_key)
    if not overlay:
        overlay = _load_grades_from_batch_dir(judge_dir, judge_key)
    return (judge_key if overlay else None), overlay


_TOKEN_PIECE_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\s]",
    re.UNICODE,
)
_GENERATED_TOKEN_ROLES = frozenset({"generated", "output", "assistant"})


def _approx_n_tokens(text: str) -> int:
    return len(_TOKEN_PIECE_RE.findall(text or ""))


def _count_raw_token_list(raw: Any) -> int | None:
    if not isinstance(raw, list) or not raw:
        return None
    if all(isinstance(token, int) for token in raw):
        return len(raw)
    if all(isinstance(token, dict) for token in raw):
        generated = [
            token
            for token in raw
            if str(token.get("role") or "").lower() in _GENERATED_TOKEN_ROLES
        ]
        return len(generated) if generated else len(raw)
    return len(raw)


def _count_generated_tokens(shot: dict[str, Any]) -> int:
    counted = _count_raw_token_list(shot.get("raw_tokens"))
    if counted is not None:
        return counted
    stored = shot.get("n_tokens")
    if isinstance(stored, int) and stored >= 0:
        return stored
    return _approx_n_tokens(str(shot.get("model_output") or ""))


def _compact_shot(shot: dict[str, Any]) -> dict[str, Any]:
    out = {key: shot.get(key) for key in SHOT_KEYS if key in shot}
    out["shot_index"] = _shot_index(shot)
    out["n_tokens"] = _count_generated_tokens(shot)
    judges = shot.get("judges")
    if isinstance(judges, dict) and judges:
        compacted: dict[str, dict[str, Any]] = {}
        for key, entry in judges.items():
            compact = _compact_judge_entry(entry)
            if compact is not None:
                compacted[str(key)] = compact
        if compacted:
            out["judges"] = compacted
    return out


def _resolve_judge_key(
    wanted: str | None,
    available: list[str] | tuple[str, ...] | set[str],
) -> str | None:
    """Map a bare judge label onto a composite ``label__prompt__gold`` key.

    Manifest ``primary_judge`` is sometimes the model slug (``gemini-3.7-flash``)
        while shot verdicts are stored under ``gemini-3.7-flash__free__nongold``.
    """
    keys = [str(key) for key in available if key]
    if not wanted:
        return None
    wanted_s = str(wanted)
    if wanted_s in keys:
        return wanted_s
    prefix = wanted_s + "__"
    matches = [key for key in keys if key.startswith(prefix)]
    if matches:
        return matches[0]
    return wanted_s


def _shot_judge_entry(shot: dict[str, Any], judge_key: str | None) -> dict[str, Any] | None:
    judges = shot.get("judges") or {}
    if not judge_key or not isinstance(judges, dict):
        return None
    entry = judges.get(judge_key)
    if isinstance(entry, dict):
        return entry
    resolved = _resolve_judge_key(judge_key, judges.keys())
    entry = judges.get(resolved) if resolved else None
    return entry if isinstance(entry, dict) else None


def _shot_correct(shot: dict[str, Any], judge_key: str | None = None) -> bool | None:
    if judge_key:
        entry = _shot_judge_entry(shot, judge_key)
        if isinstance(entry, dict) and "correct" in entry:
            value = entry.get("correct")
            if value is None:
                return None
            return bool(value)
        return None
    value = shot.get("correct")
    if value is None:
        return None
    return bool(value)


def _sample_disagreement(entry: dict[str, Any]) -> float | None:
    """1 - max(pass rate, fail rate) across majority-vote samples."""
    samples = entry.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    n_pass = 0
    n = 0
    for item in samples:
        if not isinstance(item, dict):
            continue
        value = item.get("correct")
        if value is None:
            continue
        n += 1
        if value:
            n_pass += 1
    if n == 0:
        return None
    pct_pass = n_pass / n
    return 1.0 - max(pct_pass, 1.0 - pct_pass)


def _shot_disagreement(shot: dict[str, Any]) -> float | None:
    """1 - max(pass rate, fail rate) on one generation.

    Prefers the judge's n-sample votes (Claude majority-of-3) when present.
    Otherwise uses the split across judges. Unanimous is 0; a 50/50 split
    is 0.5. Shots with no scored votes return None.
    """
    judges = shot.get("judges")
    if not isinstance(judges, dict) or not judges:
        return None
    for entry in judges.values():
        if isinstance(entry, dict):
            sample_value = _sample_disagreement(entry)
            if sample_value is not None:
                return sample_value
    n_pass = 0
    n = 0
    for entry in judges.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("correct")
        if value is None:
            continue
        n += 1
        if value:
            n_pass += 1
    if n == 0:
        return None
    pct_pass = n_pass / n
    return 1.0 - max(pct_pass, 1.0 - pct_pass)


def _mean_disagreement(records: list[dict[str, Any]]) -> float | None:
    """Average ``_shot_disagreement`` over every graded generation."""
    values: list[float] = []
    for record in records:
        for shot in record.get("shots") or []:
            value = _shot_disagreement(shot)
            if value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _judge_stats(
    shots: list[dict[str, Any]],
    judge_key: str,
    stored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stored = stored or {}
    shot_keys = [
        key
        for shot in shots
        for key in (shot.get("judges") or {})
    ]
    resolved = _resolve_judge_key(judge_key, list(stored) + shot_keys) or judge_key
    row = stored.get(resolved) or stored.get(judge_key) or {}
    n_correct = row.get("n_shot_correct")
    rate = row.get("shot_success_rate")
    n_shots = row.get("n_shots")
    if n_correct is None or rate is None:
        scored = [_shot_correct(shot, resolved) for shot in shots]
        present = [value for value in scored if value is not None]
        n_shots = len(present)
        n_correct = sum(1 for value in present if value) if present else None
        rate = (n_correct / n_shots) if n_shots and n_correct is not None else None
    try:
        rate_f = float(rate) if rate is not None else None
    except (TypeError, ValueError):
        rate_f = None
    try:
        n_correct_i = int(n_correct) if n_correct is not None else None
    except (TypeError, ValueError):
        n_correct_i = None
    try:
        n_shots_i = int(n_shots) if n_shots is not None else len(shots)
    except (TypeError, ValueError):
        n_shots_i = len(shots)
    return {
        "shot_success_rate": rate_f,
        "n_shot_correct": n_correct_i,
        "n_shots": n_shots_i,
    }


def _pack_judge_entries(
    manifest: dict[str, Any],
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str | None]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    shot_keys: set[str] = set()
    primary = manifest.get("primary_judge")
    flagged_primary: str | None = None

    def _add(
        label: str,
        *,
        model_id: Any = None,
        prompt: Any = None,
        include_gold: Any = None,
        is_primary: bool = False,
    ) -> None:
        key = str(label or "")
        if not key or key in seen:
            return
        seen.add(key)
        entry = {
            "label": key,
            "model_id": model_id,
            "prompt": prompt,
            "include_gold": include_gold,
            "primary": bool(is_primary) or key == primary,
        }
        entries.append(entry)

    for raw in manifest.get("judges") or []:
        if isinstance(raw, dict) and raw.get("label"):
            label = str(raw["label"])
            if raw.get("primary") and not flagged_primary:
                flagged_primary = label
            _add(
                label,
                model_id=raw.get("model_id"),
                prompt=raw.get("prompt"),
                include_gold=raw.get("include_gold"),
                is_primary=bool(raw.get("primary")),
            )
        elif raw:
            _add(str(raw))

    for by_id in predictions.values():
        for record in by_id.values():
            for shot in record.get("shots") or []:
                for label, entry in (shot.get("judges") or {}).items():
                    shot_keys.add(str(label))
                    extra = entry if isinstance(entry, dict) else {}
                    _add(
                        str(label),
                        model_id=extra.get("model_id"),
                        prompt=extra.get("prompt"),
                        include_gold=extra.get("include_gold"),
                    )

    for by_id in predictions.values():
        for record in by_id.values():
            for label in record.get("judges") or []:
                key = str(label)
                # Skip record-level slugs that never appear on shots when a
                # composite ``slug__prompt__gold`` key already exists.
                if key not in shot_keys and any(
                    existing.startswith(key + "__") for existing in shot_keys
                ):
                    continue
                _add(key)

    if str(primary or "") not in seen:
        if flagged_primary and flagged_primary in seen:
            primary = flagged_primary
        else:
            resolved = _resolve_judge_key(primary, seen)
            primary = resolved if resolved in seen else (entries[0]["label"] if entries else None)
    if not primary and entries:
        primary = entries[0]["label"]
    for entry in entries:
        entry["primary"] = entry["label"] == primary
    if primary:
        entries.sort(key=lambda row: (0 if row["label"] == primary else 1, row["label"]))
    return entries, str(primary) if primary else None


def _shot_success(
    record: dict[str, Any],
    judge_key: str | None = None,
) -> tuple[float | None, int | None, int]:
    shots = list(record.get("shots") or [])
    n_shots = len(shots)
    if judge_key:
        stats = _judge_stats(shots, judge_key, record.get("per_judge") or {})
        return stats["shot_success_rate"], stats["n_shot_correct"], stats["n_shots"] or n_shots
    n_correct = record.get("n_shot_correct")
    rate = record.get("shot_success_rate")
    if n_correct is None and shots:
        n_correct = sum(1 for shot in shots if _shot_correct(shot) is True)
    if rate is None and n_shots:
        n_correct_i = int(n_correct or 0)
        rate = n_correct_i / n_shots
    try:
        rate_f = float(rate) if rate is not None else None
    except (TypeError, ValueError):
        rate_f = None
    try:
        n_correct_i = int(n_correct) if n_correct is not None else None
    except (TypeError, ValueError):
        n_correct_i = None
    return rate_f, n_correct_i, n_shots


def _question_fields(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: record.get(key) for key in QUESTION_KEYS if key in record}
    if not out.get("sub-category") and record.get("sub_category"):
        out["sub-category"] = record.get("sub_category")
    return out


def build_model_prompts(item: dict[str, Any]) -> dict[str, str]:
    """Reconstruct the freeform text prompts sent to each model."""
    base = build_mmar_freeform_prompt(item)
    af_base = build_mmar_freeform_prompt(item, with_timestamps=True)
    mf_base = build_mmar_freeform_prompt(item, think_suffix=MUSIC_FLAMINGO_THINK_SUFFIX)
    return {
        "shared": base,
        "af-next-think": (
            f"<|im_start|>system\n{AF_NEXT_SYSTEM}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<sound>{af_base}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{ASSISTANT_THINK_OPEN}"
        ),
        "music-flamingo": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<sound>{mf_base}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{ASSISTANT_THINK_OPEN}"
        ),
        "mimo-audio-7b": (
            "<|im_start|>user\n"
            f"<|sosp|><|empty|><|eosp|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{ASSISTANT_THINK_OPEN}"
        ),
        "interactive-omni-8b": f"[audio attached]\n{base}",
        "qwen3-omni": (
            "<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "qwen3-omni-instruct": (
            "<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "qwen2.5-omni-7b": (
            "<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "voxtral-small-24b": f"[audio attached]\n{base}",
        "gemini-3.7-flash": base,
        "gpt-4o-mini": base,
    }


def resolve_audio(audio_path: str | None) -> Path | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    if path.is_file():
        return path
    name = path.name
    candidates = [
        Path(CONFIG["audio_dir"]) / name,
        REPO_ROOT / "data" / "mmar" / "audio" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _apply_judge_overlay(
    shots: list[dict[str, Any]],
    *,
    model: str,
    qid: str,
    overlay: dict[tuple[str, str, int], dict[str, Any]],
    judge_key: str,
) -> None:
    for shot in shots:
        entry = overlay.get((model, qid, _shot_index(shot)))
        if entry is None:
            if isinstance(shot.get("judges"), dict):
                shot["judges"] = {
                    key: value
                    for key, value in shot["judges"].items()
                    if key == judge_key
                }
            continue
        shot["judges"] = {judge_key: entry}
        if entry.get("correct") is not None:
            shot["correct"] = bool(entry.get("correct"))


@lru_cache(maxsize=2)
def load_pack(pack_dir_s: str, judge_dir_s: str = "") -> dict[str, Any]:
    pack_dir = Path(pack_dir_s)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)
    manifest = load_json(pack_dir / "manifest.json")
    ids_payload = load_json(pack_dir / "question_ids.json")
    model_labels = discover_model_labels(pack_dir, manifest)
    overlay_key, overlay = load_judge_overlay(Path(judge_dir_s) if judge_dir_s else None)

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for label in model_labels:
        rows = load_jsonl(pack_dir / "models" / label / "predictions.jsonl")
        by_id: dict[str, dict[str, Any]] = {}
        for record in rows:
            qid = str(record.get("id") or "")
            if not qid:
                continue
            shots = [_compact_shot(shot) for shot in (record.get("shots") or [])]
            shots.sort(key=_shot_index)
            if overlay_key:
                _apply_judge_overlay(
                    shots,
                    model=label,
                    qid=qid,
                    overlay=overlay,
                    judge_key=overlay_key,
                )
            if overlay_key:
                rate, n_correct, n_shots = _shot_success(
                    {"shots": shots, "per_judge": {}},
                    overlay_key,
                )
            else:
                rate, n_correct, n_shots = _shot_success({**record, "shots": shots})
            if overlay_key:
                judge_keys = [overlay_key]
            else:
                judge_keys = [
                    str(x)
                    for x in (record.get("judges") or [])
                    if x
                ]
                for shot in shots:
                    for key in (shot.get("judges") or {}):
                        if key not in judge_keys:
                            judge_keys.append(key)
            stored_per_judge = {} if overlay_key else (record.get("per_judge") or {})
            per_judge = {
                key: _judge_stats(shots, key, stored_per_judge) for key in judge_keys
            }
            by_id[qid] = {
                **_question_fields(record),
                "model": label,
                "source_run_id": record.get("source_run_id"),
                "n_shots": n_shots or record.get("n_shots") or len(shots),
                "n_shot_correct": n_correct,
                "shot_success_rate": rate,
                "judges": judge_keys,
                "primary_judge": overlay_key or record.get("primary_judge"),
                "per_judge": per_judge,
                "shots": shots,
            }
        predictions[label] = by_id

    model_labels = [
        label
        for label in model_labels
        if any((record.get("shots") or []) for record in predictions.get(label, {}).values())
    ]
    predictions = {label: predictions[label] for label in model_labels}

    judge_entries, primary_judge = _pack_judge_entries(manifest, predictions)
    if overlay_key:
        primary_judge = overlay_key
        if not any(entry.get("label") == overlay_key for entry in judge_entries):
            judge_entries.insert(
                0,
                {
                    "label": overlay_key,
                    "model_id": overlay_key.split("__")[0],
                    "prompt": "neutral_with_gt_no_audio",
                    "include_gold": True,
                    "primary": True,
                },
            )
        for entry in judge_entries:
            entry["primary"] = entry["label"] == overlay_key
        judge_entries.sort(
            key=lambda row: (0 if row["label"] == overlay_key else 1, row["label"])
        )
        judge_entries = [entry for entry in judge_entries if entry["label"] == overlay_key]

    preferred_ids = [str(qid) for qid in (ids_payload.get("ids") or []) if qid]
    seen: set[str] = set(preferred_ids)
    all_ids = list(preferred_ids)
    for label in model_labels:
        for qid in predictions[label]:
            if qid not in seen:
                seen.add(qid)
                all_ids.append(qid)

    questions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    modalities: set[str] = set()
    categories: set[str] = set()
    subcategories: set[str] = set()
    category_subs: dict[str, set[str]] = {}
    n_complete = 0
    for qid in all_ids:
        sample: dict[str, Any] | None = None
        per_model: dict[str, dict[str, Any]] = {}
        rates: list[float] = []
        n_present = 0
        for label in model_labels:
            record = predictions[label].get(qid)
            if record is None:
                per_model[label] = {"present": False}
                continue
            n_present += 1
            if sample is None:
                sample = record
            rate = record.get("shot_success_rate")
            if rate is not None:
                rates.append(float(rate))
            per_model[label] = {
                "present": True,
                "shot_success_rate": rate,
                "n_shot_correct": record.get("n_shot_correct"),
                "n_shots": record.get("n_shots"),
                "per_judge": record.get("per_judge") or {},
            }
        if sample is None:
            continue
        avg = sum(rates) / len(rates) if rates else None
        present_records = [
            predictions[label][qid]
            for label in model_labels
            if predictions[label].get(qid) is not None
        ]
        avg_disagreement = _mean_disagreement(present_records)
        complete = n_present >= len(model_labels) and len(model_labels) > 0
        if complete:
            n_complete += 1
        has_grades = False
        for label in model_labels:
            record = predictions[label].get(qid)
            if record is None:
                continue
            for shot in record.get("shots") or []:
                for entry in (shot.get("judges") or {}).values():
                    if isinstance(entry, dict) and entry.get("correct") is not None:
                        has_grades = True
                        break
                if has_grades:
                    break
            if has_grades:
                break
        modality = str(sample.get("modality") or "")
        category = str(sample.get("category") or "")
        subcat = str(sample.get("sub-category") or "")
        if modality:
            modalities.add(modality)
        if category:
            categories.add(category)
        if subcat:
            subcategories.add(subcat)
        if category and subcat:
            category_subs.setdefault(category, set()).add(subcat)
        row = {
            "id": qid,
            "question": sample.get("question") or "",
            "answer": sample.get("answer") or "",
            "choices": sample.get("choices") or [],
            "audio_path": sample.get("audio_path"),
            "url": sample.get("url") or "",
            "source": sample.get("source") or "",
            "modality": modality,
            "category": category,
            "sub-category": subcat,
            "language": sample.get("language") or "",
            "thinking": sample.get("thinking") or "",
            "rubric": sample.get("rubric") or "",
            "cue": sample.get("cue") or "",
            "avg_success_rate": avg,
            "avg_disagreement": avg_disagreement,
            "n_models": n_present,
            "n_models_total": len(model_labels),
            "complete": complete,
            "has_grades": has_grades,
            "per_model": per_model,
        }
        questions.append(
            {
                "id": qid,
                "question": row["question"],
                "modality": modality,
                "category": category,
                "sub-category": subcat,
                "avg_success_rate": avg,
                "avg_disagreement": avg_disagreement,
                "n_models": n_present,
                "n_models_total": len(model_labels),
                "complete": complete,
                "has_grades": has_grades,
                "per_model": per_model,
            }
        )
        by_id[qid] = row

    questions.sort(
        key=lambda row: (
            row["avg_success_rate"] is None,
            row["avg_success_rate"] if row["avg_success_rate"] is not None else 1.0,
            str(row["id"]),
        )
    )

    progress = manifest.get("progress") or {}
    coverage = []
    for label in model_labels:
        n_done = len(predictions[label])
        row = progress.get(label) or {}
        n_shot_correct = 0
        n_graded_shots = 0
        for record in predictions[label].values():
            for shot in record.get("shots") or []:
                value = _shot_correct(shot, overlay_key or primary_judge)
                if value is None:
                    continue
                n_graded_shots += 1
                if value:
                    n_shot_correct += 1
        accuracy = (
            n_shot_correct / n_graded_shots if n_graded_shots else None
        )
        coverage.append(
            {
                "model": label,
                "n_done": n_done,
                "n_total": int(row.get("n_total") or ids_payload.get("n") or len(questions)),
                "complete": bool(row.get("complete")) if "complete" in row else n_done >= len(questions),
                "source_run_id": row.get("source_run_id"),
                "n_shot_correct": n_shot_correct if n_graded_shots else None,
                "n_graded": n_graded_shots,
                "accuracy": accuracy,
            }
        )

    return {
        "pack_dir": str(pack_dir),
        "manifest": manifest,
        "model_labels": model_labels,
        "predictions": predictions,
        "questions": questions,
        "by_id": by_id,
        "modalities": _ordered_unique(modalities, MMAR_MODALITIES),
        "categories": _ordered_unique(categories, MMAR_CATEGORIES),
        "subcategories": _ordered_unique(subcategories),
        "category_subcategories": {
            name: _ordered_unique(category_subs.get(name) or set())
            for name in _ordered_unique(categories, MMAR_CATEGORIES)
        },
        "n_shots": int(manifest.get("n_shots") or ids_payload.get("n_shots") or 5),
        "coverage": coverage,
        "n_complete": n_complete,
        "n_graded": sum(1 for row in questions if row.get("has_grades")),
        "n_questions": len(questions),
        "judges": judge_entries,
        "primary_judge": primary_judge,
        "judge_dir": judge_dir_s,
        "n_overlay_grades": len(overlay),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Freeform</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e7eef3;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --good: #1f6b4a;
    --bad: #9b3a3a;
    --soft-good: #d7eee3;
    --soft-bad: #f3dede;
    --shadow: 0 1px 0 rgba(20,32,42,0.04), 0 10px 28px rgba(20,32,42,0.07);
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    color: var(--ink);
    background:
      linear-gradient(160deg, #dfeaf1 0%, transparent 42%),
      linear-gradient(345deg, #cfdde6 0%, transparent 36%),
      var(--bg);
  }
  header {
    position: sticky; top: 0; z-index: 20;
    backdrop-filter: blur(10px);
    background: color-mix(in srgb, var(--bg) 82%, transparent);
    border-bottom: 1px solid var(--line);
    padding: 1rem 1.25rem;
  }
  .header-inner {
    max-width: 1400px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: end;
    justify-content: space-between;
  }
  .brand h1 {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
    font-weight: 600; font-size: 1.4rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"], button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem;
  }
  select, input[type="search"] { min-width: 10rem; }
  select.filter-select { max-width: 14rem; }
  button { cursor: pointer; }
  button.active { background: #e2eef6; border-color: #8fb3c9; }
  main {
    max-width: 1400px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 360px 1fr; gap: 1rem;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
    min-height: 0;
  }
  .panel h2 {
    margin: 0; padding: 0.85rem 1rem;
    font-size: 0.85rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line);
  }
  .stats {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--line);
    font-size: 0.85rem; color: var(--muted);
  }
  .stats strong { color: var(--ink); }
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 220px); overflow: auto;
  }
  #qlist li {
    border-bottom: 1px solid var(--line);
    padding: 0.75rem 1rem; cursor: pointer;
  }
  #qlist li:hover { background: #eef5fa; }
  #qlist li.active { background: #e2eef6; }
  .qid {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; color: var(--muted);
  }
  .rate {
    font-family: "IBM Plex Mono", monospace;
    font-weight: 500; font-size: 0.95rem;
  }
  .mini-rates {
    display: flex; flex-wrap: wrap; gap: 0.35rem;
    margin-top: 0.35rem;
  }
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.missing { background: #f3e6cf; color: #5a3a12; }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .answer-box, .choice-box, .model-block {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .model-block h3 {
    margin: 0 0 0.5rem; font-size: 1rem;
    display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap;
  }
  .shot {
    border-top: 1px dashed var(--line);
    padding: 0.65rem 0;
  }
  .shot:first-of-type { border-top: none; }
  .shot-head {
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
    margin-bottom: 0.35rem;
  }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  .audio-source {
    margin: 0.3rem 0 0;
    font-size: 0.75rem;
    color: var(--muted);
    word-break: break-all;
  }
  .audio-source a { color: var(--accent); }
  .mode-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid var(--line);
    vertical-align: middle;
    color: #5a3a12; background: #f3e6cf; border-color: #d4b88a;
  }
  .brand-row {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
  }
  .run-meta {
    margin-top: 0.35rem; font-size: 0.82rem; color: var(--muted);
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
  }
  details.accordion {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.75rem 0; overflow: hidden;
  }
  details.accordion > summary {
    cursor: pointer; list-style: none;
    padding: 0.75rem 0.9rem;
    font-weight: 600; font-size: 0.92rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; user-select: none;
  }
  details.accordion > summary::-webkit-details-marker { display: none; }
  details.accordion > summary::after {
    content: "+";
    font-family: "IBM Plex Mono", monospace;
    color: var(--muted); font-weight: 500;
  }
  details.accordion[open] > summary {
    border-bottom: 1px solid var(--line);
    background: #f2f6f9;
  }
  details.accordion[open] > summary::after { content: "−"; }
  .accordion-body { padding: 0.75rem 0.9rem; }
  details.prompt-model {
    border: 1px solid var(--line); border-radius: 8px;
    margin: 0.5rem 0; background: #fbfcfd;
  }
  details.prompt-model > summary {
    cursor: pointer; padding: 0.55rem 0.7rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; font-weight: 500;
    list-style: none; display: flex; justify-content: space-between;
  }
  details.prompt-model > summary::-webkit-details-marker { display: none; }
  details.prompt-model > summary::after {
    content: "▸"; color: var(--muted);
  }
  details.prompt-model[open] > summary::after { content: "▾"; }
  details.prompt-model pre {
    margin: 0; border-radius: 0 0 8px 8px;
    border-top: 1px solid var(--line);
  }
  .choice-box.hidden-from-model {
    border-style: dashed; background: #faf7f2;
  }
  .choice-box .note {
    font-size: 0.78rem; color: var(--muted); margin: 0.25rem 0 0.45rem;
  }
  .verdict-grid-wrap {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.75rem 0; overflow: auto;
  }
  .verdict-grid-wrap > .vg-title {
    padding: 0.55rem 0.8rem; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line); background: #f2f6f9;
  }
  .verdict-grid {
    display: grid;
    gap: 3px;
    padding: 0.65rem 0.8rem 0.8rem;
    align-items: center;
    min-width: max-content;
  }
  .vg-corner { font-size: 0.72rem; color: var(--muted); }
  .vg-shot {
    text-align: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.68rem; color: var(--muted);
  }
  .vg-model {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; font-weight: 500;
    padding-right: 0.5rem; white-space: nowrap;
  }
  .vg-cell {
    width: 1.05rem; height: 1.05rem; border-radius: 3px;
    justify-self: center;
    border: 1px solid transparent;
  }
  .vg-cell.pass { background: var(--good); }
  .vg-cell.fail { background: var(--bad); }
  .vg-cell.mixed {
    background: linear-gradient(135deg, var(--good) 49%, var(--bad) 51%);
  }
  .vg-cell.pending { background: #d5dde3; border-color: var(--line); }
  .vg-cell.missing { background: transparent; border: 1px dashed var(--line); }
  .judge-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.25rem 0 0.15rem; }
  .judge-chips .sample {
    font-size: 0.72rem; opacity: 0.92;
  }
  .judge-rationale { margin-top: 0.35rem; }
  .header-right {
    display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end;
  }
  #sampling-btn {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.75rem; cursor: pointer;
    white-space: nowrap;
  }
  #sampling-btn:hover, #sampling-btn.active {
    background: #e2eef6; border-color: #8fb3c9;
  }
  .modal {
    position: fixed; inset: 0; z-index: 40;
    display: none; align-items: start; justify-content: center;
    padding: 4.5rem 1rem 1.5rem;
  }
  .modal.open { display: flex; }
  .modal-backdrop {
    position: absolute; inset: 0;
    background: rgba(20, 32, 42, 0.38);
    backdrop-filter: blur(4px);
  }
  .modal-card {
    position: relative; z-index: 1;
    width: min(760px, 100%);
    max-height: calc(100vh - 6rem);
    overflow: auto;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
  }
  .modal-head {
    position: sticky; top: 0;
    display: flex; align-items: start; justify-content: space-between;
    gap: 1rem; padding: 1rem 1.15rem;
    background: color-mix(in srgb, var(--card) 92%, transparent);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
  }
  .modal-head h2 {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem; font-weight: 600; letter-spacing: -0.03em;
    margin: 0 0 0.2rem;
  }
  .modal-head p { margin: 0; color: var(--muted); font-size: 0.85rem; }
  .modal-close {
    font: inherit; cursor: pointer;
    background: transparent; border: 1px solid var(--line);
    border-radius: 8px; padding: 0.25rem 0.55rem; color: var(--muted);
  }
  .sampling-list { list-style: none; margin: 0; padding: 0; }
  .sampling-list li {
    padding: 0.9rem 1.15rem;
    border-bottom: 1px solid var(--line);
  }
  .sampling-list li:last-child { border-bottom: none; }
  .sampling-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.88rem; font-weight: 500;
  }
  .sampling-id {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; color: var(--muted); margin: 0.15rem 0 0.45rem;
  }
  .sampling-params {
    display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0 0 0.45rem;
  }
  .sampling-params .chip { background: #e2eef6; color: var(--ink); }
  .sampling-note { margin: 0.35rem 0 0; font-size: 0.8rem; color: var(--muted); }
  .sampling-source {
    font-size: 0.82rem;
    color: var(--accent);
    word-break: break-all;
  }
  .bench-row {
    display: flex; flex-direction: column; gap: 0.35rem;
    margin: 0 0 0.55rem;
  }
  .bench-item {
    display: grid; gap: 0.15rem;
  }
  .bench-head {
    display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: baseline;
  }
  .chip.score { color: var(--accent); background: #d9e8f3; }
  .chip.none { background: #e8eef2; }
  .sampling-section {
    font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--muted); margin: 0.15rem 0 0.3rem;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-row">
        <h1>MMAR Freeform</h1>
        <span class="mode-badge">Freeform</span>
      </div>
      <p>All models × all shots from the freeform pack. Graded by Claude Sonnet 5 (majority of 3, with gold, no audio).</p>
      <div class="run-meta" id="run-meta"></div>
    </div>
    <div class="header-right">
    <div class="controls">
      <label>Search
        <input id="search" type="search" placeholder="id / question text" />
      </label>
      <label>Category
        <select id="category" class="filter-select"><option value="">All</option></select>
      </label>
      <label>Subcategory
        <select id="subcategory" class="filter-select"><option value="">All</option></select>
      </label>
      <label>Modality
        <select id="modality" class="filter-select"><option value="">All</option></select>
      </label>
      <label>Sort
        <select id="sort">
          <option value="success">Success (low → high)</option>
          <option value="disagreement">Disagreement (high → low)</option>
        </select>
      </label>
      <label>&nbsp;
        <button id="filter-graded" type="button">Graded only</button>
      </label>
      <label>&nbsp;
        <button id="filter-incomplete" type="button">Incomplete only</button>
      </label>
    </div>
    <button id="sampling-btn" type="button">Model info</button>
    </div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Questions</h2>
    <div class="stats" id="stats">Loading…</div>
    <ul id="qlist"></ul>
  </section>
  <section class="panel">
    <h2>Detail</h2>
    <div id="detail"><p class="muted">Select a question.</p></div>
  </section>
</main>
<div id="sampling-modal" class="modal" aria-hidden="true">
  <div class="modal-backdrop" data-close-sampling></div>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="sampling-title">
    <div class="modal-head">
      <div>
        <h2 id="sampling-title">Model info</h2>
        <p>Reported MMAR / MMAU scores and the sampling parameters used in this pack, with source links.</p>
      </div>
      <button class="modal-close" type="button" data-close-sampling aria-label="Close">✕</button>
    </div>
    <ul class="sampling-list" id="sampling-list">
      <li class="muted">Loading…</li>
    </ul>
  </div>
</div>
<script>
const state = {
  modelLabels: [],
  questions: [],
  modalities: [],
  categories: [],
  subcategories: [],
  categorySubcategories: {},
  coverage: [],
  nShots: 5,
  nComplete: 0,
  nGraded: 0,
  selectedId: null,
  filterIncomplete: false,
  filterGraded: false,
  sampling: null,
  judges: [],
  primaryJudge: null,
};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => {
    return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c];
  });
}

function fmtParam(key, value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number" && !Number.isInteger(value)) {
    return String(value);
  }
  return String(value);
}

function fmtScore(v) {
  if (v === null || v === undefined) return "none";
  const n = Number(v);
  if (!Number.isFinite(n)) return "none";
  return String(n);
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return (100 * n).toFixed(1) + "%";
}

function fmtDisagree(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(2);
}

function questionDisagree(row) {
  const v = row && row.avg_disagreement;
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function prettyJudge(key) {
  if (!key) return "judge";
  const parts = String(key).split("__");
  if (parts.length >= 3) {
    const gold = parts[parts.length - 1] === "nongold" ? "no gold" : parts[parts.length - 1];
    const prompt = parts[parts.length - 2];
    const label = parts.slice(0, -2).join("__");
    return `${label} · ${prompt} · ${gold}`;
  }
  return String(key);
}

function judgeModelLabel(key) {
  const parts = String(key || "").split("__");
  return parts.length >= 3 ? parts.slice(0, -2).join("__") : String(key || "");
}

function shortJudge(key) {
  return shortLabel(judgeModelLabel(key));
}

function modelJudgeStats(pm) {
  const per = (pm && pm.per_judge) || {};
  const primary = state.primaryJudge;
  if (primary && per[primary] && Number.isFinite(Number(per[primary].shot_success_rate))) {
    return per[primary];
  }
  const rows = Object.values(per).filter(row =>
    row && row.shot_success_rate !== null && row.shot_success_rate !== undefined
      && Number.isFinite(Number(row.shot_success_rate))
  );
  if (!rows.length) return pm || {};
  const rate = rows.reduce((sum, row) => sum + Number(row.shot_success_rate), 0) / rows.length;
  return {
    shot_success_rate: rate,
    n_shot_correct: rows.reduce((sum, row) => sum + (Number(row.n_shot_correct) || 0), 0),
    n_shots: rows.reduce((sum, row) => sum + (Number(row.n_shots) || 0), 0),
  };
}

function questionAvg(row) {
  const rates = (state.modelLabels || []).map(label => {
    const pm = (row.per_model || {})[label] || {};
    if (!pm.present) return null;
    return modelJudgeStats(pm).shot_success_rate;
  }).filter(v => v !== null && v !== undefined && Number.isFinite(Number(v)));
  if (!rates.length) return null;
  return rates.reduce((a, b) => a + Number(b), 0) / rates.length;
}

function questionHasGrades(row) {
  if (row && row.has_grades) return true;
  return (state.modelLabels || []).some(label => {
    const per = ((row.per_model || {})[label] || {}).per_judge || {};
    return Object.values(per).some(stats =>
      stats && stats.n_shots > 0 && stats.n_shot_correct !== null && stats.n_shot_correct !== undefined
    );
  });
}

function sampleVotes(entry) {
  const samples = (entry && entry.samples) || [];
  return samples.map(s => {
    if (!s || s.correct === undefined || s.correct === null) return null;
    return !!s.correct;
  }).filter(v => v !== null);
}

function shotJudgeKeys(shot) {
  const onShot = Object.keys((shot && shot.judges) || {});
  const ordered = (state.judges || [])
    .map(j => j.label || j)
    .filter(key => onShot.includes(key));
  for (const key of onShot) {
    if (!ordered.includes(key)) ordered.push(key);
  }
  return ordered;
}

function shotConsensus(shot) {
  const keys = shotJudgeKeys(shot);
  const verdicts = keys.map(key => {
    const entry = (shot.judges || {})[key] || {};
    if (entry.correct === undefined || entry.correct === null) return null;
    return !!entry.correct;
  }).filter(v => v !== null);
  if (!verdicts.length) {
    if (shot && shot.correct !== undefined && shot.correct !== null) {
      return shot.correct ? "pass" : "fail";
    }
    return "pending";
  }
  const nPass = verdicts.filter(Boolean).length;
  if (nPass === verdicts.length) return "pass";
  if (nPass === 0) return "fail";
  return "mixed";
}

function shotDisagreement(shot) {
  const keys = shotJudgeKeys(shot);
  for (const key of keys) {
    const votes = sampleVotes((shot.judges || {})[key] || {});
    if (votes.length >= 2) {
      const nPass = votes.filter(Boolean).length;
      const pctPass = nPass / votes.length;
      return 1 - Math.max(pctPass, 1 - pctPass);
    }
  }
  const verdicts = keys.map(key => {
    const entry = (shot.judges || {})[key] || {};
    if (entry.correct === undefined || entry.correct === null) return null;
    return !!entry.correct;
  }).filter(v => v !== null);
  if (!verdicts.length) return null;
  const nPass = verdicts.filter(Boolean).length;
  const pctPass = nPass / verdicts.length;
  return 1 - Math.max(pctPass, 1 - pctPass);
}

function sourceLink(href, label) {
  if (!href) return `<span class="muted">none</span>`;
  const text = label || href;
  return `<a class="sampling-source" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(text)}</a>`;
}

function renderBench(name, entry) {
  const row = entry || {};
  if (row.score === null || row.score === undefined) {
    return `<div class="bench-item">
      <div class="bench-head"><span class="chip none">${escapeHtml(name)} none</span></div>
    </div>`;
  }
  const split = row.split ? `<span class="muted">${escapeHtml(row.split)}</span>` : "";
  return `<div class="bench-item">
    <div class="bench-head">
      <span class="chip score">${escapeHtml(name)} ${escapeHtml(fmtScore(row.score))}</span>
      ${split}
    </div>
    ${sourceLink(row.url, row.label)}
  </div>`;
}

function renderSamplingList(models) {
  const list = document.getElementById("sampling-list");
  if (!models || !models.length) {
    list.innerHTML = `<li class="muted">No model metadata for this pack.</li>`;
    return;
  }
  list.innerHTML = models.map(row => {
    const params = Object.entries(row.sampling || {}).map(([key, value]) =>
      `<span class="chip">${escapeHtml(key)}=${escapeHtml(fmtParam(key, value))}</span>`
    ).join("");
    const note = row.note ? `<p class="sampling-note">${escapeHtml(row.note)}</p>` : "";
    return `<li>
      <div class="sampling-label">${escapeHtml(row.label)}</div>
      <div class="sampling-id">${escapeHtml(row.model_id || "")}</div>
      <div class="sampling-section">Reported scores</div>
      <div class="bench-row">
        ${renderBench("MMAR", row.mmar)}
        ${renderBench("MMAU", row.mmau)}
      </div>
      <div class="sampling-section">Sampling</div>
      <div class="sampling-params">${params || `<span class="chip">—</span>`}</div>
      ${sourceLink(row.source_url, row.source_label)}
      ${note}
    </li>`;
  }).join("");
}

function setSamplingOpen(open) {
  const modal = document.getElementById("sampling-modal");
  const btn = document.getElementById("sampling-btn");
  modal.classList.toggle("open", open);
  modal.setAttribute("aria-hidden", open ? "false" : "true");
  btn.classList.toggle("active", open);
  document.body.style.overflow = open ? "hidden" : "";
}

async function openSampling() {
  setSamplingOpen(true);
  if (state.sampling) return;
  try {
    const data = await api("/api/sampling");
    state.sampling = data.models || [];
    renderSamplingList(state.sampling);
  } catch (err) {
    document.getElementById("sampling-list").innerHTML =
      `<li class="muted">Failed to load model info: ${escapeHtml(String(err))}</li>`;
  }
}

function bindSamplingModal() {
  document.getElementById("sampling-btn").addEventListener("click", () => {
    const open = document.getElementById("sampling-modal").classList.contains("open");
    if (open) setSamplingOpen(false);
    else openSampling();
  });
  document.querySelectorAll("[data-close-sampling]").forEach(el => {
    el.addEventListener("click", () => setSamplingOpen(false));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setSamplingOpen(false);
  });
}

function shortLabel(label) {
  const map = {
    "af-next-think": "af-next",
    "music-flamingo": "mf",
    "mimo-audio-7b": "mimo",
    "interactive-omni-8b": "i-omni",
    "qwen3-omni": "qwen3",
    "qwen3-omni-instruct": "qwen3-i",
    "qwen2.5-omni-7b": "qwen2.5",
    "voxtral-small-24b": "voxtral",
    "phi-4-multimodal": "phi-4",
    "gemma-4-e4b": "gemma-e4b",
    "gemma-4-12b": "gemma-12b",
    "nemotron-3-nano-omni": "nemotron",
    "gemini-3.7-flash": "gemini",
    "gpt-4o-mini": "4o-mini",
    "claude-sonnet-5": "claude",
  };
  return map[label] || label;
}

function questionSubcat(row) {
  return String((row && (row["sub-category"] || row.sub_category)) || "");
}

function fillSelect(sel, values, allLabel) {
  const prev = sel.value;
  const items = (values || []).filter(Boolean);
  sel.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>` + items.map(v =>
    `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`
  ).join("");
  sel.value = items.includes(prev) ? prev : "";
}

function fillSubcategorySelect() {
  const category = document.getElementById("category").value;
  const values = category
    ? (state.categorySubcategories[category] || [])
    : (state.subcategories || []);
  fillSelect(document.getElementById("subcategory"), values, "All");
}

function applyTaxonomyFilters() {
  renderList();
  const items = filteredQuestions();
  if (items.length && !items.some(row => row.id === state.selectedId)) {
    selectQuestion(items[0].id);
  }
}

function filteredQuestions() {
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  const category = document.getElementById("category").value;
  const subcategory = document.getElementById("subcategory").value;
  const modality = document.getElementById("modality").value;
  const items = state.questions.filter(row => {
    if (state.filterIncomplete && row.complete) return false;
    if (state.filterGraded && !questionHasGrades(row)) return false;
    if (category && row.category !== category) return false;
    if (subcategory && questionSubcat(row) !== subcategory) return false;
    if (modality && row.modality !== modality) return false;
    if (!q) return true;
    return String(row.id).toLowerCase().includes(q)
      || String(row.question || "").toLowerCase().includes(q);
  });
  const sort = (document.getElementById("sort") || {}).value || "success";
  return items.slice().sort((a, b) => {
    if (sort === "disagreement") {
      const ad = questionDisagree(a);
      const bd = questionDisagree(b);
      if (ad === null && bd === null) return String(a.id).localeCompare(String(b.id));
      if (ad === null) return 1;
      if (bd === null) return -1;
      if (ad !== bd) return bd - ad;
      return String(a.id).localeCompare(String(b.id));
    }
    const ar = questionAvg(a);
    const br = questionAvg(b);
    if (ar === null && br === null) return String(a.id).localeCompare(String(b.id));
    if (ar === null) return 1;
    if (br === null) return -1;
    if (ar !== br) return ar - br;
    return String(a.id).localeCompare(String(b.id));
  });
}

function renderStats() {
  const items = filteredQuestions();
  const parts = [
    `<span><strong>${items.length}</strong> shown</span>`,
    `<span>${state.questions.length} total</span>`,
    `<span>${state.nComplete} complete</span>`,
    `<span>${state.nGraded} graded</span>`,
    `<span>${state.nShots} shots</span>`,
  ];
  const accParts = [];
  for (const row of state.coverage) {
    if (row.n_graded) {
      accParts.push(
        `<span>${escapeHtml(shortLabel(row.model))}: <strong>${fmtRate(row.accuracy)}</strong> (${row.n_shot_correct}/${row.n_graded})</span>`
      );
    } else {
      accParts.push(
        `<span>${escapeHtml(shortLabel(row.model))}: <strong>${row.n_done}</strong>/${row.n_total}</span>`
      );
    }
  }
  if (accParts.length) {
    parts.push(`<span>accuracy</span>`);
    parts.push(...accParts);
  }
  document.getElementById("stats").innerHTML = parts.join(" · ");
}

function renderList() {
  renderStats();
  const list = document.getElementById("qlist");
  const labels = state.modelLabels;
  const items = filteredQuestions();
  list.innerHTML = items.map(row => {
    const chips = labels.map(label => {
      const pm = (row.per_model || {})[label] || {};
      if (!pm.present) {
        return `<span class="chip missing" title="${escapeHtml(label)} missing">${escapeHtml(shortLabel(label))} —</span>`;
      }
      const stats = modelJudgeStats(pm);
      const counted = (stats.n_shot_correct != null && stats.n_shots)
        ? ` ${stats.n_shot_correct}/${stats.n_shots}`
        : "";
      return `<span class="chip" title="${escapeHtml(label)}">${escapeHtml(shortLabel(label))} ${fmtRate(stats.shot_success_rate)}${counted}</span>`;
    }).join("");
    const active = row.id === state.selectedId ? "active" : "";
    const cover = `${row.n_models}/${row.n_models_total}`;
    const disagreeTitle = "Claude 3-sample disagreement: 1 − max(percent pass, percent fail). 0 = unanimous, 0.33 = 2/3 split.";
    const taxonomy = [row.modality, row.category, questionSubcat(row)].filter(Boolean).join(" · ");
    return `<li class="${active}" data-id="${escapeHtml(row.id)}">
      <div class="qid">${escapeHtml(row.id)}${taxonomy ? ` · ${escapeHtml(taxonomy)}` : ""}</div>
      <div class="rate">${fmtRate(questionAvg(row))} avg · <span title="${escapeHtml(disagreeTitle)}">${fmtDisagree(questionDisagree(row))} disagree</span> · ${cover} models</div>
      <div class="mini-rates">${chips}</div>
      <p class="qtext">${escapeHtml(row.question || "")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

function renderPromptAccordion(prompts) {
  if (!prompts) return "";
  const shared = prompts.shared || "";
  const seen = new Set();
  const modelBlocks = state.modelLabels
    .filter(label => {
      if (!prompts[label] || seen.has(label)) return false;
      seen.add(label);
      return true;
    })
    .map(label => `<details class="prompt-model">
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(prompts[label])}</pre>
    </details>`)
    .join("");
  return `<details class="accordion" open>
    <summary><span>Full prompt</span></summary>
    <div class="accordion-body">
      <p class="muted" style="margin:0 0 0.55rem">Question only — multiple-choice options were not shown to the model.</p>
      <strong style="font-size:0.82rem;color:var(--muted)">Shared question text</strong>
      <pre>${escapeHtml(shared)}</pre>
      <strong style="display:block;margin-top:0.75rem;font-size:0.82rem;color:var(--muted)">Per-model chat wrappers</strong>
      ${modelBlocks || `<p class="muted">No per-model wrappers stored.</p>`}
    </div>
  </details>`;
}

function goldAccordion(title, text) {
  if (!text) return "";
  return `<details class="accordion">
    <summary><span>${escapeHtml(title)}</span></summary>
    <div class="accordion-body"><pre>${escapeHtml(text)}</pre></div>
  </details>`;
}

function renderVerdictGrid(modelLabels, predictions, nShots) {
  const labels = modelLabels || [];
  const shotsN = Math.max(1, nShots || 5);
  if (!labels.length) return "";
  const shotHeaders = Array.from({ length: shotsN }, (_, i) =>
    `<div class="vg-shot">s${i}</div>`
  ).join("");
  const rows = labels.map(label => {
    const pred = (predictions || {})[label] || null;
    const shots = pred ? (pred.shots || []) : [];
    const cells = Array.from({ length: shotsN }, (_, i) => {
      if (!pred) {
        return `<div class="vg-cell missing" title="${escapeHtml(label)} s${i}: missing"></div>`;
      }
      const shot = shots.find(s => Number(s.shot_index) === i) || shots[i];
      if (!shot) {
        return `<div class="vg-cell missing" title="${escapeHtml(label)} s${i}: missing"></div>`;
      }
      const verdict = shotConsensus(shot);
      if (verdict === "pending") {
        return `<div class="vg-cell pending" title="${escapeHtml(label)} s${i}: pending"></div>`;
      }
      const disagree = shotDisagreement(shot);
      const extra = disagree === null ? "" : ` · disagree ${fmtDisagree(disagree)}`;
      return `<div class="vg-cell ${verdict}" title="${escapeHtml(label)} s${i}: ${verdict}${extra}"></div>`;
    }).join("");
    return `<div class="vg-model">${escapeHtml(label)}</div>${cells}`;
  }).join("");
  return `<div class="verdict-grid-wrap">
    <div class="vg-title">Verdict grid · Claude majority of 3 · model × shot</div>
    <div class="verdict-grid" style="grid-template-columns: max-content repeat(${shotsN}, 1.15rem)">
      <div class="vg-corner"></div>
      ${shotHeaders}
      ${rows}
    </div>
  </div>`;
}

function shotJudgeChips(shot) {
  const keys = shotJudgeKeys(shot);
  if (!keys.length) {
    return `<div class="judge-chips"><span class="chip">pending</span></div>`;
  }
  return `<div class="judge-chips">${keys.map(key => {
    const entry = (shot.judges || {})[key] || {};
    const pending = entry.correct === null || entry.correct === undefined;
    const klass = pending ? "chip" : (entry.correct ? "pass" : "fail");
    const text = pending ? "pending" : (entry.correct ? "pass" : "fail");
    const votes = sampleVotes(entry);
    const nPass = votes.filter(Boolean).length;
    const voteLabel = votes.length ? ` ${nPass}/${votes.length}` : "";
    const majority = `<span class="${klass}" title="${escapeHtml(prettyJudge(key))} majority of ${votes.length || 1}">${escapeHtml(shortJudge(key))}: ${text}${voteLabel}</span>`;
    const sampleChips = votes.map((v, i) => {
      const label = v ? "pass" : "fail";
      return `<span class="${v ? "pass" : "fail"} sample" title="sample ${i + 1}">s${i} ${label}</span>`;
    }).join("");
    return majority + sampleChips;
  }).join("")}</div>`;
}

function shortJudgeText(text) {
  const t = String(text || "").trim();
  return !t || /^(correct|incorrect|pass|fail|yes|no|0|1)$/i.test(t);
}

function shotJudgeAccordions(shot) {
  return shotJudgeKeys(shot).map(key => {
    const entry = (shot.judges || {})[key] || {};
    const samples = entry.samples || [];
    const blocks = samples.map((sample, i) => {
      const text = (sample && (sample.generation || sample.reasoning || sample.output)) || "";
      if (shortJudgeText(text)) return "";
      const v = sample.correct === true ? "pass" : sample.correct === false ? "fail" : "pending";
      return `<details class="accordion judge-rationale">
        <summary><span>Claude sample ${i + 1} · ${v}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(text)}</pre></div>
      </details>`;
    }).join("");
    if (blocks) return blocks;
    const text = entry.generation || entry.reasoning || entry.output || "";
    if (shortJudgeText(text)) return "";
    return `<details class="accordion judge-rationale">
      <summary><span>Judge ${escapeHtml(prettyJudge(key))}</span></summary>
      <div class="accordion-body"><pre>${escapeHtml(text)}</pre></div>
    </details>`;
  }).join("");
}

function fmtTokens(n) {
  if (n === null || n === undefined || n === "") return "";
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  return `${num.toLocaleString()} tok`;
}

function shotBlock(shot) {
  const parsed = shot.answer_prediction || "";
  const generated = shot.model_output || "";
  const tokLabel = fmtTokens(shot.n_tokens);
  const tokChip = tokLabel
    ? `<span class="chip" title="Generated tokens">${escapeHtml(tokLabel)}</span>`
    : "";
  const gen = generated
    ? `<details class="accordion"><summary><span>Generated${tokLabel ? ` · ${escapeHtml(tokLabel)}` : ""}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(generated)}</pre></div>
       </details>`
    : "";
  const disagree = shotDisagreement(shot);
  const disagreeChip = disagree === null
    ? ""
    : `<span class="chip" title="1 − max(percent pass, percent fail)">${fmtDisagree(disagree)} disagree</span>`;
  return `<div class="shot">
    <div class="shot-head">
      <span>shot ${shot.shot_index ?? "—"}</span>
      ${tokChip}
      ${disagreeChip}
    </div>
    ${shotJudgeChips(shot)}
    ${parsed ? `<pre>${escapeHtml(parsed)}</pre>` : `<p class="muted">No parsed answer.</p>`}
    ${gen}
    ${shotJudgeAccordions(shot)}
  </div>`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const data = await api(`/api/question?id=${encodeURIComponent(id)}`);
    const row = data.question || {};
    const labels = (data.model_labels && data.model_labels.length)
      ? data.model_labels
      : state.modelLabels;
    const nShots = data.n_shots || state.nShots;
    const choices = (row.choices || []).map((c, i) =>
      `<div>(${String.fromCharCode(65+i)}) ${escapeHtml(c)}</div>`
    ).join("");
    let modelsHtml = "";
    for (const label of labels) {
      const pred = (data.predictions || {})[label];
      const pm = (row.per_model || {})[label] || {};
      if (!pred) {
        modelsHtml += `<div class="model-block"><h3>${escapeHtml(label)} <span class="muted">missing</span></h3></div>`;
        continue;
      }
      const shots = (pred.shots || []).slice().sort((a, b) => (a.shot_index ?? 0) - (b.shot_index ?? 0));
      const shotsHtml = shots.map(shotBlock).join("");
      const stats = modelJudgeStats(pm);
      const counted = (stats.n_shot_correct != null && stats.n_shots)
        ? ` ${stats.n_shot_correct}/${stats.n_shots}`
        : "";
      const rateChip = `<span class="chip">${fmtRate(stats.shot_success_rate)}${counted}</span>`;
      modelsHtml += `<div class="model-block">
        <h3>${escapeHtml(label)} ${rateChip}</h3>
        ${shotsHtml || "<p class='muted'>No shots stored.</p>"}
      </div>`;
    }
    const audio = data.audio_url
      ? `<audio controls preload="none" src="${escapeHtml(data.audio_url)}"></audio>`
      : `<p class="muted">Audio not found locally.</p>`;
    const sourceUrl = row.url || "";
    const audioSource = sourceUrl
      ? `<p class="audio-source"><a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceUrl)}</a></p>`
      : "";
    const cover = `${row.n_models ?? "—"}/${row.n_models_total ?? labels.length} models`;
    detail.innerHTML = `
      <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap">
        <div class="qid">${escapeHtml(row.id || id)}</div>
        <span class="mode-badge">Freeform</span>
        ${row.complete ? "" : `<span class="chip missing">${escapeHtml(cover)}</span>`}
      </div>
      <h3 style="margin:0.4rem 0 0.2rem;font-family:Space Grotesk,sans-serif">${escapeHtml(row.question || "")}</h3>
      <p class="muted">${escapeHtml(row.modality || "")} · ${escapeHtml(row.category || "")}${questionSubcat(row) ? ` · ${escapeHtml(questionSubcat(row))}` : ""} · avg ${fmtRate(questionAvg(row))} · disagree ${fmtDisagree(questionDisagree(row))} · ${escapeHtml(cover)}</p>
      ${renderVerdictGrid(labels, data.predictions || {}, nShots)}
      ${audio}
      ${audioSource}
      ${renderPromptAccordion(data.prompts)}
      <div class="choice-box hidden-from-model">
        <strong>Choices</strong>
        <div class="note">Not shown to the model in freeform mode (gold answer reference only).</div>
        ${choices || `<span class="muted">No choices stored.</span>`}
      </div>
      <div class="answer-box"><strong>Gold</strong><div>${escapeHtml(row.answer || "")}</div></div>
      ${goldAccordion("Gold thinking", row.thinking)}
      ${goldAccordion("Rubric", row.rubric)}
      ${goldAccordion("Cue", row.cue)}
      ${modelsHtml}
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load question: ${escapeHtml(String(err))}</p>`;
  }
}

async function init() {
  const data = await api("/api/pack");
  state.modelLabels = data.model_labels || [];
  state.questions = data.questions || [];
  state.modalities = data.modalities || [];
  state.categories = data.categories || [];
  state.subcategories = data.subcategories || [];
  state.categorySubcategories = data.category_subcategories || {};
  state.coverage = data.coverage || [];
  state.nShots = data.n_shots || 5;
  state.nComplete = data.n_complete || 0;
  state.nGraded = data.n_graded || state.questions.filter(questionHasGrades).length;
  state.judges = data.judges || [];
  state.primaryJudge = data.primary_judge || (state.judges[0] && state.judges[0].label) || null;
  const metaBits = [];
  if (data.n_shots) metaBits.push(`${data.n_shots} shots`);
  metaBits.push(`${state.modelLabels.length} models`);
  metaBits.push(`${state.questions.length} questions`);
  if (data.n_complete != null) metaBits.push(`${data.n_complete} with every model`);
  if (state.nGraded) metaBits.push(`${state.nGraded} graded`);
  if (state.primaryJudge) metaBits.push(`judge ${prettyJudge(state.primaryJudge)}`);
  if (state.judges.length && !state.primaryJudge) metaBits.push(`${state.judges.length} judges`);
  document.getElementById("run-meta").textContent = metaBits.join(" · ");
  fillSelect(document.getElementById("category"), state.categories, "All");
  fillSelect(document.getElementById("modality"), state.modalities, "All");
  fillSubcategorySelect();
  document.getElementById("search").addEventListener("input", renderList);
  document.getElementById("category").addEventListener("change", () => {
    fillSubcategorySelect();
    applyTaxonomyFilters();
  });
  document.getElementById("subcategory").addEventListener("change", applyTaxonomyFilters);
  document.getElementById("modality").addEventListener("change", applyTaxonomyFilters);
  document.getElementById("sort").addEventListener("change", renderList);
  document.getElementById("filter-graded").addEventListener("click", () => {
    state.filterGraded = !state.filterGraded;
    document.getElementById("filter-graded").classList.toggle("active", state.filterGraded);
    renderList();
    const items = filteredQuestions();
    if (items.length && !items.some(row => row.id === state.selectedId)) {
      selectQuestion(items[0].id);
    }
  });
  document.getElementById("filter-incomplete").addEventListener("click", () => {
    state.filterIncomplete = !state.filterIncomplete;
    document.getElementById("filter-incomplete").classList.toggle("active", state.filterIncomplete);
    renderList();
    const items = filteredQuestions();
    if (items.length && !items.some(row => row.id === state.selectedId)) {
      selectQuestion(items[0].id);
    }
  });
  bindSamplingModal();
  if (!state.questions.length) {
    document.getElementById("stats").textContent = "No questions in this pack.";
    document.getElementById("detail").innerHTML = `<p class="muted">Pack is empty or missing predictions.</p>`;
    return;
  }
  const preferred = new URLSearchParams(location.search).get("id");
  const start = (preferred && state.questions.find(q => q.id === preferred))
    ? preferred
    : state.questions[0].id;
  renderList();
  await selectQuestion(start);
}

init().catch(err => {
  document.getElementById("stats").textContent = String(err);
});
</script>
</body>
</html>
"""


def _current_pack() -> dict[str, Any]:
    return load_pack(
        str(CONFIG.get("pack_dir") or ""),
        str(CONFIG.get("judge_dir") or ""),
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_mmar] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/pack":
                bundle = _current_pack()
                self._send_json(
                    {
                        "questions": bundle["questions"],
                        "manifest": bundle["manifest"],
                        "model_labels": bundle["model_labels"],
                        "modalities": bundle["modalities"],
                        "categories": bundle.get("categories") or [],
                        "subcategories": bundle.get("subcategories") or [],
                        "category_subcategories": bundle.get("category_subcategories")
                        or {},
                        "coverage": bundle["coverage"],
                        "n_shots": bundle["n_shots"],
                        "n_complete": bundle["n_complete"],
                        "n_graded": bundle.get("n_graded") or 0,
                        "n_questions": bundle["n_questions"],
                        "judges": bundle.get("judges") or [],
                        "primary_judge": bundle.get("primary_judge"),
                    }
                )
                return

            if path == "/api/sampling":
                bundle = _current_pack()
                self._send_json({"models": sampling_entries(bundle["model_labels"])})
                return

            if path == "/api/question":
                qid = (qs.get("id") or [""])[0]
                if not qid:
                    self._send_json({"error": "missing id"}, 400)
                    return
                bundle = _current_pack()
                row = bundle["by_id"].get(qid)
                if row is None:
                    self._send_json({"error": "question not found"}, 404)
                    return
                model_labels = bundle["model_labels"]
                preds = {
                    label: bundle["predictions"].get(label, {}).get(qid)
                    for label in model_labels
                }
                audio = resolve_audio(row.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                sample = next(
                    (preds[label] for label in model_labels if preds.get(label)),
                    row,
                )
                self._send_json(
                    {
                        "question": row,
                        "predictions": preds,
                        "audio_url": audio_url,
                        "model_labels": model_labels,
                        "n_shots": bundle["n_shots"],
                        "prompts": build_model_prompts(sample),
                        "judges": bundle.get("judges") or [],
                        "primary_judge": bundle.get("primary_judge"),
                    }
                )
                return

            if path.startswith("/audio/"):
                name = unquote(path[len("/audio/") :])
                audio = Path(CONFIG["audio_dir"]) / name
                if not audio.is_file():
                    self.send_error(404, "audio not found")
                    return
                data = audio.read_bytes()
                ctype = mimetypes.guess_type(str(audio))[0] or "audio/wav"
                self._send(200, data, ctype)
                return

            self.send_error(404, "not found")
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help="Freeform-thinking pack (default: outputs/mmar-freeform-thinking)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument(
        "--skip-audio-download",
        action="store_true",
        help="Do not download MMAR wavs if the local audio cache is incomplete",
    )
    parser.add_argument(
        "--force-audio-download",
        action="store_true",
        help="Re-download the MMAR wav archive even if wavs are present",
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=DEFAULT_JUDGE_DIR,
        help=(
            "Claude majority-of-3 verdicts. Default is the "
            "claude-sonnet-5__neutral_with_gt_no_audio__gold batch dir; "
            "applied grades are read from the parent llm-judge-gt pack."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    CONFIG["pack_dir"] = pack_dir
    judge_dir = args.judge_dir.expanduser().resolve()
    CONFIG["judge_dir"] = judge_dir
    audio_dir = args.audio_dir.expanduser().resolve()
    if not args.skip_audio_download:
        try:
            audio_dir = ensure_mmar_audio(
                audio_dir, force=args.force_audio_download
            )
        except SystemExit as exc:
            print(f"Audio setup failed: {exc}", flush=True)
            print("Continuing without local audio; pass --skip-audio-download to silence.")
    CONFIG["audio_dir"] = audio_dir
    load_pack.cache_clear()

    print(f"Pack:  {pack_dir}")
    print(f"Audio: {audio_dir}")
    print(f"Judge: {judge_dir}")
    if not pack_dir.is_dir():
        print("Pack directory not found. Run download_results.py first.")
    else:
        bundle = load_pack(str(pack_dir), str(judge_dir))
        print(
            f"Loaded {bundle['n_questions']} questions, "
            f"{len(bundle['model_labels'])} models, "
            f"{bundle['n_shots']} shots"
        )
        if bundle.get("primary_judge"):
            n_overlay = bundle.get("n_overlay_grades") or 0
            print(
                f"Grading: {bundle['primary_judge']} "
                f"({n_overlay} shot verdicts overlaid)"
            )
        for row in bundle["coverage"]:
            status = "complete" if row["complete"] else "partial"
            acc = row.get("accuracy")
            acc_s = (
                f"  acc {100 * acc:5.1f}% ({row.get('n_shot_correct')}/{row.get('n_graded')})"
                if acc is not None
                else ""
            )
            print(
                f"  {row['model']:<24} {row['n_done']:>4}/{row['n_total']:<4} "
                f"{status}{acc_s}"
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
