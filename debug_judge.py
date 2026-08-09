"""Limited-scope debug for the freeform LLM judge (Qwen3.6 thinking + budget).

Reproduces a handful of grade prompts from a past freeform run and compares
raw generations under several chat-template / max_tokens settings. Does not
write predictions or scores.

Usage::

    uv run modal run debug_judge.py
    uv run modal run debug_judge.py --model-id qwen3.6-35b-a3b-fp8 --max-cases 4
    uv run modal run debug_judge.py --run-id 20260807T145000Z --model-label qwen3-omni
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_cache import (
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"
DEFAULT_RUN_ID = "20260807T145000Z"
DEFAULT_MODEL_LABEL = "qwen3-omni"
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"

app = modal.App("debug-judge")


def _cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


grader_image = (
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.26.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_python_source(
        "modal_cache",
        "mmar_common",
        "audio_flamingo_runtime",
        "grader",
    )
)


def _format_chat(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool | None,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return f"User: {user_text}\nAssistant:"
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older templates may not accept enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _prompt_tail(prompt: str, n: int = 240) -> str:
    text = prompt.replace("\n", "\\n")
    if len(text) <= n:
        return text
    return "…" + text[-n:]


def _load_cases(
    *,
    run_dir: Path,
    model_label: str,
    max_cases: int,
) -> list[dict]:
    pred_path = run_dir / "models" / model_label / "predictions.jsonl"
    if not pred_path.is_file():
        raise SystemExit(f"predictions.jsonl not found: {pred_path}")

    cases: list[dict] = []
    with open(pred_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            shots = record.get("shots") or []
            if not shots:
                continue
            shot = shots[0]
            prediction = (
                shot.get("answer_prediction")
                or shot.get("model_output")
                or ""
            )
            if not str(prediction).strip():
                continue
            judge_entry = (shot.get("judges") or {}).get("qwen3.6-35b-a3b-fp8") or {}
            cases.append(
                {
                    "id": record.get("id"),
                    "question": record.get("question") or "",
                    "answer": record.get("answer") or "",
                    "prediction": prediction,
                    "stored_output": judge_entry.get("output"),
                    "stored_correct": judge_entry.get("correct"),
                }
            )
            if len(cases) >= max_cases:
                break
    if not cases:
        raise SystemExit(f"No usable shots under {pred_path}")
    return cases


@app.function(
    image=grader_image,
    gpu="H100",
    timeout=60 * 45,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def probe_judge(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    model_label: str = DEFAULT_MODEL_LABEL,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    max_cases: int = 4,
) -> dict:
    from grader import (
        build_grade_prompt,
        load_grader,
        parse_grade_verdict,
        resolve_judge_model_id,
    )

    volume.reload()
    results_volume.reload()

    model_id = resolve_judge_model_id(model_id)
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    cases = _load_cases(run_dir=run_dir, model_label=model_label, max_cases=max_cases)
    print(f"Loaded {len(cases)} cases from {run_dir / 'models' / model_label}")

    handle = load_grader(model_id)
    tokenizer = handle["tokenizer"]
    SamplingParams = handle["SamplingParams"]

    # Variants: current grader defaults vs disable-thinking vs more budget.
    variants: list[dict[str, Any]] = [
        {
            "name": "baseline_grader",
            "enable_thinking": None,  # whatever the template defaults to
            "max_tokens": 8,
            "note": "matches grader._format_chat + grade_shot_batch defaults",
        },
        {
            "name": "thinking_on_max8",
            "enable_thinking": True,
            "max_tokens": 8,
            "note": "explicit thinking; same tiny budget as grader",
        },
        {
            "name": "thinking_off_max8",
            "enable_thinking": False,
            "max_tokens": 8,
            "note": "disable thinking; keep grader budget",
        },
        {
            "name": "thinking_on_max256",
            "enable_thinking": True,
            "max_tokens": 256,
            "note": "allow room for CoT then YES/NO",
        },
        {
            "name": "thinking_off_max32",
            "enable_thinking": False,
            "max_tokens": 32,
            "note": "no thinking + small answer budget",
        },
    ]

    # Inspect chat template tails once on the first case.
    first_prompt = build_grade_prompt(
        question=str(cases[0]["question"]),
        answer=str(cases[0]["answer"]),
        prediction=str(cases[0]["prediction"]),
    )
    template_probe: dict[str, str] = {}
    for flag in (None, True, False):
        key = {None: "default", True: "thinking_on", False: "thinking_off"}[flag]
        rendered = _format_chat(tokenizer, first_prompt, enable_thinking=flag)
        template_probe[key] = _prompt_tail(rendered, 320)
        print(f"\n=== chat template tail ({key}) ===")
        print(template_probe[key])

    results: list[dict] = []
    for case in cases:
        user_text = build_grade_prompt(
            question=str(case["question"]),
            answer=str(case["answer"]),
            prediction=str(case["prediction"]),
        )
        case_row: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "answer": case["answer"],
            "prediction": case["prediction"],
            "stored_output": case.get("stored_output"),
            "stored_correct": case.get("stored_correct"),
            "variants": {},
        }
        print("\n" + "=" * 72)
        print(f"CASE {case['id']}")
        print(f"Q: {case['question']}")
        print(f"Gold: {case['answer']}")
        print(f"Pred: {case['prediction']}")
        print(
            f"Stored judge: correct={case.get('stored_correct')} "
            f"output={case.get('stored_output')!r}"
        )

        for variant in variants:
            prompt = _format_chat(
                tokenizer,
                user_text,
                enable_thinking=variant["enable_thinking"],
            )
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(variant["max_tokens"]),
                seed=0,
            )
            outs = handle["llm"].generate([prompt], sampling_params=sampling)
            text = ""
            finish = None
            n_tokens = None
            if outs and outs[0].outputs:
                completion = outs[0].outputs[0]
                text = str(getattr(completion, "text", "") or "")
                finish = getattr(completion, "finish_reason", None)
                token_ids = getattr(completion, "token_ids", None)
                n_tokens = len(token_ids) if token_ids is not None else None
            verdict = parse_grade_verdict(text)
            correct = bool(verdict) if verdict is not None else False
            entry = {
                "note": variant["note"],
                "enable_thinking": variant["enable_thinking"],
                "max_tokens": variant["max_tokens"],
                "raw": text,
                "n_tokens": n_tokens,
                "finish_reason": finish,
                "parsed_verdict": verdict,
                "correct_default_false": correct,
            }
            case_row["variants"][variant["name"]] = entry
            print(
                f"\n-- {variant['name']} "
                f"(thinking={variant['enable_thinking']}, "
                f"max_tokens={variant['max_tokens']}) --"
            )
            print(f"raw ({n_tokens} toks, finish={finish}): {text!r}")
            print(f"parsed={verdict} -> correct={correct}")

        results.append(case_row)

    summary = {
        "model_id": model_id,
        "run_id": run_id,
        "model_label": model_label,
        "n_cases": len(results),
        "template_probe_tails": template_probe,
        "cases": results,
    }
    print("\n" + "=" * 72)
    print("SUMMARY JSON")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:12000])
    return summary


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    model_label: str = DEFAULT_MODEL_LABEL,
    max_cases: int = 4,
):
    summary = probe_judge.remote(
        model_id=model_id,
        run_id=run_id,
        model_label=model_label,
        max_cases=max_cases,
    )
    # Compact local echo of generations for the caller.
    print("\n=== local echo: generations ===")
    for case in summary.get("cases") or []:
        print(f"\n[{case['id']}] gold={case['answer']!r} pred={case['prediction']!r}")
        for name, entry in (case.get("variants") or {}).items():
            print(
                f"  {name}: parsed={entry.get('parsed_verdict')} "
                f"raw={entry.get('raw')!r}"
            )
