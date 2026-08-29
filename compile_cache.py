"""Warm torch.compile / Triton caches for each MMAR eval model.

Each model gets its own Modal GPU function so you can run them independently::

    uv run modal run --detach compile_cache.py::compile_af_next_think
    uv run modal run --detach compile_cache.py::compile_qwen3_omni

Or spawn one or more from the local entrypoint::

    uv run modal run --detach compile_cache.py
    uv run modal run --detach compile_cache.py --models gemma-4-e4b,qwen3-omni

The worker points inductor / Triton / vLLM at the existing
``/cache/vllm/<label>/`` tree, then loads the model. A warmup generate
runs only when that load is a cache miss (new or rewritten artifacts).
A hit skips generate. Warmup sampling keeps ``thinking_token_budget``
(``reasoning_budget`` + ``grace_period``) and only caps ``max_tokens``.
Later GPU containers call ``configure_compile_cache`` from ``load_model``
and reuse the tree.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    VOLUME_MOUNT,
    commit_compile_cache,
    compile_cache_stats,
    configure_compile_cache,
    hf_secret,
    volume,
)
from mmar_common import load_jsonl, resolve_path
from mmar_models import (
    ALL_MODEL_LABELS,
    MODEL_SPECS,
    chat_kwargs_for,
    engine_kwargs_for,
    generate_one,
    load_model,
    parse_model_list,
    resolve_sampling,
)
from modal_images import eval_image

DEFAULT_QUESTION_ID = "GJ6r_T6ckc4_00-00-00_00-00-06"
# Long enough to compile decode kernels; short enough not to wait on CoT.
WARMUP_MAX_TOKENS = 16


def _cache_missed(before: dict[str, object], after: dict[str, object]) -> bool:
    """True when load/generate wrote or replaced files in the cache tree."""
    return (
        int(after["n_files"]) > int(before["n_files"])
        or int(after["n_bytes"]) > int(before["n_bytes"])
        or float(after.get("newest_mtime") or 0)
        > float(before.get("newest_mtime") or 0)
    )

app = modal.App("compile-cache")

_COMPILE_KW = dict(
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    memory=65536,
    single_use_containers=True,
)


def _pick_warmup_item(question_id: str | None = None) -> dict[str, Any] | None:
    meta_path = Path(DEFAULT_MMAR_META)
    data_root = Path(DEFAULT_MMAR_DATA_ROOT)
    if not meta_path.is_file():
        print(f"[compile] MMAR meta missing at {meta_path}; load-only warmup")
        return None
    items = load_jsonl(meta_path)
    wanted = str(question_id or DEFAULT_QUESTION_ID)
    match = next((item for item in items if str(item.get("id")) == wanted), None)
    ordered = ([match] if match else []) + [
        item for item in items if item is not match
    ]
    for item in ordered:
        audio_path = resolve_path(data_root, item.get("audio_path") or "")
        if audio_path and Path(audio_path).is_file():
            if str(item.get("id")) != wanted:
                print(f"[compile] {wanted} missing; using {item.get('id')}")
            return {**item, "audio_path": str(audio_path)}
    print(f"[compile] no MMAR wav under {data_root / 'audio'}; load-only warmup")
    return None


def _compile_one(
    *,
    model_label: str,
    question_id: str | None = None,
) -> dict[str, Any]:
    volume.reload()
    spec = MODEL_SPECS[model_label]
    args = SimpleNamespace(
        model_id=spec["model_id"],
        tokenizer_id=spec.get("tokenizer_id"),
        local_model_dir=None,
        local_tokenizer_dir=None,
        temperature=None,
        top_p=None,
        max_new_tokens=WARMUP_MAX_TOKENS,
        greedy_non_thinking=False,
        seed=0,
        max_num_seqs=None,
        gpu_memory_utilization=None,
        prompt_mode="freeform",
    )
    engine = engine_kwargs_for(model_label, args)
    sampling = resolve_sampling(model_label, args)
    chat_kwargs = chat_kwargs_for(model_label)
    args.temperature = float(sampling["temperature"])
    args.top_p = float(sampling.get("top_p", 1.0))
    args.max_new_tokens = int(sampling["max_tokens"])
    args.repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
    args.sampling = sampling

    print(
        f"[compile {model_label}] backend={spec.get('backend')} "
        f"gpu={spec.get('gpu')} engine={engine} "
        f"chat_kwargs={chat_kwargs} sampling={sampling} "
        f"warmup_max_tokens={WARMUP_MAX_TOKENS}"
    )
    configure_compile_cache(model_label)
    before = compile_cache_stats(model_label)
    print(
        f"[compile {model_label}] existing cache files={before['n_files']} "
        f"size={before['n_mib']} MiB"
    )
    started = time.time()
    handle = load_model(model_label, args)
    load_secs = time.time() - started
    after_load = compile_cache_stats(model_label)
    cache_miss = _cache_missed(before, after_load)
    print(
        f"[compile {model_label}] loaded backend={handle.get('backend')} "
        f"in {load_secs:.1f}s cache={'miss' if cache_miss else 'hit'} "
        f"files={after_load['n_files']} size={after_load['n_mib']} MiB"
    )

    item = _pick_warmup_item(question_id) if cache_miss else None
    generate_secs = 0.0
    generate_error: str | None = None
    output_preview = ""
    if not cache_miss:
        print(f"[compile {model_label}] cache hit; skipping warmup generate")
    elif item is None:
        commit_compile_cache(model_label)
        print(f"[compile {model_label}] cache miss; no wav, load-only")
    else:
        commit_compile_cache(model_label)
        gen_started = time.time()
        try:
            parsed = generate_one(model_label, handle, item, args)
            generate_secs = time.time() - gen_started
            output_preview = str(parsed.get("answer_prediction") or "")[:120]
            print(
                f"[compile {model_label}] warmup generate "
                f"id={item.get('id')} in {generate_secs:.1f}s "
                f"preview={output_preview!r}"
            )
        except Exception as exc:  # noqa: BLE001 — cache after load still counts
            generate_secs = time.time() - gen_started
            generate_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[compile {model_label}] warmup generate failed "
                f"after {generate_secs:.1f}s: {generate_error}"
            )
        commit_compile_cache(model_label)

    stats = compile_cache_stats(model_label)
    print(
        f"[compile {model_label}] cache {stats['path']} "
        f"files={stats['n_files']} size={stats['n_mib']} MiB "
        f"load={load_secs:.1f}s generate={generate_secs:.1f}s"
    )
    if generate_error is not None:
        status = "load_ok_generate_failed"
    elif not cache_miss:
        status = "cache_hit"
    else:
        status = "ok"
    return {
        "status": status,
        "model_label": model_label,
        "backend": handle.get("backend", spec.get("backend")),
        "engine": engine,
        "chat_kwargs": chat_kwargs,
        "sampling": sampling,
        "cache": stats,
        "cache_root": stats["path"],
        "cache_miss": cache_miss,
        "load_secs": round(load_secs, 1),
        "generate_secs": round(generate_secs, 1),
        "question_id": None if item is None else item.get("id"),
        "generate_error": generate_error,
    }


def _compile_function(label: str, image: modal.Image, gpu: str):
    def run(question_id: str | None = None) -> dict:
        return _compile_one(model_label=label, question_id=question_id)

    # Modal rejects nested @app.function unless serialized=True (cloudpickle
    # from this process into the CUDA image). Give the worker a global
    # __qualname__ and bind it on the module so FILE load works. Dots must
    # go too: a dotted __qualname__ looks like a class method.
    name = f"compile_{label.replace('-', '_').replace('.', '_')}"
    run.__name__ = name
    run.__qualname__ = name
    fn = app.function(
        image=image, gpu=gpu, name=f"compile-{label}", **_COMPILE_KW
    )(run)
    globals()[name] = fn
    return fn


_COMPILE_FNS: dict[str, Any] = {}
_missing_compile: list[str] = []
for _label in ALL_MODEL_LABELS:
    _gpu = MODEL_SPECS[_label].get("gpu")
    if not _gpu:
        _missing_compile.append(_label)
        continue
    _COMPILE_FNS[_label] = _compile_function(_label, eval_image, str(_gpu))
if _missing_compile:
    raise RuntimeError(f"No compile-cache worker for models: {_missing_compile}")


def _spawn_compile(label: str, question_id: str | None = None):
    fn = _COMPILE_FNS.get(label)
    if fn is None:
        raise SystemExit(f"No compile-cache worker for model {label!r}")
    call = fn.spawn(question_id=question_id)
    print(f"Spawned compile-{label} call_id={call.object_id}")
    return call


@app.local_entrypoint()
def main(
    models: str = "all",
    question_id: str = DEFAULT_QUESTION_ID,
):
    """Warm torch.compile / Triton caches for one or more eval models.

    Each worker attaches the existing ``/cache/vllm/<label>/`` tree, loads
    the model, and runs a warmup generate only on a cache miss.

    Args:
        models: Comma-separated labels or ``all``.
        question_id: MMAR clip used for the short decode warmup.
    """
    try:
        labels = parse_model_list(models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[compile] models={labels} question_id={question_id}")
    print(
        "Use ``modal run --detach``; without it this process exiting "
        "stops the ephemeral app and kills those workers."
    )
    results: list[dict[str, Any]] = []
    for label in labels:
        call = _spawn_compile(label, question_id=question_id)
        results.append(
            {
                "status": "spawned",
                "model_label": label,
                "call_id": call.object_id,
            }
        )
    print("Orchestrator:", {"models": results})
