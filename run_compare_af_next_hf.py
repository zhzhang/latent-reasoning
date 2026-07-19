"""Run local-vs-HF Audio Flamingo Next comparison on Modal GPU.

Uses the seeded ``latent-reasoning`` Volume model snapshot when available.

Usage:

    uv run modal run run_compare_af_next_hf.py
    uv run modal run run_compare_af_next_hf.py --dtype float32
"""

from __future__ import annotations

import json
import sys

import modal

from modal_cache import VOLUME_MOUNT, hf_secret, mmar_eval_image, volume

image = mmar_eval_image("compare_af_next_hf")
app = modal.App("compare-af-next-hf", image=image)


@app.function(
    gpu="L40S",
    timeout=60 * 60,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
)
def run_compare(
    model_id: str = "nvidia/audio-flamingo-next-think-hf",
    dtype: str = "bfloat16",
    sequential: bool = True,
    atol: float | None = None,
    cosine_atol: float = 0.999,
    synthetic_frames: int = 3000,
    seed: int = 42,
) -> dict:
    from audio_flamingo_runtime import resolve_model_dir
    import compare_af_next_hf as compare

    model_dir = resolve_model_dir(model_id, None)
    argv = [
        "compare_af_next_hf.py",
        "--model_dir",
        model_dir,
        "--dtype",
        dtype,
        "--device",
        "cuda",
        "--cosine_atol",
        str(cosine_atol),
        "--synthetic_frames",
        str(synthetic_frames),
        "--seed",
        str(seed),
    ]
    if sequential:
        argv.append("--sequential")
    if atol is not None:
        argv.extend(["--atol", str(atol)])

    sys.argv = argv
    try:
        compare.main()
        passed = True
    except SystemExit as exc:
        passed = exc.code in (0, None)

    from pathlib import Path

    report_path = Path("outputs") / "af_next_compare.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    report["pass"] = bool(passed)
    report["model_dir"] = model_dir
    return report


@app.local_entrypoint()
def main(
    model_id: str = "nvidia/audio-flamingo-next-think-hf",
    dtype: str = "bfloat16",
    sequential: bool = True,
    atol: float | None = None,
    cosine_atol: float = 0.999,
    synthetic_frames: int = 3000,
    seed: int = 42,
):
    result = run_compare.remote(
        model_id=model_id,
        dtype=dtype,
        sequential=sequential,
        atol=atol,
        cosine_atol=cosine_atol,
        synthetic_frames=synthetic_frames,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
    if not result.get("pass"):
        raise SystemExit(1)
