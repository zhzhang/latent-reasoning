# MMAR Question Difficulty Experiment

Run the full MMAR set, generate 5 freeform responses per model
(per-model sampling; see `MODEL_SPECS`), then grade and browse
questions hardest-first by mean success rate.

Prompts are question-only (no multiple-choice options). Generation is
`run_experiment.py`; grading is a separate pipeline (`run_judges.py`).
The first judge is **primary** and drives difficulty ranking.

Inference uses **offline** vLLM (`LLM.generate`) with continuous
batching — not an OpenAI-compatible server. `n_shots` means independent
temperature **samples** of the same zero-shot prompt (not few-shot ICL).
Plain vLLM forks those samples with `SamplingParams(n=...)` so they share
one prefill; HF fallbacks duplicate the prompt per shot and rely on prefix caching.

All eval workers share one vLLM 0.28.0 container (`modal_images.eval_image`).
GPU type still comes from each `MODEL_SPECS` entry.

## Models

| Label | Checkpoint | Backend |
|-------|------------|---------|
| `af-next-think` | `nvidia/audio-flamingo-next-think-hf` | vLLM 0.28 MusicFlamingo (HF fallback); `T=0.2, max_tokens=2048, rep=1.2` |
| `music-flamingo` | `nvidia/music-flamingo-think-2601-hf` | vLLM 0.28 MusicFlamingo (HF fallback); `T=0.7, top_p=0.9, max_tokens=2048` |
| `qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | vLLM 0.28 thinker-only (B200); `T=0.6, top_p=0.95, top_k=20, max_tokens=16384` |
| `voxtral-small-24b` | `mistralai/Voxtral-Small-24B-2507` | vLLM 0.28 Mistral audio (B200); `T=0.2, top_p=0.95, max_tokens=2048` |
| `phi-4-multimodal` | `microsoft/Phi-4-multimodal-instruct` | vLLM 0.28 + speech LoRA (L40S) |
| `gemma-4-e4b` | `google/gemma-4-E4B-it` | vLLM 0.28 chat (B200) |
| `gemma-4-12b` | `google/gemma-4-12B-it` | vLLM 0.28 chat (B200) |
| `nemotron-3-nano-omni` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` | vLLM 0.28 chat (B200) |

## Seed

```bash
uv run modal run seed_volume.py --datasets mmar \
  --models af-next-think,qwen3-omni,voxtral-small-24b

# Freeform judge weights (add more aliases as needed)
uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8
```

Judge aliases: `qwen2.5-3b`, `qwen3.6-35b-a3b-fp8` (→ `Qwen/Qwen3.6-35B-A3B-FP8`).

## Run

```bash
# Full freeform generation (--detach required so GPU workers survive the client exit)
uv run modal run --detach run_experiment.py

# Smoke test one model (plain vLLM: SamplingParams n=2)
uv run modal run --detach run_experiment.py \
  --models af-next-think --n-shots 2

# Fill missing models / questions / shots (skip GPU workers with no work)
uv run modal run --detach run_experiment.py --n-shots 5

# After grading via run_judges.py, rebuild difficulty.jsonl / scores.json
uv run modal run run_experiment.py --aggregate-only
```

Defaults: `--n-shots 5 --seed 42` on the full MMAR set, freeform prompts.
Writes to the root of the `mmar-freeform-thinking` Modal Volume. Sampling
(`temperature` / `top_p` / `max_tokens` / `repetition_penalty`) is
per-model in `mmar_models.MODEL_SPECS`; optional `--temperature` / `--top-p` /
`--max-new-tokens` override every model when set.

Each run reads existing generations and fills gaps. New models generate
up to `n_shots` per question. Uncovered questions are filled for each
requested model. A model with 3 shots when `n_shots=5` generates 2 more.
Workload is computed on CPU before any GPU container starts; models that
already have `n_shots` for every question are not spawned. Each model
writes `models/<label>/predictions.jsonl`. Grade those predictions with
`run_judges.py`, then `--aggregate-only` to rank questions.

### Throughput knobs

- All pending questions for a model go in **one** offline `generate()` call;
  vLLM continuous-batches internally (no app-level `--batch-size`).
- Plain vLLM (`af-next`, `qwen3-omni`, `voxtral`, Gemma, Nemotron): one
  prompt per question with `SamplingParams(n=n_shots)` so N samples share
  prefill. Seed is per question (not per shot).
- HF fallback (AF-Next / Music Flamingo if vLLM load fails): duplicate the
  prompt per shot. Prefix caching reuses identical audio/prompt prefixes
  where enabled.
- **`max_num_seqs`**: set from measured average sequence length against the
  reported `GPU KV cache size`, not from `max_model_len`. PagedAttention
  allocates on demand, so this is a concurrency *cap*, not a reservation.
  Oversizing causes preemption; undersizing (e.g. 4 when the cache holds
  ~180 seqs) leaves most of the GPU idle. Large models use 64 here.
- **`max_model_len`**: context safety cap for startup / overlong prompts.
  Keep it close to real prompt+output length — an inflated value deflates
  vLLM's reported max concurrency and tempts a too-low `max_num_seqs`.
- **`enforce_eager: False`**: enables torch.compile + CUDA graphs where the
  model supports it (`af-next`, `qwen3-omni`, `voxtral`).
  vLLM 0.28 fixed Qwen3-Omni's Dynamo meta/cuda `profile_run` crash; graphs
  are on for the thinker-only path. Keep `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (in-process EngineCore) — multiprocess `profile_run` was the other
  meta/cuda failure mode.
- **Compile cache**: `compile_cache.py` attaches the existing
  `/cache/vllm/<label>/` tree, loads the model, and runs a warmup generate
  only on a cache miss (new or rewritten artifacts). Later eval / smoke /
  judge containers load that tree (`VLLM_CACHE_ROOT`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`). CUDA graphs still
  recapture on each boot. FlashInfer TRT-LLM cubins are version-keyed (one
  tree for the whole image, not per model) and are downloaded into
  `/opt/flashinfer/cubins` at image build (`FLASHINFER_CUBIN_DIR`); they
  are not stored on the compile-cache volume.

    ```bash
    uv run modal run --detach compile_cache.py
    uv run modal run --detach compile_cache.py --models qwen3-omni
    uv run modal run --detach compile_cache.py::compile_gemma_4_e4b
    ```
- Prefill-oriented: engines set `max_num_batched_tokens` (≥8192 where possible)
  and `enable_prefix_caching`. Qwen3 Thinking is decode-heavy
  (`max_tokens=2048`; measured outputs peak ~870).
- `--max-num-seqs` / `--gpu-memory-utilization`: optional CLI escape hatches.
- Judge engines are tuned with `tune_judge.py` (see below). For the
  `qwen3.6-35b-a3b-fp8` MoE judge, `enforce_eager: False` was worth 332 →
  4,824 output tok/s on an H100; only 3B params are active per token, so
  decode is kernel-launch bound and CUDA graphs dominate everything else.
- Watch logs for `Avg generation throughput`, `Running: N reqs`, and
  `GPU KV cache usage` (`disable_log_stats: False`). Confirm loaders print
  `vLLM ready` rather than `falling back to` HF.
- Qwen3 fused-MoE: image build installs an `E=128,N=768` config under
  `NVIDIA_B200` (copied from the H200 bf16 stand-in when the wheel has no
  native B200 file).
- The shared eval image keeps `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (in-process EngineCore).

## Download + view

```bash
# Default: mmar-freeform-thinking volume root -> outputs/mmar-freeform-thinking/
uv run modal run download_results.py
uv run python view_difficulty.py          # single-run UI (:7860)
uv run python view_mode_compare.py        # MCQ ↔ freeform compare (:7861)

# Judge pack vs human labels in exports/
uv run modal run download_judges.py
uv run python view_judges.py              # judge outcomes UI (:7862)
```

Open http://127.0.0.1:7860 — questions are ordered hardest-first
(ascending average `shot_success_rate` across models). Each example detail
opens with a **verdict grid** (test model × judge × shot, green/red cells).
If a run is missing `difficulty.jsonl` / `scores.json`, the viewer
aggregates them on startup.

Open http://127.0.0.1:7861 for the mode-compare viewer: it pairs an MC run
with a freeform run on the same question ids, sorts by
Δ = MCQ avg − freeform avg, and shows both modes’ per-model shots when a
question is selected.

Open http://127.0.0.1:7862 for judge outcomes vs the human labels in
`exports/`.

## Judge labeled MMAR generations

`run_judges.py` grades labeled questions from `exports/labels.csv` and
`exports/generations.csv`, joining question text, gold answers, and audio
from MMAR-meta. Verdicts are written to `outputs/mmar-judging` (Modal
volume `mmar-judging`). With no `--judge-model-id`, every suite model
grades every pack model's shots, including its own. Pass ids to run only
those judges. Shots that already have a verdict for the same judge key are
skipped. Pass `--force` to replace existing verdicts.

vLLM suite / dedicated judges run on Modal (this script starts a detached
App). API judges (`gemini-3.7-flash`; aliases
`gemini-3.7-mini`, `gemini`, or `api`) run locally against the
same pack and do not start Modal. Empty `--judge-model-id` stays
suite-only so a bare run does not spend API quota.
Default runs both `with_gt` (text, sees gold) and `free` (audio, no gold).
`--grade-prompt` selects any key in `JUDGE_FORMATS` (comma-separated, or
`all`). Audio is attached when that format sets `audio_included`.

```bash
# Seed judge weights first if needed
uv run modal run seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8

# All suite judges
uv run run_judges.py

# Only these suite judges
uv run run_judges.py \
  --judge-model-id qwen3-omni-instruct,phi-4-multimodal

# Dedicated text judge
uv run run_judges.py \
  --judge-model-id qwen3.6-35b-a3b-fp8

# Promote the first selected judge to primary
uv run run_judges.py \
  --judge-model-id Qwen/Qwen3.6-35B-A3B-FP8 \
  --make-primary

# Replace existing verdicts for this judge
uv run run_judges.py \
  --judge-model-id qwen3.6-35b-a3b-fp8 \
  --force

# API judges (local; needs GEMINI_API_KEY)
uv run run_judges.py \
  --judge-model-id gemini-3.7-flash
uv run run_judges.py --judge-model-id api --no-include-gold

# Named recipe from JUDGE_FORMATS (or comma-separated / all)
uv run run_judges.py --grade-prompt neutral_with_gt_no_audio
uv run run_judges.py --grade-prompt all

# Mixed: API locally while Modal vLLM runs detached
uv run run_judges.py \
  --judge-model-id gemini-3.7-flash,qwen3-omni-instruct

# Recompute Alt-Test scores from existing local verdicts
uv run run_judges.py --accuracy-only

# Download the judging pack and inspect vs human labels
uv run modal run download_judges.py
uv run python view_judges.py
```

API path knobs: `--qps` (default 4), `--max-workers` (8), `--timeout`
(180s), `--retries` (20).

`view_judges.py` (http://127.0.0.1:7862) joins judge verdicts from
`outputs/mmar-judging` to `exports/labels.csv` and
`exports/generations.csv`. Judges are scored with the
[Alt-Test](https://arxiv.org/abs/2501.10970) on shots that have at least
three human ratings. The headline number is Average Advantage Probability
ρ (probability the judge is as good as or better than a randomly chosen
annotator), one value per composite key
`{label}__{JUDGE_FORMATS key}__{gold|nongold}`, one table per recipe.
Winning rate ω
uses `--epsilon` (default 0.15) and is secondary.
Per-shot chips use majority vote of those ratings.

## Tune a judge engine

`tune_judge.py` replays real grade prompts from a past freeform run against
one judge under several vLLM engine configs and concurrency levels, reporting
output tok/s, parse rate, and verdict agreement. It writes nothing:

```bash
# Full sweep — one H100 container per engine variant, run in parallel
uv run modal run tune_judge.py::main

# Fast speed check on a single variant
uv run modal run tune_judge.py::main --variants graphs --n-cases 64

# Concurrency sweep for one engine config
uv run modal run tune_judge.py::main --variants graphs --batch-sizes 128,256,512

# Confirm the committed JUDGE_SPECS entry through the real grader path
uv run modal run tune_judge.py::verify
```

Measured for `qwen3.6-35b-a3b-fp8` on one H100 over 512 replayed shots
(mean ~600 output tokens/shot):

| config | batch | output tok/s | sec / 1k shots |
| --- | --- | --- | --- |
| eager (pre-tuning) | 64 | 332 | 1,730 |
| eager (pre-tuning) | 128 | 648 | 902 |
| CUDA graphs | 128 | 3,542 | 171 |
| CUDA graphs | 256 | 4,824 | 126 |
| CUDA graphs | 512 | 7,044 | 84 |

`async_scheduling` and raising `max_num_seqs` / `max_num_batched_tokens` were
within run-to-run noise once CUDA graphs were on, so the committed spec leaves
them at their defaults. `tune_judge.py::verify` measures the committed spec
end-to-end through `grade_shot_batch` at 5,908 output tok/s (100 s per 1k
shots, ~9 min engine init), so grading all 10k shots of a 5-model run takes
roughly 17 minutes of H100 time.

The `agree` column compares against stored verdicts from another judge on the
same shots. Check `stored_pass_rate` before trusting it — a stored judge that
passed 0 of 1000 shots (e.g. one run with too small a token budget to emit a
verdict) makes agreement against it only restate the new judge's fail rate.

## Retrofit existing runs

Older freeform outputs store a single flat `grader` / `grader_output` per
shot. Migrate them locally (no GPU) into the multi-judge schema:

```bash
uv run python retrofit_judges.py
uv run python retrofit_judges.py --run 20260807T145000Z --dry-run
uv run python retrofit_judges.py --set-primary qwen2.5-3b-instruct --backup
```

This rewrites `models/*/predictions.jsonl`, stamps `judges` /
`primary_judge` on `manifest.json`, and regenerates `difficulty.jsonl` /
`scores.json`. MC runs get a synthetic `string-match` judge so the viewer
grid is uniform. Idempotent — safe to re-run.

## Output layout

```
exports/
  labels.csv       # question_id, generation_id, model_label, shot_index, ratings
  generations.csv  # question_id, generation_id, model_label, shot_index, answer_prediction

outputs/mmar-freeform-thinking/
  question_ids.json
  manifest.json
  models/<label>/predictions.jsonl
  difficulty.jsonl
  scores.json

outputs/mmar-judging/
  labels.csv
  question_ids.json
  manifest.json
  judge_accuracy.json  # Alt-Test ρ / ω per judge×format
  models/<label>/predictions.jsonl
  models/<label>/judge_partials/<judge_key>.jsonl
```

Freeform shot records store per-judge verdicts under `shots[].judges`:

```json
{
  "shot_index": 0,
  "answer_prediction": "…",
  "correct": false,
  "grader": "Qwen/Qwen2.5-3B-Instruct",
  "grader_output": "0",
  "judges": {
    "qwen2.5-3b-instruct": {
      "correct": false,
      "verdict": "fail",
      "output": "0",
      "generation": "…full judge generation ending in <answer>0</answer>…",
      "model_id": "Qwen/Qwen2.5-3B-Instruct"
    }
  }
}
```

`generation` is the full judge reply (up to 4096 tokens); `output` /
`verdict` are the parsed final `1`/`0` (`pass`/`fail`). Re-running the same judge
(via `run_judges.py --force` or `--force-grade`) replaces prior entries for
that label.

Record / manifest fields: `judges` (ordered labels, `[0]` = primary),
`primary_judge`, `per_judge`, plus legacy `grader` / `scoring:
qwen_freeform_judge` mirrors of the primary.
