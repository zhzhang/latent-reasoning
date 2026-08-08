# MMAR Question Difficulty Experiment

Sample a fixed 200 MMAR questions, generate 10 responses per model
(per-model sampling; see `MODEL_SPECS`), then browse questions hardest-first
by mean success rate.

Two prompt / scoring modes:

| Mode | Prompt | Scoring |
|------|--------|---------|
| `mc` (default) | question + 4 choices | string-match against gold choice (stored as a synthetic `string-match` judge) |
| `freeform` | question only (no choices) | one or more local vLLM judges grade each shot vs gold; first judge is **primary** and drives difficulty ranking |

Default freeform judge: `Qwen/Qwen2.5-3B-Instruct`. Pass multiple with `--judge-model-ids` (comma-separated; first = primary).

Inference uses **offline** vLLM (`LLM.generate` / Omni) with continuous
batching — not an OpenAI-compatible server. `n_shots` means independent
temperature **samples** of the same zero-shot prompt (not few-shot ICL).
Plain vLLM forks those samples with `SamplingParams(n=...)` so they share
one prefill; Omni/HF duplicate the prompt per shot and rely on prefix caching.

## Models

| Label | Checkpoint | Backend |
|-------|------------|---------|
| `af-next-think` | `nvidia/audio-flamingo-next-think-hf` | vLLM 0.24 MusicFlamingo (HF fallback); `T=0.2, max_tokens=2048, rep=1.2` |
| `mimo-audio-7b` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ tokenizer) | vLLM-Omni; `T=0.3, top_p=0.95, max_tokens=512, rep=1.1` |
| `interactive-omni-8b` | `sensenova/InteractiveOmni-8B` | HF `.chat` (vLLM transformers backend incompatible); `T=1.0, max_tokens=1024` |
| `qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | vLLM 0.26 thinker-only (A100-80GB); `T=0.6, top_p=0.95, top_k=20, max_tokens=2048` |
| `voxtral-small-24b` | `mistralai/Voxtral-Small-24B-2507` | vLLM 0.26 Mistral audio (A100-80GB); `T=0.2, top_p=0.95, max_tokens=512` |

`step-audio-2-mini` is temporarily excluded from difficulty aggregation.

## Seed

```bash
uv run modal run seed_volume.py --datasets mmar \
  --models af-next-think,mimo-audio-7b,interactive-omni-8b,qwen3-omni,voxtral-small-24b

# Freeform judge weights (add more aliases as needed)
uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-27b-fp8
```

`mimo-audio-7b` also seeds `XiaomiMiMo/MiMo-Audio-Tokenizer` automatically.
Judge aliases: `qwen2.5-3b`, `qwen3.6-27b-fp8` (→ `Qwen/Qwen3.6-27B-FP8`).

## Run

```bash
# Full MC experiment (detach recommended; models run in parallel by default)
uv run modal run --detach run_experiment.py

# Free-form answers + Qwen 3B grading on the same 200 questions as
# 20260727T154400Z (10 shots / model)
uv run modal run --detach run_experiment.py \
  --mode freeform --source-run-id 20260727T154400Z

# Multiple judges (first is primary — drives difficulty ranking)
uv run modal run --detach run_experiment.py \
  --mode freeform --source-run-id 20260727T154400Z \
  --judge-model-ids qwen2.5-3b,qwen3.6-27b-fp8

# Smoke test one model on 8 questions (plain vLLM: SamplingParams n=2)
uv run modal run run_experiment.py \
  --models af-next-think --num-samples 8 --n-shots 2

# Resume after a crash (same question set; skip finished models / questions)
uv run modal run --detach run_experiment.py \
  --run-id <run_id>

# Re-grade / re-aggregate an existing freeform run
uv run modal run run_experiment.py \
  --grade-only --run-id <run_id> --mode freeform
# Add another judge to an existing run (only missing judge entries are graded)
uv run modal run run_experiment.py \
  --grade-only --run-id <run_id> --mode freeform \
  --judge-model-ids Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen3-8B
uv run modal run run_experiment.py \
  --aggregate-only --run-id <run_id>
```

Defaults: `--num-samples 200 --n-shots 10 --seed 42 --parallel-models --mode mc`.
Sampling (`temperature` / `top_p` / `max_tokens` / `repetition_penalty`) is
per-model in `mmar_models.MODEL_SPECS`; optional `--temperature` / `--top-p` /
`--max-new-tokens` override every model when set.

Freeform defaults `--source-run-id` to `20260727T154400Z` so the question
set matches that MC difficulty run. `--grader-model-id` remains a single-judge
alias when `--judge-model-ids` is unset.

Resume is keyed by `--run-id`. Each model appends to
`models/<label>/predictions.jsonl` and skips ids already present, so a
partial model continues mid-file. Models that already cover all sampled
questions are not re-spawned. Freeform then runs a separate grader pass
per judge (sequentially, one Modal container each) over pending shots
before aggregation.

### Throughput knobs

- All pending questions for a model go in **one** offline `generate()` call;
  vLLM continuous-batches internally (no app-level `--batch-size`).
- Plain vLLM (`af-next`, `qwen3-omni`, `voxtral`, InteractiveOmni-vLLM): one
  prompt per question with `SamplingParams(n=n_shots)` so N samples share
  prefill. Seed is per question (not per shot).
- Omni / HF: duplicate the prompt per shot; Omni regroups by shot inside the
  adapter. Prefix caching reuses identical audio/prompt prefixes where enabled.
- **`max_num_seqs`**: set from measured average sequence length against the
  reported `GPU KV cache size`, not from `max_model_len`. PagedAttention
  allocates on demand, so this is a concurrency *cap*, not a reservation.
  Oversizing causes preemption; undersizing (e.g. 4 when the cache holds
  ~180 seqs) leaves most of the GPU idle. Large models use 64 here.
- **`max_model_len`**: context safety cap for startup / overlong prompts.
  Keep it close to real prompt+output length — an inflated value deflates
  vLLM's reported max concurrency and tempts a too-low `max_num_seqs`.
- **`enforce_eager: False`**: enables torch.compile + CUDA graphs where the
  model supports it (`af-next`, `voxtral`, InteractiveOmni). Qwen3-Omni MoE
  keeps `enforce_eager: True` — torch.compile hits a Dynamo meta/cuda
  mismatch during `profile_run`; its throughput win comes from concurrency.
- Prefill-oriented: engines set `max_num_batched_tokens` (≥8192 where possible)
  and `enable_prefix_caching`. Qwen3 Thinking is decode-heavy
  (`max_tokens=2048`; measured outputs peak ~870).
- `--max-num-seqs` / `--gpu-memory-utilization`: optional CLI escape hatches.
- MiMo Omni YAML: stage-0 prefix caching on; `max_num_seqs: 1` stays for the
  two-stage memory split on one GPU (not raised for speed).
- `--grader-batch-size`: freeform judge only (separate from model inference).
- Watch logs for `Avg generation throughput`, `Running: N reqs`, and
  `GPU KV cache usage` (`disable_log_stats: False`). Confirm loaders print
  `vLLM ready` rather than `falling back to` HF.
- Qwen3 fused-MoE: image build installs an `E=128,N=768` config under both
  `NVIDIA_A100_80GB_PCIe` and `NVIDIA_A100-SXM4-80GB` names (copied from the
  H200 bf16 stand-in; vLLM ships no A100 tune for this shape).
- AF-Next / Qwen3 / Voxtral images keep `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (in-process EngineCore). Multiprocess EngineCore breaks Qwen3-Omni
  `profile_run` with a meta/cuda device mismatch.

## Download + view

```bash
# Default remote path is exp-mmar-question-difficulty/
uv run modal run download_results.py
uv run python view_difficulty.py          # results UI (:7860)
```

Open http://127.0.0.1:7860 — questions are ordered hardest-first
(ascending average `shot_success_rate` across models). Each example detail
opens with a **verdict grid** (test model × judge × shot, green/red cells).
If a run is missing `difficulty.jsonl` / `scores.json`, the viewer
aggregates them on startup.

## Re-judge a past freeform run

Add (or replace) a judge on an existing freeform run without regenerating
answers. Errors if the run is MCQ:

```bash
# Seed the new judge weights first if needed
uv run modal run seed_volume.py --datasets none --models qwen2.5-3b

# Add a judge; keep the existing primary for difficulty ranking
uv run modal run --detach rejudge_run.py \
  --run-id 20260807T145000Z \
  --judge-model-id qwen3.6-27b-fp8

# Promote the new judge to primary
uv run modal run --detach rejudge_run.py \
  --run-id 20260807T145000Z \
  --judge-model-id Qwen/Qwen3.6-27B-FP8 \
  --make-primary
```

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
outputs/exp-mmar-question-difficulty/<run_id>/
  question_ids.json
  manifest.json
  models/<label>/predictions.jsonl
  difficulty.jsonl
  scores.json
```

Freeform shot records store per-judge verdicts under `shots[].judges`:

```json
{
  "shot_index": 0,
  "answer_prediction": "…",
  "correct": false,
  "grader": "Qwen/Qwen2.5-3B-Instruct",
  "grader_output": "NO",
  "judges": {
    "qwen2.5-3b-instruct": {
      "correct": false,
      "output": "NO",
      "model_id": "Qwen/Qwen2.5-3B-Instruct"
    }
  }
}
```

Record / manifest fields: `judges` (ordered labels, `[0]` = primary),
`primary_judge`, `per_judge`, plus legacy `grader` / `scoring:
qwen_freeform_judge` mirrors of the primary.
