# MMAR Question Difficulty Experiment

Sample a fixed 200 MMAR questions, generate 10 responses per model
(per-model sampling; see `MODEL_SPECS`), then browse questions hardest-first
by mean success rate.

Two prompt / scoring modes:

| Mode | Prompt | Scoring |
|------|--------|---------|
| `mc` (default) | question + 4 choices | string-match against gold choice (stored as a synthetic `string-match` judge) |
| `freeform` | question only (no choices) | one or more local vLLM judges grade each shot vs gold; first judge is **primary** and drives difficulty ranking |

Default freeform judge: `Qwen/Qwen3.6-35B-A3B-FP8`. Pass multiple with `--judge-model-ids` (comma-separated; first = primary).

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
uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8
```

`mimo-audio-7b` also seeds `XiaomiMiMo/MiMo-Audio-Tokenizer` automatically.
Judge aliases: `qwen2.5-3b`, `qwen3.6-35b-a3b-fp8` (→ `Qwen/Qwen3.6-35B-A3B-FP8`).

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
  --judge-model-ids qwen2.5-3b,qwen3.6-35b-a3b-fp8

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
- Judge engines are tuned with `tune_judge.py` (see below). For the
  `qwen3.6-35b-a3b-fp8` MoE judge, `enforce_eager: False` was worth 332 →
  4,824 output tok/s on an H100; only 3B params are active per token, so
  decode is kernel-launch bound and CUDA graphs dominate everything else.
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
grades every other pack model's shots. Pass ids to run only those judges.
Shots that already have a verdict for the same judge key are skipped. Pass
`--force` to replace existing verdicts.

vLLM suite / dedicated judges run on Modal (this script starts a detached
App). API judges (`gemini-3.7-flash`; aliases
`gemini-3.7-mini`, `gemini`, or `api`) run locally against the
same pack and do not start Modal. Empty `--judge-model-id` stays
suite-only so a bare run does not spend API quota. API judges skip their own pack label (round-robin).
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

outputs/exp-mmar-question-difficulty/<run_id>/
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
