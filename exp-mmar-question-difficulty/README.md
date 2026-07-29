# MMAR Question Difficulty Experiment

Sample a fixed 200 MMAR questions, generate 10 responses per model
(per-model sampling; see `MODEL_SPECS`), then browse questions hardest-first
by mean success rate.

Two prompt / scoring modes:

| Mode | Prompt | Scoring |
|------|--------|---------|
| `mc` (default) | question + 4 choices | string-match against gold choice |
| `freeform` | question only (no choices) | `Qwen/Qwen2.5-3B-Instruct` judges each shot vs gold answer |

Inference uses **offline** vLLM (`LLM.generate` / Omni) with continuous
batching — not an OpenAI-compatible server. `n_shots` means independent
temperature **samples** of the same zero-shot prompt (not few-shot ICL).
Plain vLLM forks those samples with `SamplingParams(n=...)` so they share
one prefill; Omni/HF duplicate the prompt per shot and rely on prefix caching.

## Models

| Label | Checkpoint | Backend |
|-------|------------|---------|
| `af-next-think` | `nvidia/audio-flamingo-next-think-hf` | vLLM 0.24 MusicFlamingo (HF fallback); `T=1.0, max_tokens=512, rep=1.2` |
| `mimo-audio-7b` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ tokenizer) | vLLM-Omni; `T=1.0, max_tokens=512, rep=1.1` |
| `interactive-omni-8b` | `sensenova/InteractiveOmni-8B` | HF `.chat` (vLLM transformers backend incompatible); `T=1.0, max_tokens=512` |
| `qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | vLLM 0.26 thinker-only (A100-80GB); `T=0.6, top_p=0.95, top_k=20, max_tokens=16384` |
| `voxtral-small-24b` | `mistralai/Voxtral-Small-24B-2507` | vLLM 0.26 Mistral audio (A100-80GB); `T=0.2, top_p=0.95, max_tokens=512` |

`step-audio-2-mini` is temporarily excluded from difficulty aggregation.

## Seed

```bash
uv run modal run seed_volume.py --datasets mmar \
  --models af-next-think,mimo-audio-7b,interactive-omni-8b,qwen3-omni,voxtral-small-24b

# Freeform judge weights
uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
```

`mimo-audio-7b` also seeds `XiaomiMiMo/MiMo-Audio-Tokenizer` automatically.

## Run

```bash
# Full MC experiment (detach recommended; models run in parallel by default)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py

# Free-form answers + Qwen 3B grading on the same 200 questions as
# 20260727T154400Z (10 shots / model)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --mode freeform --source-run-id 20260727T154400Z

# Smoke test one model on 8 questions (plain vLLM: SamplingParams n=2)
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --models af-next-think --num-samples 8 --n-shots 2

# Resume after a crash (same question set; skip finished models / questions)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --run-id <run_id>

# Re-grade / re-aggregate an existing freeform run
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --grade-only --run-id <run_id> --mode freeform
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --aggregate-only --run-id <run_id>
```

Defaults: `--num-samples 200 --n-shots 10 --seed 42 --parallel-models --mode mc`.
Sampling (`temperature` / `top_p` / `max_tokens` / `repetition_penalty`) is
per-model in `models.MODEL_SPECS`; optional `--temperature` / `--top-p` /
`--max-new-tokens` override every model when set.

Freeform defaults `--source-run-id` to `20260727T154400Z` so the question
set matches that MC difficulty run.

Resume is keyed by `--run-id`. Each model appends to
`models/<label>/predictions.jsonl` and skips ids already present, so a
partial model continues mid-file. Models that already cover all sampled
questions are not re-spawned. Freeform then runs a separate Qwen grader
pass over pending shots before aggregation.

### Throughput knobs

- All pending questions for a model go in **one** offline `generate()` call;
  vLLM continuous-batches internally (no app-level `--batch-size`).
- Plain vLLM (`af-next`, `qwen3-omni`, `voxtral`, InteractiveOmni-vLLM): one
  prompt per question with `SamplingParams(n=n_shots)` so N samples share
  prefill. Seed is per question (not per shot).
- Omni / HF: duplicate the prompt per shot; Omni regroups by shot inside the
  adapter. Prefix caching reuses identical audio/prompt prefixes where enabled.
- `--max-num-seqs` / `--gpu-memory-utilization`: optional escape hatches.
  Prefer engine defaults; raising `max_num_seqs` past what KV cache supports
  causes preemption, not speedup. Large models keep low `max_num_seqs` as an
  OOM guard only.
- `max_model_len` is a context safety cap (startup / overlong prompts), not a
  KV-reservation throughput lever — PagedAttention allocates on demand.
- Prefill-oriented: engines set `max_num_batched_tokens` (≥8192 where possible)
  and `enable_prefix_caching`. Qwen3 Thinking is also decode-heavy
  (`max_tokens=16384`).
- MiMo Omni YAML: stage-0 prefix caching on; `max_num_seqs: 1` stays for the
  two-stage memory split on one GPU (not raised for speed).
- `--grader-batch-size`: freeform judge only (separate from model inference).

## Download + view

```bash
uv run modal run download_results.py --remote-path exp-mmar-question-difficulty
uv run python exp-mmar-question-difficulty/view_difficulty.py
```

Open http://127.0.0.1:7861 — questions are ordered hardest-first
(ascending average `shot_success_rate` across models).

## Output layout

```
outputs/exp-mmar-question-difficulty/<run_id>/
  question_ids.json
  manifest.json
  models/<label>/predictions.jsonl
  difficulty.jsonl
  scores.json
```

Freeform shot records also store `grader`, `grader_output`, and set
`scoring: qwen_freeform_judge` on the question record / manifest.
