# MMAR Question Difficulty Experiment

Sample a fixed 200 MMAR questions, generate 10 responses per model at
temperature 1.0, then browse questions hardest-first by mean success rate.

Two prompt / scoring modes:

| Mode | Prompt | Scoring |
|------|--------|---------|
| `mc` (default) | question + 4 choices | string-match against gold choice |
| `freeform` | question only (no choices) | `Qwen/Qwen2.5-3B-Instruct` judges each shot vs gold answer |

Inference uses **vLLM** (and **vLLM-Omni** for MiMo) with continuous
batching for higher GPU throughput.

## Models

| Label | Checkpoint | Backend |
|-------|------------|---------|
| `af-next-think` | `nvidia/audio-flamingo-next-think-hf` | vLLM 0.24 MusicFlamingo (HF fallback) |
| `mimo-audio-7b` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ tokenizer) | vLLM-Omni |
| `interactive-omni-8b` | `sensenova/InteractiveOmni-8B` | HF `.chat` (vLLM transformers backend incompatible) |
| `qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | vLLM 0.26 thinker-only (A100-80GB); sampling `T=0.6, top_p=0.95, top_k=20, max_tokens=16384` |
| `voxtral-small-24b` | `mistralai/Voxtral-Small-24B-2507` | vLLM 0.26 Mistral audio (A100-80GB); sampling `T=0.2, top_p=0.95` |

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

# Smoke test one model on 8 questions (shots flattened into one generate)
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --models af-next-think --num-samples 8 --n-shots 2 --batch-size 8

# Larger continuous-batch width (throughput)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --batch-size 32 --max-num-seqs 48

# Resume after a crash (same question set; skip finished models / questions)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --run-id <run_id>

# Re-grade / re-aggregate an existing freeform run
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --grade-only --run-id <run_id> --mode freeform
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --aggregate-only --run-id <run_id>
```

Defaults: `--num-samples 200 --n-shots 10 --temperature 1.0 --seed 42
--batch-size 16 --parallel-models --mode mc`.

Freeform defaults `--source-run-id` to `20260727T154400Z` so the question
set matches that MC difficulty run.

Resume is keyed by `--run-id`. Each model appends to
`models/<label>/predictions.jsonl` and skips ids already present, so a
partial model continues mid-file. Models that already cover all sampled
questions are not re-spawned. Freeform then runs a separate Qwen grader
pass over pending shots before aggregation.

### Throughput knobs

- `--batch-size`: questions per checkpoint wave; all `n_shots` for those questions go in one generate queue (prefix caching reuses shared audio/prompt across shots on vLLM). Omni regroups by shot inside the adapter because stage sampling params are shared per call.
- `--max-num-seqs` / `--gpu-memory-utilization`: override vLLM engine packing (AF-Next default `max_num_seqs=32`)
- `--parallel-models` / `--no-parallel-models`: run model workers concurrently or sequentially
- Omni stage YAMLs under `deploy/` raise `max_num_seqs` / GPU util vs upstream defaults; prefix caching is enabled for n-shot prefill reuse

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
