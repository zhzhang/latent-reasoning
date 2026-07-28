# MMAR Question Difficulty Experiment

Sample a fixed 200 MMAR questions, generate 10 responses per model at
temperature 1.0 (string-match scoring only — no rubrics), then browse
questions hardest-first by mean success rate.

Inference uses **vLLM** (and **vLLM-Omni** for MiMo) with continuous
batching for higher GPU throughput.

## Models

| Label | Checkpoint | Backend |
|-------|------------|---------|
| `af-next-think` | `nvidia/audio-flamingo-next-think-hf` | vLLM 0.24 MusicFlamingo (HF fallback) |
| `mimo-audio-7b` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ tokenizer) | vLLM-Omni |
| `interactive-omni-8b` | `sensenova/InteractiveOmni-8B` | HF `.chat` (vLLM transformers backend incompatible) |

`step-audio-2-mini` is temporarily excluded from difficulty aggregation.

## Seed

```bash
uv run modal run seed_volume.py --datasets mmar \
  --models af-next-think,mimo-audio-7b,interactive-omni-8b
```

`mimo-audio-7b` also seeds `XiaomiMiMo/MiMo-Audio-Tokenizer` automatically.

## Run

```bash
# Full experiment (detach recommended; models run in parallel by default)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py

# Smoke test one model on 8 questions (shots flattened into one generate)
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --models af-next-think --num-samples 8 --n-shots 2 --batch-size 8

# Larger continuous-batch width (throughput)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --batch-size 32 --max-num-seqs 48

# Resume after a crash (same question set; skip finished models / questions)
uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \
  --run-id <run_id>

# Re-aggregate an existing run
uv run modal run exp-mmar-question-difficulty/run_experiment.py \
  --aggregate-only --run-id <run_id>
```

Defaults: `--num-samples 200 --n-shots 10 --temperature 1.0 --seed 42
--batch-size 16 --parallel-models`.

Resume is keyed by `--run-id`. Each model appends to
`models/<label>/predictions.jsonl` and skips ids already present, so a
partial model continues mid-file. Models that already cover all sampled
questions are not re-spawned.

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
