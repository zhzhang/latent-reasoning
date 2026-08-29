# Project context

Research code for a NeurIPS 2026 paper on whether audio-language models can judge the
correctness of free-form answers to audio-reasoning questions **without a reference
answer**. Deadline is imminent. Prefer small, verifiable changes over refactors.

## The finding this code supports

With the gold answer, judges match human annotators. Without it, none do. Removing the
audio from a reference-supplied judge changes agreement by under 1%, so the reference,
not the audio, is doing the work. RQ4 asks whether giving the judge audio-analysis tools
recovers any of the gap.

## Layout

- `tooljudge/` is mine, on branch `tool-call-judge`. Everything else is Jordan's.
- `audio_tools.py` four CPU tools the judge may call: overview, transcribe,
  tempo/pitch over time, onset timeline.
- `judge.py` the Anthropic tool-use loop and verdict parser.
- `local_tool_judge.py` batch runner over a downloaded run directory.
- `score.py` agreement metrics and the alt-test.
- `check_oracle.py` validity checks on the Claude+GT oracle.

## Hard constraints

- **No Modal.** I have no Modal account. Never suggest `modal run`, and never edit
  anything that requires it: `run_experiment.py`, `seed_volume.py`, `rejudge_run.py`,
  `tune_judge.py`, `grader.py`, `download_results.py`, `modal_cache.py`.
- **No GPU.** Judges are the Anthropic API; tools are librosa and faster-whisper on CPU.
- **Do not modify files outside `tooljudge/`** without asking. Other people are working
  in this repo against the same deadline.
- **Never commit** `.env`, API keys, `mmar/`, `outputs/`, or any `predictions.jsonl`.
- Package manager is `uv`, not pip. Use `uv add`, `uv run`.

## Output schema, do not break it

Verdicts are written into `shots[].judges[<label>]` in the exact shape `grader.py`
produces, so `view_difficulty.py` and `aggregate.py` read them unchanged:
`{correct, verdict, output, generation, model_id}`. Extra keys are fine. Three labels:
`claude-audio-nogt`, `claude-tooljudge-nogt`, `claude-text-nogt`.
Never change `primary_judge` in `manifest.json`; it drives Jordan's difficulty ranking.

## Data

- `labels.csv` human pass/fail, `ratings` is a JSON list of 3 booleans.
  149 questions, 4,023 rows, 3,429 with all three raters.
  Cohen's kappa 0.73-0.78, Krippendorff alpha 0.835, annotators split on 16.4%.
- `generations.csv` model answers. Join on `generation_id`, but verify: ids may have
  been reassigned in the newer thinking-models-only run. Fall back to
  `(question_id, model_label, shot_index)`.
- MMAR audio in `mmar/audio/`, metadata `mmar/MMAR-meta.jsonl` keyed by `id`.
  Fields used: `question`, `answer`, `category`, `modality`.

## Conventions

- Every batch script resumes. Never re-run work that is already on disk.
- Write output atomically via a temp file and `shutil.move`; a crash must not corrupt
  Jordan's `predictions.jsonl`.
- Cache tool results by audio path; three shots of one question re-analyse audio once.
- Log the parse rate alongside accuracy. A judge that emits no parseable verdict looks
  identical to a judge that fails everything, and that has already happened once here.
- Smoke test with `--limit 5` before any full run. API calls cost money.

## What I am usually asking for

Changes to tool descriptions and the judge system prompt in `audio_tools.py` and
`judge.py`, then a 5-shot run to see whether the judge actually calls the tools and
cites their output. If it ignores them, that is the experiment, not a bug.
