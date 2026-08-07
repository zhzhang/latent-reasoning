# Audio Caption Benchmark Browser

Local tool to sample audio-caption datasets, browse clips with captions/metadata, and draft Q&A pairs into a small benchmark stored in SQLite.

Sources (200 each on a full run):

- [cvssp/WavCaps](https://huggingface.co/datasets/cvssp/WavCaps) — SoundBible subset (zip on full run)
- [d0rj/audiocaps](https://huggingface.co/datasets/d0rj/audiocaps) — YouTube clips via `yt-dlp`
- [mteb/Clotho](https://huggingface.co/datasets/mteb/Clotho) — via datasets-server audio URLs (no full parquet pull)

## Quick start (smoke)

Tiny isolated run under `data/smoke/` (2 examples per source; WavCaps uses per-clip links, not the 580MB zip):

```bash
uv run python audio-caption-benchmark/download_samples.py --smoke
uv run python audio-caption-benchmark/browse.py --smoke --self-check
uv run python audio-caption-benchmark/browse.py --smoke
# open http://127.0.0.1:7870
```

## Full corpus (you run this)

```bash
uv run python audio-caption-benchmark/download_samples.py
uv run python audio-caption-benchmark/browse.py
```

Options:

| Flag | Meaning |
|------|---------|
| `--n N` | Examples per source (default 200; smoke uses 2) |
| `--sources wavcaps clotho` | Subset of sources |
| `--force` | Re-download even if DB already has enough |
| `--seed 42` | Sampling seed |
| `--data-dir PATH` | Override data root |

## Browser

- Primary: audio player + caption
- Auxiliary: other dataset metadata
- Benchmark form: question + answer (saved to SQLite; not required for every example)
- Filters: source, annotated / unannotated
- Export: `/api/export` → JSONL of completed Q&A rows

## Layout

```
audio-caption-benchmark/
  download_samples.py
  browse.py
  db.py
  data/                 # gitignored
    browser.db
    audio/{wavcaps,audiocaps,clotho}/
    smoke/              # smoke isolation
```

## Notes

- WavCaps is for academic / research use only (see dataset card).
- AudioCaps depends on YouTube availability; the downloader oversamples until it hits `--n` successes.
- **ffmpeg** (optional but recommended): enables fast AudioCaps section downloads. Without it, the script downloads full audio and trims with `torchaudio`.
- Requires network access; full WavCaps pull downloads ~580MB `SoundBible.zip`.
