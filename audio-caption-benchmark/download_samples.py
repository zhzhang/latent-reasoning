"""Download random captioning samples into a local SQLite DB + audio tree.

Full run (user):
    uv run python audio-caption-benchmark/download_samples.py

Smoke (dev / agent — tiny, isolated under data/smoke/):
    uv run python audio-caption-benchmark/download_samples.py --smoke
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from huggingface_hub import hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

from db import (  # noqa: E402
    SOURCES,
    connect,
    count_examples,
    delete_examples_for_source,
    upsert_example,
)

DEFAULT_DATA_DIR = PKG_DIR / "data"
SMOKE_DATA_DIR = PKG_DIR / "data" / "smoke"

WAVCAPS_REPO = "cvssp/WavCaps"
WAVCAPS_JSON = "json_files/SoundBible/sb_final.json"
WAVCAPS_ZIP = "Zip_files/SoundBible/SoundBible.zip"
AUDIOCAPS_DS = "d0rj/audiocaps"
AUDIOCAPS_CLIP_S = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Examples per source (default 200; --smoke forces 2).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: 2/source under data/smoke/, WavCaps via download links.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data directory (default: data/ or data/smoke/ with --smoke).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES),
        default=list(SOURCES),
        help="Subset of sources to download.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the DB already has enough examples.",
    )
    parser.add_argument(
        "--audiocaps-attempts",
        type=int,
        default=None,
        help="Max yt-dlp attempts for AudioCaps (default: max(n*8, 40)).",
    )
    return parser.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir.expanduser().resolve()
    if args.smoke:
        return SMOKE_DATA_DIR.resolve()
    return DEFAULT_DATA_DIR.resolve()


def target_n(args: argparse.Namespace) -> int:
    if args.n is not None:
        return max(1, args.n)
    return 2 if args.smoke else 200


def metadata_without_caption(row: dict[str, Any], caption_keys: tuple[str, ...] = ("caption",)) -> dict[str, Any]:
    skip = set(caption_keys) | {"audio", "array", "bytes", "path"}
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
        else:
            out[key] = str(value)
    return out


def write_audio_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def download_url(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Download URL to dest; returns the final path written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "latent-reasoning-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"Empty download from {url}")
    # Some sources return zip-wrapped wav
    if url.endswith(".zip") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise RuntimeError(f"No files in zip from {url}")
            payload = zf.read(names[0])
            suffix = Path(names[0]).suffix or ".wav"
            if dest.suffix.lower() != suffix.lower():
                dest = dest.with_suffix(suffix)
            write_audio_bytes(dest, payload)
            return dest
    write_audio_bytes(dest, data)
    return dest


def load_soundbible_json(cache_dir: Path) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=WAVCAPS_REPO,
        filename=WAVCAPS_JSON,
        repo_type="dataset",
        local_dir=cache_dir / "hub_cache",
    )
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("SoundBible JSON missing data[]")
    return rows


def match_zip_member(namelist: list[str], source_id: str) -> str | None:
    # Prefer exact basename matches containing the id.
    candidates = [
        name
        for name in namelist
        if not name.endswith("/")
        and Path(name).suffix.lower() in {".flac", ".wav", ".mp3", ".ogg"}
        and (source_id in Path(name).stem or Path(name).stem == source_id)
    ]
    if not candidates:
        # Fallback: any path ending with /{id}.ext
        for name in namelist:
            stem = Path(name).stem
            if stem == source_id or stem.endswith(f"_{source_id}") or stem.startswith(f"{source_id}_"):
                if Path(name).suffix.lower() in {".flac", ".wav", ".mp3", ".ogg"}:
                    candidates.append(name)
    if not candidates:
        return None
    # Prefer flac then wav
    candidates.sort(
        key=lambda n: (
            0 if n.lower().endswith(".flac") else 1 if n.lower().endswith(".wav") else 2,
            n,
        )
    )
    return candidates[0]


def download_wavcaps(
    *,
    conn,
    data_dir: Path,
    n: int,
    seed: int,
    smoke: bool,
    force: bool,
) -> int:
    source = "wavcaps"
    have = count_examples(conn, source)
    if have >= n and not force:
        print(f"[wavcaps] already have {have} examples (>= {n}); skip", flush=True)
        return have

    if force:
        delete_examples_for_source(conn, source)
        conn.commit()

    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = load_soundbible_json(cache_dir)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    audio_dir = data_dir / "audio" / source
    audio_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    if smoke:
        print(f"[wavcaps] smoke: downloading {n} via SoundBible download_link", flush=True)
        for row in shuffled:
            if saved >= n:
                break
            source_id = str(row.get("id") or "").strip()
            caption = str(row.get("caption") or "").strip()
            link = str(row.get("download_link") or "").strip()
            if not source_id or not caption or not link:
                continue
            dest = audio_dir / f"{source_id}.wav"
            try:
                dest = download_url(link, dest, timeout=45.0)
                if not dest.is_file() or dest.stat().st_size < 100:
                    raise RuntimeError("downloaded file missing or too small")
                rel = dest.relative_to(data_dir).as_posix()
                upsert_example(
                    conn,
                    source=source,
                    source_id=source_id,
                    caption=caption,
                    audio_path=rel,
                    metadata=metadata_without_caption(row),
                )
                saved += 1
                print(f"[wavcaps] {saved}/{n} {source_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 — keep sampling
                print(f"[wavcaps] skip {source_id}: {exc}", flush=True)
                if dest.exists():
                    dest.unlink(missing_ok=True)
        conn.commit()
        if saved < n:
            raise SystemExit(f"[wavcaps] smoke only saved {saved}/{n}")
        return saved

    print("[wavcaps] full: downloading SoundBible.zip (~580MB)…", flush=True)
    zip_path = hf_hub_download(
        repo_id=WAVCAPS_REPO,
        filename=WAVCAPS_ZIP,
        repo_type="dataset",
        local_dir=cache_dir / "hub_cache",
    )
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
        for row in tqdm(shuffled, desc="wavcaps"):
            if saved >= n:
                break
            source_id = str(row.get("id") or "").strip()
            caption = str(row.get("caption") or "").strip()
            if not source_id or not caption:
                continue
            member = match_zip_member(namelist, source_id)
            if member is None:
                continue
            suffix = Path(member).suffix.lower() or ".flac"
            dest = audio_dir / f"{source_id}{suffix}"
            try:
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            except Exception as exc:  # noqa: BLE001
                print(f"[wavcaps] extract fail {source_id}: {exc}", flush=True)
                dest.unlink(missing_ok=True)
                continue
            rel = dest.relative_to(data_dir).as_posix()
            upsert_example(
                conn,
                source=source,
                source_id=source_id,
                caption=caption,
                audio_path=rel,
                metadata=metadata_without_caption(row),
            )
            saved += 1
    conn.commit()
    if saved < n:
        raise SystemExit(f"[wavcaps] only extracted {saved}/{n}")
    print(f"[wavcaps] saved {saved}", flush=True)
    return saved


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _trim_to_wav(src: Path, dest: Path, *, start_s: float, duration_s: float) -> None:
    import torchaudio

    waveform, sr = torchaudio.load(str(src))
    start = int(start_s * sr)
    end = int((start_s + duration_s) * sr)
    if start >= waveform.shape[-1]:
        raise RuntimeError(
            f"start {start_s}s beyond audio length {waveform.shape[-1] / sr:.1f}s"
        )
    clip = waveform[:, start:end]
    if clip.numel() == 0:
        raise RuntimeError("empty trimmed clip")
    if clip.shape[0] > 1:
        clip = clip.mean(dim=0, keepdim=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dest), clip, sr)
    if float(clip.abs().max()) < 1e-6:
        raise RuntimeError("trimmed clip is silent")


def download_audiocaps_clip(
    *,
    youtube_id: str,
    start_time: int,
    dest: Path,
    timeout_s: int = 90,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    with tempfile.TemporaryDirectory(prefix="audiocaps_") as tmp:
        tmp_dir = Path(tmp)
        if _ffmpeg_available():
            section = f"*{start_time}-{start_time + AUDIOCAPS_CLIP_S}"
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "-x",
                "--audio-format",
                "wav",
                "--download-sections",
                section,
                "--force-keyframes-at-cuts",
                "-o",
                str(tmp_dir / "clip"),
                url,
            ]
        else:
            # No ffmpeg: download full audio stream, trim in Python.
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "-f",
                "bestaudio/best",
                "-o",
                str(tmp_dir / "clip.%(ext)s"),
                url,
            ]
        try:
            subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:400]
            raise RuntimeError(stderr or str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"yt-dlp timeout for {youtube_id}") from exc

        produced = [p for p in sorted(tmp_dir.glob("clip*")) if p.is_file()]
        if not produced:
            raise RuntimeError(f"yt-dlp produced no file for {youtube_id}")
        src = produced[-1]
        if _ffmpeg_available() and src.suffix.lower() == ".wav":
            shutil.copy2(src, dest)
        else:
            _trim_to_wav(
                src,
                dest,
                start_s=float(start_time),
                duration_s=float(AUDIOCAPS_CLIP_S),
            )

    if not dest.is_file() or dest.stat().st_size < 1000:
        raise RuntimeError("output missing or too small")


def download_audiocaps(
    *,
    conn,
    data_dir: Path,
    n: int,
    seed: int,
    force: bool,
    max_attempts: int,
) -> int:
    source = "audiocaps"
    have = count_examples(conn, source)
    if have >= n and not force:
        print(f"[audiocaps] already have {have} examples (>= {n}); skip", flush=True)
        return have
    if force:
        delete_examples_for_source(conn, source)
        conn.commit()

    from datasets import load_dataset

    print("[audiocaps] loading metadata…", flush=True)
    ds = load_dataset(AUDIOCAPS_DS, split="train")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    audio_dir = data_dir / "audio" / source
    audio_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    attempts = 0
    for idx in indices:
        if saved >= n or attempts >= max_attempts:
            break
        attempts += 1
        row = ds[idx]
        youtube_id = str(row["youtube_id"]).strip()
        start_time = int(row["start_time"])
        audiocap_id = str(row["audiocap_id"])
        caption = str(row["caption"]).strip()
        if not youtube_id or not caption:
            continue
        dest = audio_dir / f"{audiocap_id}.wav"
        try:
            download_audiocaps_clip(
                youtube_id=youtube_id,
                start_time=start_time,
                dest=dest,
                timeout_s=60 if n <= 5 else 120,
            )
            if not dest.is_file() or dest.stat().st_size < 1000:
                raise RuntimeError("output missing or too small")
            rel = dest.relative_to(data_dir).as_posix()
            meta = {
                "audiocap_id": int(row["audiocap_id"]),
                "youtube_id": youtube_id,
                "start_time": start_time,
                "clip_seconds": AUDIOCAPS_CLIP_S,
            }
            upsert_example(
                conn,
                source=source,
                source_id=audiocap_id,
                caption=caption,
                audio_path=rel,
                metadata=meta,
            )
            saved += 1
            print(f"[audiocaps] {saved}/{n} {youtube_id}@{start_time}s", flush=True)
            if saved % 5 == 0:
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[audiocaps] fail {youtube_id}: {exc}", flush=True)
            dest.unlink(missing_ok=True)
            for leftover in audio_dir.glob(f"{audiocap_id}.*"):
                leftover.unlink(missing_ok=True)

    conn.commit()
    if saved < n:
        raise SystemExit(
            f"[audiocaps] only saved {saved}/{n} after {attempts} attempts "
            f"(many YouTube clips are unavailable)"
        )
    print(f"[audiocaps] saved {saved}", flush=True)
    return saved


def _clotho_row_count() -> int:
    url = "https://datasets-server.huggingface.co/size?dataset=mteb/Clotho"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    for split in payload.get("size", {}).get("splits", []):
        if split.get("split") == "test":
            return int(split["num_rows"])
    return int(payload["size"]["dataset"]["num_rows"])


def _fetch_clotho_rows(offset: int, length: int = 1) -> list[dict[str, Any]]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset=mteb/Clotho&config=default&split=test"
        f"&offset={offset}&length={length}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "latent-reasoning-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [item["row"] for item in payload.get("rows", [])]


def _clotho_audio_url(audio_field: Any) -> str | None:
    if isinstance(audio_field, list) and audio_field:
        first = audio_field[0]
        if isinstance(first, dict) and first.get("src"):
            return str(first["src"])
    if isinstance(audio_field, dict) and audio_field.get("src"):
        return str(audio_field["src"])
    return None


def download_clotho(
    *,
    conn,
    data_dir: Path,
    n: int,
    seed: int,
    force: bool,
) -> int:
    """Fetch Clotho via datasets-server (audio URLs) — no multi-GB parquet pull."""
    source = "clotho"
    have = count_examples(conn, source)
    if have >= n and not force:
        print(f"[clotho] already have {have} examples (>= {n}); skip", flush=True)
        return have
    if force:
        delete_examples_for_source(conn, source)
        conn.commit()

    total_rows = _clotho_row_count()
    print(f"[clotho] sampling {n} of {total_rows} via datasets-server…", flush=True)
    rng = random.Random(seed)
    offsets = list(range(total_rows))
    rng.shuffle(offsets)

    audio_dir = data_dir / "audio" / source
    audio_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for offset in offsets:
        if saved >= n:
            break
        try:
            rows = _fetch_clotho_rows(offset, length=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[clotho] rows API fail offset={offset}: {exc}", flush=True)
            continue
        if not rows:
            continue
        row = rows[0]
        source_id = str(row.get("index") or f"offset{offset}").strip()
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_id)[:80]
        caption = str(row.get("text") or "").strip()
        audio_url = _clotho_audio_url(row.get("audio"))
        if not caption or not audio_url:
            continue
        dest = audio_dir / f"{safe_id}.wav"
        try:
            download_url(audio_url, dest, timeout=90.0)
            if not dest.is_file() or dest.stat().st_size < 1000:
                raise RuntimeError("audio missing or too small")
            rel = dest.relative_to(data_dir).as_posix()
            meta = {
                "index": row.get("index"),
                "taskname": row.get("taskname") or row.get("datasetname"),
                "audio_len": row.get("audio_len"),
                "raw_text": row.get("raw_text"),
                "row_offset": offset,
            }
            upsert_example(
                conn,
                source=source,
                source_id=safe_id,
                caption=caption,
                audio_path=rel,
                metadata=meta,
            )
            saved += 1
            print(f"[clotho] {saved}/{n} {safe_id}", flush=True)
            if saved % 10 == 0:
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[clotho] skip {safe_id}: {exc}", flush=True)
            dest.unlink(missing_ok=True)

    conn.commit()
    if saved < n:
        raise SystemExit(f"[clotho] only saved {saved}/{n}")
    print(f"[clotho] saved {saved}", flush=True)
    return saved


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args)
    n = target_n(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "audio").mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "browser.db"
    conn = connect(db_path)
    max_attempts = args.audiocaps_attempts or max(n * 8, 40)

    print(
        f"data_dir={data_dir}\n"
        f"db={db_path}\n"
        f"n_per_source={n}\n"
        f"sources={args.sources}\n"
        f"smoke={args.smoke}",
        flush=True,
    )

    totals_before = {src: count_examples(conn, src) for src in SOURCES}
    errors: list[str] = []

    if "wavcaps" in args.sources:
        try:
            download_wavcaps(
                conn=conn,
                data_dir=data_dir,
                n=n,
                seed=args.seed,
                smoke=args.smoke or n <= 5,
                force=args.force,
            )
        except SystemExit as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[wavcaps] {exc}")
    if "audiocaps" in args.sources:
        try:
            download_audiocaps(
                conn=conn,
                data_dir=data_dir,
                n=n,
                seed=args.seed,
                force=args.force,
                max_attempts=max_attempts,
            )
        except SystemExit as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[audiocaps] {exc}")
    if "clotho" in args.sources:
        try:
            download_clotho(
                conn=conn,
                data_dir=data_dir,
                n=n,
                seed=args.seed,
                force=args.force,
            )
        except SystemExit as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[clotho] {exc}")

    totals = {src: count_examples(conn, src) for src in SOURCES}
    print(f"done. counts={totals} total={sum(totals.values())}", flush=True)
    if errors:
        print("errors:", flush=True)
        for err in errors:
            print(f"  - {err}", flush=True)
        # Smoke can proceed if at least one source succeeded
        if args.smoke and sum(totals.values()) > 0:
            print("smoke: continuing despite partial failures", flush=True)
        else:
            conn.close()
            raise SystemExit(1)
    _ = totals_before
    conn.close()


if __name__ == "__main__":
    main()
