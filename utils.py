import gzip
import json
import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

MMAR_DATASET_REPO = "BoJack/MMAR"
MMAR_AUDIO_ARCHIVE = "mmar-audio.tar.gz"


def download_url(url: str, folder: str = "folder") -> str:
    """Download a file from ``url`` into ``folder`` and return the local path."""
    os.makedirs(folder, exist_ok=True)
    filename = url.rpartition("/")[2]
    filename = filename if filename else "downloaded_file"
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        print(f"File already exists: {path}")
        return path

    print(f"Downloading {url} to {path} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    return path


def load_jsonl(
    file_path: str,
    instruction: str = "instruction",
    input: str = "input",
    output: str = "output",
    category: str = "category",
    is_gzip: bool = False,
):
    """Load a JSONL dataset, remapping fields onto a common schema.

    Each returned dict has ``instruction`` and ``output`` keys (others are kept
    when present in the source records).
    """
    open_func = open if not is_gzip else gzip.open
    list_data_dict = []
    with open_func(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            new_item = {
                "instruction": item.get(instruction),
                "output": item.get(output),
            }
            if input in item:
                new_item["input"] = item[input]
            if category in item:
                new_item["category"] = item[category]
            list_data_dict.append(new_item)
    return list_data_dict


def count_wav_files(audio_dir):
    """Return the number of ``.wav`` files directly under ``audio_dir``."""
    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        return 0
    return sum(1 for path in audio_path.iterdir() if path.is_file() and path.suffix.lower() == ".wav")


def ensure_mmar_audio(
    data_root,
    audio_dir=None,
    min_wav_files=1000,
    force_download=False,
):
    """Download and extract the MMAR audio archive if local clips are missing.

    The archive is cached under ``<data_root>/data/mmar-audio.tar.gz`` and
    extracted so metadata paths like ``./audio/<id>.wav`` resolve under
    ``<data_root>/audio``.
    """
    data_root = Path(data_root).expanduser().resolve()
    audio_path = Path(audio_dir).expanduser().resolve() if audio_dir else data_root / "audio"
    cache_dir = data_root / "data"
    archive_path = cache_dir / MMAR_AUDIO_ARCHIVE

    wav_count = count_wav_files(audio_path)
    if wav_count >= min_wav_files and not force_download:
        print(f"MMAR audio already present: {wav_count} wav files in {audio_path}")
        return str(audio_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download MMAR audio. Try:\n"
            "  pip install huggingface_hub\n"
            f"Original import error: {exc}"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists() or force_download:
        print(f"Downloading {MMAR_AUDIO_ARCHIVE} from {MMAR_DATASET_REPO} ...")
        downloaded = hf_hub_download(
            repo_id=MMAR_DATASET_REPO,
            filename=MMAR_AUDIO_ARCHIVE,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        archive_path = Path(downloaded)
    else:
        print(f"Using cached archive: {archive_path}")

    print(f"Extracting {archive_path} into {data_root} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_root)

    wav_count = count_wav_files(audio_path)
    print(f"Extracted {wav_count} wav files to {audio_path}")
    if wav_count < min_wav_files:
        raise RuntimeError(
            f"Expected at least {min_wav_files} wav files in {audio_path}, found {wav_count}."
        )
    return str(audio_path)
