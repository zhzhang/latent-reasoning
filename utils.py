import gzip
import json
import os

import requests
from tqdm import tqdm


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
