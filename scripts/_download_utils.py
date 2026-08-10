"""Shared resumable-download helper for the dataset acquisition scripts.

The dataset hosts used here (FMA's switch.ch mirror, DEAM's cvml.unige.ch)
reset connections or stall mid-transfer often enough on multi-GB files that
a plain single-shot `requests.get` isn't reliable. This resumes via HTTP
Range requests and retries with backoff instead of restarting from zero.
"""

import time
from pathlib import Path

import requests
from tqdm import tqdm

MAX_RETRIES = 20
RETRY_BACKOFF_S = 5
READ_TIMEOUT_S = 60


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    total = int(requests.head(url, timeout=30).headers.get("content-length", 0))
    bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)

    for attempt in range(1, MAX_RETRIES + 1):
        resume_at = tmp.stat().st_size if tmp.exists() else 0
        bar.n = resume_at
        bar.refresh()
        if total and resume_at >= total:
            break
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, READ_TIMEOUT_S)) as r:
                r.raise_for_status()
                mode = "ab" if resume_at else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt == MAX_RETRIES:
                bar.close()
                raise
            wait = RETRY_BACKOFF_S * attempt
            tqdm.write(f"  retry {attempt}/{MAX_RETRIES} after error ({e}); resuming in {wait}s from byte {tmp.stat().st_size if tmp.exists() else 0}")
            time.sleep(wait)

    bar.close()
    if total and tmp.stat().st_size != total:
        raise IOError(f"Incomplete download: got {tmp.stat().st_size} of {total} bytes for {dest.name}")
    tmp.rename(dest)
