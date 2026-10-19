"""Download Criteo Uplift v2.1 ONCE (run in Colab) and save a random sample as Parquet.

    python -m src.criteo.prepare --out data/criteo_1m.parquet --n 1000000

Source: Criteo's Hugging Face mirror (the old go.criteo.net and scikit-uplift links are dead).
The 311 MB .csv.gz is downloaded to the runtime's LOCAL disk (not Drive), read in chunks, and sampled
row by row with a fixed seed. Rows are never selected by outcome, so the sample stays unbiased.
Only the small sample Parquet is written to your project folder.

Test without internet:  --src path/to/local_file.csv.gz
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

URLS = [
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
]
EXPECTED_BYTES = 311_422_618
EXPECTED_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
PUBLISHED = {"rows": 13_979_592, "visit_rate": 0.046992, "conversion_rate": 0.00292, "treatment_ratio": 0.85}
FEATS = [f"f{i}" for i in range(12)]
DTYPES = {**{c: "float32" for c in FEATS}, "treatment": "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"}


def download(dest: Path) -> Path:
    import requests
    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES:
        print(f"already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in URLS:
        try:
            print("downloading", url)
            sha = hashlib.sha256()
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk); sha.update(chunk)
            size = dest.stat().st_size
            print(f"downloaded {size / 1e6:.1f} MB")
            if size != EXPECTED_BYTES or sha.hexdigest() != EXPECTED_SHA256:
                print("WARNING: size or SHA-256 differs from the published file. "
                      "Continue only if you accept that; row counts below will show whether it is the v2.1 file.")
            return dest
        except Exception as exc:                       # try the next mirror
            last = exc
            print("  failed:", exc)
    raise RuntimeError(f"all download sources failed (last error: {last}). Download the file manually and use --src.")


def sample_stream(src: str, n: int, seed: int, chunksize: int = 1_000_000) -> tuple[pd.DataFrame, dict]:
    """One pass: keep each row with probability p = n / total (Bernoulli sampling) + full-data summary stats."""
    rng = np.random.default_rng(seed)
    p = min(1.0, n / PUBLISHED["rows"])
    kept, tot = [], {"rows": 0, "treated": 0, "visit": 0, "conversion": 0}
    for chunk in pd.read_csv(src, chunksize=chunksize, dtype=DTYPES):
        tot["rows"] += len(chunk)
        tot["treated"] += int(chunk.treatment.sum())
        tot["visit"] += int(chunk.visit.sum())
        tot["conversion"] += int(chunk.conversion.sum())
        kept.append(chunk[rng.random(len(chunk)) < p])
        print(f"  read {tot['rows']:>11,d} rows, kept {sum(len(k) for k in kept):>9,d}", flush=True)
    return pd.concat(kept, ignore_index=True), tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/criteo_1m.parquet")
    ap.add_argument("--n", type=int, default=1_000_000, help="approximate number of rows to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", default=None, help="local .csv/.csv.gz to use instead of downloading")
    ap.add_argument("--raw-dir", default="/content/criteo_raw" if os.path.exists("/content") else "data/raw")
    a = ap.parse_args()

    src = a.src or str(download(Path(a.raw_dir) / "criteo-research-uplift-v2.1.csv.gz"))
    sample, tot = sample_stream(src, a.n, a.seed)

    r = tot["rows"]
    print("\nFULL DATA          rows=%s  treatment_ratio=%.4f  visit_rate=%.5f  conversion_rate=%.5f"
          % (f"{r:,}", tot["treated"] / r, tot["visit"] / r, tot["conversion"] / r))
    print("PUBLISHED          rows=13,979,592  treatment_ratio~0.85  visit_rate=0.04699  conversion_rate=0.00292")
    print("SAMPLE             rows=%s  treatment_ratio=%.4f  visit_rate=%.5f  conversion_rate=%.5f"
          % (f"{len(sample):,}", sample.treatment.mean(), sample.visit.mean(), sample.conversion.mean()))
    if r != PUBLISHED["rows"] and not a.src:
        print("WARNING: row count differs from the published v2.1 file.")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False, compression="zstd")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
