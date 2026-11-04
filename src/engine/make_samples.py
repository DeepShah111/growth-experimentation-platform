"""Build the small committed sample files the deployed app needs.

    python -m src.engine.make_samples                  # Hillstrom (downloads if needed) + a 100k Criteo sample if data/criteo_1m.parquet exists

Outputs go to sample_data/ (git-tracked, tiny): hillstrom.csv.gz (~1 MB) and criteo_100k.parquet (~2 MB).
"""
from __future__ import annotations

import gzip

import pandas as pd

from src.engine.datasets import HILLSTROM_URL, ROOT, SAMPLE_DIR, load_hillstrom


def main() -> None:
    SAMPLE_DIR.mkdir(exist_ok=True)
    out = SAMPLE_DIR / "hillstrom.csv.gz"
    if not out.exists():
        df = load_hillstrom(allow_download=True)
        with gzip.open(out, "wt") as f:
            df.to_csv(f, index=False)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) from {HILLSTROM_URL}")
    else:
        print(f"{out} already there ({out.stat().st_size / 1e6:.1f} MB)")
    src = ROOT / "data" / "criteo_1m.parquet"
    if src.exists():
        d = pd.read_parquet(src).sample(n=100_000, random_state=2026).reset_index(drop=True)
        dst = SAMPLE_DIR / "criteo_100k.parquet"
        d.to_parquet(dst, index=False, compression="zstd")
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB), treatment share {d.treatment.mean():.3f}")
    else:
        print("data/criteo_1m.parquet not found: run python -m src.criteo.prepare first if you want the Criteo sample page")


if __name__ == "__main__":
    main()
