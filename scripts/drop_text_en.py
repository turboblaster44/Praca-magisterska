"""
Remove the text_en column from a structured CSV (in place).

Usage:
    python scripts/drop_text_en.py path/to/file.csv

The path is required: this rewrites the file in place, so there is no default
that could strip the translation off the pipeline input by accident.
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/drop_text_en.py path/to/file.csv")
        sys.exit(1)
    path = Path(sys.argv[1])

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text_en" not in df.columns:
        print(f"No text_en column in {path}, nothing to do.")
        return

    df = df.drop(columns=["text_en"])
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Dropped text_en, wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
