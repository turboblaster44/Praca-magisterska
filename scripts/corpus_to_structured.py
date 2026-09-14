"""
One-time script: convert data/corpus_summary_all.csv to structured CSV format.

Input  (semicolon-delimited): filename;id;phrase;comments;mos_pred;grade;...
Output (CSV): id,text,text_en

Steps:
  1. Read corpus_summary_all.csv (delimiter=';')
  2. Keep (id, phrase), deduplicate by id
  3. Translate phrase (PL) -> text_en (EN) via MarianMT
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from preprocessing.transform import translate, clean_text
from config import CORPUS_SUMMARY_CSV, DATA_DIR


SRC = ROOT / CORPUS_SUMMARY_CSV
DST = ROOT / DATA_DIR / "structured_from_corpus.csv"


def main() -> None:
    df = pd.read_csv(SRC, delimiter=";", dtype=str)

    df = df[["id", "phrase"]].rename(columns={"phrase": "text"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"Translating {len(df)} unique records...")
    df["text_en"] = [translate(t) for t in df["text"]]

    df = df[["id", "text", "text_en"]]
    df.to_csv(DST, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows -> {DST}")


if __name__ == "__main__":
    main()
