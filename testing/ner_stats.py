"""
NER entity coverage statistics.

Reads a NER results CSV and prints a table showing, for every ENT label:
  - total mention count
  - % of all mentions
  - number of documents containing at least one entity of that type
  - % of all documents that contain it

Sorted from most-found to least. Labels with zero occurrences are shown last.

Usage:
    python -m testing.ner_stats                        # uses config.NER_OUTPUT_CSV
    python -m testing.ner_stats data/ner_results1.csv  # custom path
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import NER_OUTPUT_CSV, STRUCTURED_CSV
from nlp.ner.ontology import ENT

ALL_LABELS = sorted(ENT.all())


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else NER_OUTPUT_CSV
    df = pd.read_csv(csv_path)

    total_mentions = len(df)
    total_docs     = df["id"].nunique()

    # Per-label counts
    mention_counts = df["label"].value_counts()
    doc_counts     = df.groupby("label")["id"].nunique()

    rows = []
    for label in ALL_LABELS:
        mentions  = int(mention_counts.get(label, 0))
        docs      = int(doc_counts.get(label, 0))
        pct_ment  = mentions / total_mentions * 100 if total_mentions else 0.0
        pct_docs  = docs     / total_docs     * 100 if total_docs     else 0.0
        rows.append((label, mentions, pct_ment, docs, pct_docs))

    rows.sort(key=lambda r: r[1], reverse=True)

    # ── column widths ──────────────────────────────────────────────────────
    W_LABEL = max(len(r[0]) for r in rows)
    W_CNT   = max(len(str(r[1])) for r in rows)
    W_CNT   = max(W_CNT, len("mentions"))
    HEADER  = (
        f"{'ENTITY LABEL':<{W_LABEL}}  "
        f"{'mentions':>{W_CNT}}  "
        f"{'% mentions':>10}  "
        f"{'docs':>6}  "
        f"{'% docs':>8}"
    )
    SEP = "-" * len(HEADER)

    print()
    print(f"  NER coverage  —  {csv_path}")
    print(f"  {total_mentions} mentions  |  {total_docs} documents")
    print(SEP)
    print(HEADER)
    print(SEP)

    for label, mentions, pct_ment, docs, pct_docs in rows:
        print(
            f"{label:<{W_LABEL}}  "
            f"{mentions:>{W_CNT}}  "
            f"{pct_ment:>9.1f}%  "
            f"{docs:>6}  "
            f"{pct_docs:>7.1f}%"
        )

    print(SEP)

    zero = [r[0] for r in rows if r[1] == 0]
    if zero:
        print(f"\n  Labels with 0 occurrences ({len(zero)}):  {', '.join(zero)}")
    print()


if __name__ == "__main__":
    main()
