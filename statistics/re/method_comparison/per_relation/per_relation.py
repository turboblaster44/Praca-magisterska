"""Per-relation breakdown — which relation types each RE method finds (raw, pre-dedup)."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import RE_RAW_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / RE_RAW_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    pivot = df.groupby(["relation", "method"]).size().unstack(fill_value=0).reset_index()
    for m in ["spacy_lexical", "biolink_bert"]:
        if m not in pivot.columns:
            pivot[m] = 0
    pivot["total"] = pivot["spacy_lexical"] + pivot["biolink_bert"]
    pivot = pivot.sort_values("total", ascending=False)
    pivot[["relation", "spacy_lexical", "biolink_bert", "total"]].to_csv(
        out / "per_relation.csv", index=False
    )
    print(pivot.to_string(index=False))

    relations = pivot["relation"].tolist()
    x = np.arange(len(relations)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(relations) * 0.7), 6))
    ax.bar(x - w/2, pivot["spacy_lexical"], w, label="spacy_lexical", color="#2980b9")
    ax.bar(x + w/2, pivot["biolink_bert"],  w, label="biolink_bert",  color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(relations, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Relation counts per method")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "per_relation.png", dpi=150)


if __name__ == "__main__":
    main()
