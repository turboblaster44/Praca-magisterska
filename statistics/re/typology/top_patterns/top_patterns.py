"""Top relation patterns: (entity1_label, relation, entity2_label) frequency."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import RE_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent
TOP_N = 20


def main(argv=None) -> None:
    args = stats_argparser(ROOT / RE_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    table = (df.groupby(["entity1_label", "relation", "entity2_label"])
               .size().reset_index(name="count")
               .sort_values("count", ascending=False))
    total = int(table["count"].sum())
    table["pct"] = table["count"] / total * 100
    table.to_csv(out / "top_patterns.csv", index=False)
    print(table.head(TOP_N).to_string(index=False))

    top = table.head(TOP_N)
    labels = [f"({r.entity1_label}) -[{r.relation}]-> ({r.entity2_label})"
              for r in top.itertuples()]

    fig, ax = plt.subplots(figsize=(12, max(6, TOP_N * 0.35)))
    ax.barh(labels[::-1], top["count"].values[::-1], color="#8e44ad")
    ax.set_xlabel("Count")
    ax.set_title(f"Top-{TOP_N} relation patterns (of {len(table)} distinct, {total} total)")
    plt.tight_layout()
    plt.savefig(out / "top_patterns.png", dpi=150)


if __name__ == "__main__":
    main()
