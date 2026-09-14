"""Distribution of relation types across the whole RE output."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import RE_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / RE_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    counts = df["relation"].value_counts()
    total = int(counts.sum())

    table = pd.DataFrame({
        "relation": counts.index,
        "count": counts.values,
        "pct": counts.values / total * 100,
    })
    table["cum_pct"] = table["pct"].cumsum()
    table.to_csv(out / "relation_distribution.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(table["relation"], table["count"], color="#2980b9")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of relation types ({total} relations, {len(counts)} types)")
    plt.xticks(rotation=45, ha="right")
    for i, (c, p) in enumerate(zip(table["count"], table["pct"])):
        ax.text(i, c, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "relation_distribution.png", dpi=150)


if __name__ == "__main__":
    main()
