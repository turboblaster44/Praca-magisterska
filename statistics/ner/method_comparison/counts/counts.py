"""Counts of NER entities per extraction method (pre-dedup)."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import NER_RAW_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / NER_RAW_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    counts = df["method"].value_counts()
    total = int(counts.sum())

    table = pd.DataFrame({
        "method": counts.index,
        "count": counts.values,
        "pct": counts.values / total * 100,
    })
    table.to_csv(out / "counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(table["method"], table["count"], color=["#2980b9", "#27ae60"])
    ax.set_ylabel("Entity count")
    ax.set_title(f"NER method comparison — counts (total = {total})")
    for b, c, p in zip(bars, table["count"], table["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c} ({p:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out / "counts.png", dpi=150)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
