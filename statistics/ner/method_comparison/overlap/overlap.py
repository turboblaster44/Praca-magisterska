"""Method overlap — entities found by both NER methods vs only one (raw, pre-dedup)."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import NER_RAW_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent
KEY = ["id", "word_lower", "label"]


def main(argv=None) -> None:
    args = stats_argparser(ROOT / NER_RAW_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    df["word_lower"] = df["word"].astype(str).str.lower()

    n_keys = set(map(tuple, df[df["method"] == "neural"][KEY].drop_duplicates().values.tolist()))
    g_keys = set(map(tuple, df[df["method"] == "gazetteer"][KEY].drop_duplicates().values.tolist()))

    both = n_keys & g_keys
    only_neural = n_keys - g_keys
    only_gaz = g_keys - n_keys
    total = len(n_keys | g_keys)

    rows = [
        ("both",           len(both),        len(both)        / total * 100 if total else 0),
        ("only_neural",    len(only_neural), len(only_neural) / total * 100 if total else 0),
        ("only_gazetteer", len(only_gaz),    len(only_gaz)    / total * 100 if total else 0),
    ]
    table = pd.DataFrame(rows, columns=["category", "count", "pct"])
    table.to_csv(out / "overlap.csv", index=False)

    def to_df(keys): return pd.DataFrame(list(keys), columns=KEY).head(20)
    to_df(both).to_csv(out / "sample_both.csv", index=False)
    to_df(only_neural).to_csv(out / "sample_only_neural.csv", index=False)
    to_df(only_gaz).to_csv(out / "sample_only_gazetteer.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(table["category"], table["count"], color=["#8e44ad", "#2980b9", "#27ae60"])
    ax.set_ylabel("Unique entities")
    ax.set_title(f"NER method overlap (union = {total} unique entities)")
    for b, c, p in zip(bars, table["count"], table["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c} ({p:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out / "overlap.png", dpi=150)
    print(table.to_string(index=False))
    print(f"Union: {total}, |neural|={len(n_keys)}, |gazetteer|={len(g_keys)}")


if __name__ == "__main__":
    main()
