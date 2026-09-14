"""Method overlap — relations found by both methods vs only one (raw, pre-dedup)."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import RE_RAW_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent
KEY = ["id", "entity1", "relation", "entity2"]


def main(argv=None) -> None:
    args = stats_argparser(ROOT / RE_RAW_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    s_keys = set(map(tuple, df[df["method"] == "spacy_lexical"][KEY].drop_duplicates().values.tolist()))
    b_keys = set(map(tuple, df[df["method"] == "biolink_bert"][KEY].drop_duplicates().values.tolist()))

    both = s_keys & b_keys
    only_spacy = s_keys - b_keys
    only_bert = b_keys - s_keys
    total = len(s_keys | b_keys)

    rows = [
        ("both",       len(both),       len(both)       / total * 100 if total else 0),
        ("only_spacy", len(only_spacy), len(only_spacy) / total * 100 if total else 0),
        ("only_bert",  len(only_bert),  len(only_bert)  / total * 100 if total else 0),
    ]
    table = pd.DataFrame(rows, columns=["category", "count", "pct"])
    table.to_csv(out / "overlap.csv", index=False)

    def to_df(keys): return pd.DataFrame(list(keys), columns=KEY).head(20)
    to_df(both).to_csv(out / "sample_both.csv", index=False)
    to_df(only_spacy).to_csv(out / "sample_only_spacy.csv", index=False)
    to_df(only_bert).to_csv(out / "sample_only_bert.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(table["category"], table["count"], color=["#8e44ad", "#2980b9", "#27ae60"])
    ax.set_ylabel("Unique relations")
    ax.set_title(f"Method overlap (union = {total} unique relations)")
    for b, c, p in zip(bars, table["count"], table["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c} ({p:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out / "overlap.png", dpi=150)
    print(table.to_string(index=False))
    print(f"Union: {total}, |spacy|={len(s_keys)}, |bert|={len(b_keys)}")


if __name__ == "__main__":
    main()
