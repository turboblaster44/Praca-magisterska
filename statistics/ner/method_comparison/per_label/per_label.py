"""Per-label breakdown — which entity labels each NER method finds (raw, pre-dedup)."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import NER_RAW_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / NER_RAW_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    pivot = df.groupby(["label", "method"]).size().unstack(fill_value=0).reset_index()
    for m in ["neural", "gazetteer"]:
        if m not in pivot.columns:
            pivot[m] = 0
    pivot["total"] = pivot["neural"] + pivot["gazetteer"]
    pivot = pivot.sort_values("total", ascending=False)
    pivot[["label", "neural", "gazetteer", "total"]].to_csv(out / "per_label.csv", index=False)
    print(pivot.to_string(index=False))

    labels = pivot["label"].tolist()
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
    ax.bar(x - w/2, pivot["neural"],    w, label="neural",    color="#2980b9")
    ax.bar(x + w/2, pivot["gazetteer"], w, label="gazetteer", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("NER entity counts per method (raw, pre-dedup)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "per_label.png", dpi=150)


if __name__ == "__main__":
    main()
