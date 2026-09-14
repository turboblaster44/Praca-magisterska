"""Confidence score distribution per NER method (gazetteer typically constant 1.0)."""

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

    df = pd.read_csv(args.input)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    stats = df.groupby("method")["score"].describe()[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    stats.to_csv(out / "confidence_stats.csv")
    print(stats)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in [("neural", "#2980b9"), ("gazetteer", "#27ae60")]:
        vals = df[df["method"] == method]["score"]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.6, label=f"{method} (n={len(vals)})", color=color)
    ax.set_xlabel("Score"); ax.set_ylabel("Frequency")
    ax.set_title("NER confidence distribution per method (raw, pre-dedup)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "confidence_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    methods = df["method"].unique()
    ax.boxplot([df[df["method"] == m]["score"].values for m in methods], tick_labels=methods)
    ax.set_ylabel("Score")
    ax.set_title("NER confidence — boxplot per method (raw, pre-dedup)")
    plt.tight_layout()
    plt.savefig(out / "confidence_box.png", dpi=150)


if __name__ == "__main__":
    main()
