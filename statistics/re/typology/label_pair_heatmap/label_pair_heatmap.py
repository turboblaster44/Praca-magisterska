"""Heatmap of entity-label pair co-occurrences in relations."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import RE_OUTPUT_CSV  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / RE_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    matrix = df.groupby(["entity1_label", "entity2_label"]).size().unstack(fill_value=0)
    matrix.to_csv(out / "label_pair_matrix.csv")
    print(matrix)

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.7),
                                    max(6, matrix.shape[0] * 0.5)))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("entity2_label"); ax.set_ylabel("entity1_label")
    ax.set_title("Entity-label pair co-occurrences in relations")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if v:
                ax.text(j, i, int(v), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out / "label_pair_heatmap.png", dpi=150)


if __name__ == "__main__":
    main()
