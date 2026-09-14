"""NER label distribution — count of entities per label across the ontology."""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "statistics")))
from _base import ROOT, ensure_dir, stats_argparser  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import NER_OUTPUT_CSV  # noqa: E402
from nlp.ner.ontology import ENT  # noqa: E402


DEFAULT_OUT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    args = stats_argparser(ROOT / NER_OUTPUT_CSV, DEFAULT_OUT, __doc__).parse_args(argv)
    out = ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, dtype=str)
    counts = df["label"].value_counts().to_dict()
    rows = sorted(
        [(lbl, counts.get(lbl, 0)) for lbl in ENT.all()],
        key=lambda r: r[1], reverse=True,
    )
    total = sum(c for _, c in rows)

    table = pd.DataFrame(
        [(lbl, c, (c / total * 100 if total else 0.0)) for lbl, c in rows],
        columns=["label", "count", "pct"],
    )
    table.to_csv(out / "label_distribution.csv", index=False)

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#c0392b" if v == 0 else "#2980b9" for v in values]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Entity count")
    ax.set_title(f"NER label distribution ({total} entities, 41 classes, red = unused)")
    plt.tight_layout()
    plt.savefig(out / "label_distribution.png", dpi=150)
    print(f"Wrote {out/'label_distribution.csv'} and {out/'label_distribution.png'}")


if __name__ == "__main__":
    main()
