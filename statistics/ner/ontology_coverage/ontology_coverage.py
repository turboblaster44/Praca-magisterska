"""Ontology coverage — how many of the 41 entity classes actually appear in the data."""

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
    all_labels = ENT.all()

    rows = [(lbl, lbl in counts, counts.get(lbl, 0)) for lbl in all_labels]
    table = pd.DataFrame(rows, columns=["label", "present", "count"])
    table = table.sort_values(["present", "count"], ascending=[False, False])
    table.to_csv(out / "ontology_coverage.csv", index=False)

    total = len(all_labels)
    used = int(table["present"].sum())
    unused = total - used
    pct = used / total * 100

    summary = (
        f"Ontology coverage: {used}/{total} = {pct:.1f}%\n"
        f"Used classes   : {used}\n"
        f"Unused classes : {unused}\n\n"
        f"Unused labels:\n"
        + "\n".join(f"  - {lbl}" for lbl in table.loc[~table["present"], "label"])
    )
    (out / "ontology_coverage.txt").write_text(summary, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [used, unused],
        labels=[f"Used ({used})", f"Unused ({unused})"],
        colors=["#2980b9", "#c0392b"],
        autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4),
    )
    ax.set_title(f"Ontology coverage: {used}/{total} = {pct:.1f}%")
    plt.tight_layout()
    plt.savefig(out / "ontology_coverage.png", dpi=150)
    print(summary)


if __name__ == "__main__":
    main()
