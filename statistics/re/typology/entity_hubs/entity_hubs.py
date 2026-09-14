"""Entity hubs — in-degree / out-degree of entities in the relation graph."""

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
    out_deg = (df.groupby(["entity1", "entity1_label"]).size()
                 .reset_index(name="out_degree")
                 .rename(columns={"entity1": "entity", "entity1_label": "label"}))
    in_deg = (df.groupby(["entity2", "entity2_label"]).size()
                .reset_index(name="in_degree")
                .rename(columns={"entity2": "entity", "entity2_label": "label"}))

    merged = pd.merge(out_deg, in_deg, on=["entity", "label"], how="outer").fillna(0)
    merged["out_degree"] = merged["out_degree"].astype(int)
    merged["in_degree"] = merged["in_degree"].astype(int)
    merged["total"] = merged["out_degree"] + merged["in_degree"]
    merged = merged.sort_values("total", ascending=False).reset_index(drop=True)
    merged.to_csv(out / "entity_hubs.csv", index=False)
    print(merged.head(TOP_N).to_string(index=False))

    top = merged.head(TOP_N)
    labels = [f"{e} ({l})" for e, l in zip(top["entity"], top["label"])]

    fig, ax = plt.subplots(figsize=(11, max(6, TOP_N * 0.35)))
    ax.barh(labels[::-1], top["out_degree"].values[::-1], color="#2980b9", label="out_degree")
    ax.barh(labels[::-1], top["in_degree"].values[::-1],
            left=top["out_degree"].values[::-1], color="#27ae60", label="in_degree")
    ax.set_xlabel("Degree")
    ax.set_title(f"Top-{TOP_N} entity hubs by total degree")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "entity_hubs.png", dpi=150)


if __name__ == "__main__":
    main()
