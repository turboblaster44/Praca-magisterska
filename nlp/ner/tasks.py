"""NER step: runs every enabled method and deduplicates the union."""

import argparse
import ast
from functools import lru_cache

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from config import (
    NER_OUTPUT_CSV, NER_RAW_OUTPUT_CSV, HF_MEDICAL_NER_MODEL, STRUCTURED_CSV,
    HF_NER_AGGREGATION, NER_SCORE_THRESHOLD,
)
from nlp._pipeline import NER_COLUMNS, dedup, load_csv, save_csv
from nlp.ner import assertion, gazetteer, linker, llm


# Which method wins when several describe the same stretch of text. The LLM goes
# first: measured against the reference annotation of test_data/structured_easy.csv
# it is the only source that gets whole spans right — the gazetteer contributes a
# bare "X-ray" where the LLM has "chest x-ray", and the neural model swallows a
# whole clause ("mother had breast cancer") where the LLM separates "mother" and
# "breast cancer". Ordering the other way costs 2 points of F1.
_METHOD_PRECEDENCE = ["llm", "gazetteer", "neural"]


def _resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one mention per stretch of text.

    Walks each record's spans in method-precedence order (longest first within a
    method) and drops any span overlapping one already kept, so "chest" + "pain"
    cannot coexist with "chest pain". Rows without offsets are always kept.
    """
    if df.empty:
        return df

    rank = {m: i for i, m in enumerate(_METHOD_PRECEDENCE)}
    ordered = df.assign(
        _rank=df["method"].map(lambda m: rank.get(m, len(rank))),
        _len=df["end"].fillna(0).astype(int) - df["start"].fillna(0).astype(int),
    ).sort_values(["_rank", "_len"], ascending=[True, False], kind="stable")

    keep = []
    for _, grp in ordered.groupby("id", sort=False):
        taken: list[tuple[int, int]] = []
        for idx, row in grp.iterrows():
            if pd.isna(row["start"]) or pd.isna(row["end"]):
                keep.append(idx)
                continue
            start, end = int(row["start"]), int(row["end"])
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            keep.append(idx)

    dropped = len(df) - len(keep)
    if dropped:
        print(f"[NER] overlap resolution: dropped {dropped} overlapping mentions.")
    return df.loc[sorted(keep)].reset_index(drop=True)


@lru_cache(maxsize=1)
def _neural_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(HF_MEDICAL_NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        HF_MEDICAL_NER_MODEL
    )
    model.eval()
    return pipeline(
        "ner", model=model, tokenizer=tokenizer,
        # Strategy set in config (HF_NER_AGGREGATION); see the trade-off note there.
        aggregation_strategy=HF_NER_AGGREGATION, device=device,
    )


def _reshape_neural(df: pd.DataFrame, lang_col: str) -> pd.DataFrame:
    """Explode HuggingFace pipeline's per-row entity lists into one row per entity."""
    df = df.copy()
    df["entities"] = df["entities"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df.explode("entities")
    df = df[df["entities"].notna()]

    entities = pd.json_normalize(df["entities"])
    out = pd.concat(
        [
            entities[["word", "entity_group", "score", "start", "end"]].reset_index(drop=True),
            df[lang_col].reset_index(drop=True),
            df["id"].reset_index(drop=True),
        ],
        axis=1,
    ).rename(columns={"entity_group": "label"})
    # Normalize label case so different model backends align on one ontology:
    # d4data emits "Disease_disorder", blaze999 "DISEASE_DISORDER". Upper-casing
    # unifies them with the ENT ontology, the gazetteer labels, dedup keys, and
    # visualization's ENTITY_LABEL_MAP.
    out["label"] = out["label"].astype(str).str.upper()
    out["method"] = "neural"
    return out[["id", "word", "label", "score", lang_col, "method", "start", "end"]]


def _run_neural(df: pd.DataFrame, lang_col: str, batch_size: int = 32) -> pd.DataFrame:
    ner_pipe = _neural_pipeline()

    df = df[df[lang_col].notna()]
    df = df[df[lang_col].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(
        lambda batch: {"entities": ner_pipe(batch[lang_col], batch_size=batch_size)},
        batched=True, batch_size=batch_size,
    )

    return _reshape_neural(dataset.to_pandas()[["id", lang_col, "entities"]], lang_col)


def run(
    df_structured: pd.DataFrame,
    lang_col: str = "text_en",
    use_neural: bool = True,
    use_gazetteer: bool = True,
    use_llm: bool = False,
    use_context: bool = True,
    use_umls: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run enabled NER methods and return (deduplicated, raw DataFrames)."""
    print("=" * 60)
    print(f"NER PIPELINE (neural={use_neural}, gazetteer={use_gazetteer}, "
          f"llm={use_llm}, context={use_context}, umls={use_umls})")
    print("=" * 60)

    parts = []
    if use_neural:
        parts.append(_run_neural(df_structured, lang_col))
    if use_gazetteer:
        parts.append(gazetteer.run(df_structured))
    if use_llm:
        parts.append(llm.run(df_structured, lang_col))

    if not parts:
        print("[NER] No methods enabled.")
        empty = pd.DataFrame(columns=NER_COLUMNS)
        return empty, empty

    raw = pd.concat(parts, ignore_index=True)
    raw = raw[[c for c in NER_COLUMNS if c in raw.columns]]
    # The neural model sometimes spans a lone whitespace character; the resulting
    # empty surface form is a substring of every sentence downstream.
    raw = raw[raw["word"].notna() & (raw["word"].astype(str).str.strip() != "")]

    # Filter a COPY: `raw` is the audit trail (ner_results_raw.csv) and must keep the
    # rejected mentions, otherwise the threshold can never be re-tuned from a saved run.
    kept = raw
    if NER_SCORE_THRESHOLD > 0:
        kept = raw[raw["score"].astype(float) >= NER_SCORE_THRESHOLD]
        print(f"[NER] score >= {NER_SCORE_THRESHOLD}: kept {len(kept)}/{len(raw)} mentions.")

    keyed = kept.copy()
    keyed["_word_lower"] = keyed["word"].astype(str).str.lower()
    deduped = dedup(
        keyed,
        key_cols=["id", "_word_lower"],  # label NOT in the key — see _METHOD_PRECEDENCE
        method_priority=_METHOD_PRECEDENCE,
    ).drop(columns="_word_lower")
    deduped = _resolve_overlaps(deduped)

    if use_context:
        deduped = assertion.enrich(deduped, df_structured, text_col=lang_col)
    if use_umls:
        deduped = linker.link(deduped, df_structured, text_col=lang_col)

    print(f"[NER] {len(deduped)} unique entities (raw: {len(raw)})")
    return deduped, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the NER step.")
    parser.add_argument("--input", default=STRUCTURED_CSV,
                        help="Structured CSV produced by preprocessing.")
    parser.add_argument("--output", default=NER_OUTPUT_CSV,
                        help="Deduplicated NER CSV.")
    parser.add_argument("--raw-output", default=NER_RAW_OUTPUT_CSV,
                        help="Raw (pre-dedup) NER CSV for method-comparison stats.")
    parser.add_argument("--lang-col", default="text_en")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--no-gazetteer", action="store_true")
    parser.add_argument("--llm", action="store_true",
                        help="Enable the opt-in LLM NER method (needs an API key).")
    parser.add_argument("--no-context", action="store_true",
                        help="Skip the medSpaCy ConText assertion enrichment.")
    parser.add_argument("--no-umls", action="store_true",
                        help="Skip the UMLS entity-linking enrichment.")
    args = parser.parse_args(argv)

    df_structured = load_csv(args.input)
    deduped, raw = run(
        df_structured,
        lang_col=args.lang_col,
        use_neural=not args.no_neural,
        use_gazetteer=not args.no_gazetteer,
        use_llm=args.llm,
        use_context=not args.no_context,
        use_umls=not args.no_umls,
    )
    save_csv(raw, args.raw_output, columns=NER_COLUMNS)
    save_csv(deduped, args.output, columns=NER_COLUMNS)
    print(f"[NER] Wrote {args.output} and {args.raw_output}")


if __name__ == "__main__":
    main()