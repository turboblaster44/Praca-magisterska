"""Relation extraction step: runs every enabled method and deduplicates the union."""

import argparse

import pandas as pd

from config import (
    BIOLINK_BERT_MODEL,
    NER_OUTPUT_CSV,
    RE_OUTPUT_CSV,
    RE_RAW_OUTPUT_CSV,
    STRUCTURED_CSV,
)
from nlp._pipeline import (
    RE_COLUMNS, annotate_assertion, annotate_umls, dedup, load_csv, save_csv,
)
from nlp.re import biolink
from nlp.re import llm as llm_re
from nlp.re import spacy as spacy_rules


def run(
    df_ner: pd.DataFrame,
    df_structured: pd.DataFrame,
    biolink_model_name: str = BIOLINK_BERT_MODEL,
    use_biolink: bool = True,
    use_spacy: bool = True,
    use_llm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run enabled RE methods and return (deduplicated, raw) DataFrames."""
    print("=" * 60)
    print(f"RELATION EXTRACTION PIPELINE (biolink={use_biolink}, spacy={use_spacy}, "
          f"llm={use_llm})")
    print("=" * 60)

    parts = []
    if use_biolink:
        parts.append(biolink.run(df_ner, df_structured, biolink_model_name))
    if use_spacy:
        parts.append(spacy_rules.run(df_ner, df_structured))
    if use_llm:
        parts.append(llm_re.run(df_ner, df_structured))

    non_empty = [df for df in parts if not df.empty]
    if not non_empty:
        print("[RE] No relations extracted.")
        empty = pd.DataFrame(columns=RE_COLUMNS)
        return empty, empty

    raw = pd.concat(non_empty, ignore_index=True)
    raw = raw[[c for c in RE_COLUMNS if c in raw.columns]]

    deduped = dedup(
        raw,
        key_cols=["id", "entity1", "entity2", "relation"],
        method_priority=["spacy_lexical", "spacy_dependency", "biolink_bert", "llm"],
    )

    # Carry NER assertion flags onto relations + derive edge-level `assertion`.
    raw = annotate_assertion(raw, df_ner)
    deduped = annotate_assertion(deduped, df_ner)

    # Carry NER UMLS linking (cui/canonical_name) onto endpoints
    # so the graph can merge nodes by CUI.
    raw = annotate_umls(raw, df_ner)
    deduped = annotate_umls(deduped, df_ner)

    print(f"[RE] {len(deduped)} unique relations (raw: {len(raw)})")
    return deduped, raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the RE step.")
    parser.add_argument("--ner", default=NER_OUTPUT_CSV,
                        help="Deduplicated NER CSV.")
    parser.add_argument("--structured", default=STRUCTURED_CSV,
                        help="Structured CSV with translated text.")
    parser.add_argument("--output", default=RE_OUTPUT_CSV,
                        help="Deduplicated RE CSV.")
    parser.add_argument("--raw-output", default=RE_RAW_OUTPUT_CSV,
                        help="Raw (pre-dedup) RE CSV for method-comparison stats.")
    parser.add_argument("--biolink-model", default=BIOLINK_BERT_MODEL)
    parser.add_argument("--no-biolink", action="store_true")
    parser.add_argument("--no-spacy", action="store_true")
    parser.add_argument("--llm", action="store_true",
                        help="Enable the opt-in LLM RE method (needs an API key).")
    args = parser.parse_args(argv)

    df_ner = load_csv(args.ner)
    df_structured = load_csv(args.structured)
    deduped, raw = run(
        df_ner,
        df_structured,
        biolink_model_name=args.biolink_model,
        use_biolink=not args.no_biolink,
        use_spacy=not args.no_spacy,
        use_llm=args.llm,
    )
    save_csv(raw, args.raw_output, columns=RE_COLUMNS)
    save_csv(deduped, args.output, columns=RE_COLUMNS)
    print(f"[RE] Wrote {args.output} and {args.raw_output}")


if __name__ == "__main__":
    main()
