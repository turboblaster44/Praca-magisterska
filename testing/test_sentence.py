"""
Run a single sentence through the FULL pipeline (NER + ConText assertion + RE),
exactly like orchestrator.py, with per-step toggles and full output — including
the assertion flags (negated / uncertain / family / historical) and the
relation-level `assertion` edge status.

Usage:
    python -m testing.test_sentence "There is no sign of enlarged lymph nodes."
    python -m testing.test_sentence            # built-in example
    python -m testing.test_sentence "..." --no-context        # skip ConText
    python -m testing.test_sentence "..." --no-biolink --no-spacy  # NER only
    python -m testing.test_sentence "..." --no-neural         # gazetteer only

Toggles mirror orchestrator STEPS:
    --no-neural --no-gazetteer --no-context --no-umls   (NER methods / enrichers)
    --no-biolink --no-spacy                             (RE methods)
    --no-fhir                                           (skip FHIR graph mapping)
"""

import argparse
import csv
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import pandas as pd

import config
from nlp import holistic
from nlp._pipeline import ASSERTION_FLAGS
from nlp.ner import tasks as ner_tasks
from nlp.re import tasks as re_tasks
from visualization.fhir import build_fhir_layer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
RECORD_ID = 0

DEFAULT_SENTENCE = "There is no sign of enlarged lymph nodes; the patient denies fever."


def _flag_markers(row) -> str:
    """Compact marker string for the assertion flags present on a row."""
    marks = {
        "is_negated": "NEG", "is_uncertain": "UNC",
        "is_family": "FAM", "is_historical": "HIST",
    }
    on = [m for col, m in marks.items() if col in row and bool(row[col])]
    return ",".join(on) if on else "-"


def print_ner(df_ner: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STEP 1 - Named Entity Recognition (+ ConText assertion)")
    print("=" * 70)
    if df_ner.empty:
        print("  (no entities found)")
        return
    has_assert = any(f in df_ner.columns for f in ASSERTION_FLAGS)
    header = f"  {'entity':<28}{'label':<24}{'method':<11}{'score':<7}"
    if has_assert:
        header += "flags"
    print(header)
    print("  " + "-" * 68)
    for _, r in df_ner.iterrows():
        line = (f"  {str(r['word'])[:27]:<28}{str(r['label'])[:23]:<24}"
                f"{str(r.get('method',''))[:10]:<11}{float(r['score']):.3f}  ")
        if has_assert:
            line += _flag_markers(r)
        print(line)


def print_umls(df_ner: pd.DataFrame) -> None:
    """Show the UMLS entity-linking columns, when the linker ran."""
    if df_ner.empty or "cui" not in df_ner.columns:
        return
    print("\n" + "=" * 70)
    print("STEP 1b - UMLS entity linking (CUI + canonical name)")
    print("=" * 70)
    print(f"  {'entity':<24}{'CUI':<11}{'canonical_name':<30}score")
    print("  " + "-" * 68)
    for _, r in df_ner.iterrows():
        cui = str(r.get("cui", "") or "")
        if not cui:
            print(f"  {str(r['word'])[:23]:<24}{'(no match)':<11}")
            continue
        print(
            f"  {str(r['word'])[:23]:<24}{cui:<11}"
            f"{str(r.get('canonical_name',''))[:29]:<30}"
            f"{str(r.get('umls_score',''))}"
        )


def print_re(df_rel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STEP 2 - Relation Extraction (+ edge assertion)")
    print("=" * 70)
    if df_rel.empty:
        print("  (no relations found)")
        return
    for _, r in df_rel.iterrows():
        assertion = f"  {{assertion={r['assertion']}}}" if "assertion" in r else ""
        print(
            f"  [{r['entity1_label']}] {r['entity1']}"
            f"  --{r['relation']}-->"
            f"  [{r['entity2_label']}] {r['entity2']}"
            f"  (score={float(r['score']):.3f}, method={r['method']}){assertion}"
        )


def print_fhir(patients: pd.DataFrame, concepts: pd.DataFrame,
               facts: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Show the FHIR-reification graph (GRAPH_MODE=fhir) built from NER + RE.

    Makes visible where each RE relation ended up: a fact<->fact edge, a field on
    a fact, or dropped (only the NER-derived SUBJECT / HAS_CODE / MENTIONS edges
    remain for it).
    """
    print("\n" + "=" * 70)
    print("STEP 3 - FHIR graph mapping (patient-centric)")
    print("=" * 70)

    # Resolve node keys -> readable names for the edge listing.
    key2name: dict[str, str] = {}
    for _, p in patients.iterrows():
        key2name[str(p["key"])] = f"Patient {p['id']}"
    for _, c in concepts.iterrows():
        key2name[str(c["key"])] = f"{c['name']}"
    for _, f in facts.iterrows():
        key2name[str(f["key"])] = f"{f['word']}"

    def nm(key) -> str:
        k = str(key)
        return key2name.get(k, k)

    # --- Patients ---
    print("\n  :Patient")
    demo_cols = [c for c in patients.columns if c not in ("key", "id")]
    for _, p in patients.iterrows():
        demo = [f"{c}={p[c]}" for c in demo_cols if str(p[c]).strip()]
        extra = f"  ({', '.join(demo)})" if demo else ""
        print(f"    [{p['id']}] Patient{extra}")

    # --- Facts (reified mentions) ---
    print("\n  Facts (reified mentions):")
    if facts.empty:
        print("    (none)")
    for _, f in facts.iterrows():
        attrs = [f"{col}={f[col]}" for col in config.ATTRIBUTE_FIELDS
                 if col in f and str(f[col]).strip()]
        attr_str = f", {', '.join(attrs)}" if attrs else ""
        cui = f", cui={f['cui']}" if str(f["cui"]).strip() else ""
        rel = f", relative={f['relative']}" if ("relative" in f and str(f["relative"]).strip()) else ""
        print(f"    [{f['resource']}] {f['word']}"
              f"  (assertion={f['assertion']}, verif={f['verificationStatus']},"
              f" status={f['clinicalStatus']}{cui}{rel}{attr_str})")

    # --- Concepts (shared, merged by CUI) ---
    print("\n  Concepts (shared, merged by CUI-or-string):")
    if concepts.empty:
        print("    (none)")
    for _, c in concepts.iterrows():
        cui = f" ({c['cui']})" if str(c["cui"]).strip() else ""
        print(f"    [{c['entity_type']}] {c['name']}{cui}  x{c['count']}")

    # --- Edges ---
    print("\n  Edges:")
    if edges.empty:
        print("    (none)")
    for _, e in edges.iterrows():
        assertion = f"  {{assertion={e['assertion']}}}" if str(e["assertion"]).strip() else ""
        print(f"    ({nm(e['start_key'])}) -[{e[':TYPE']}]-> ({nm(e['end_key'])}){assertion}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Full pipeline on one sentence.")
    parser.add_argument("sentence", nargs="?", default=DEFAULT_SENTENCE)
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--no-gazetteer", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--no-umls", action="store_true")
    parser.add_argument("--no-biolink", action="store_true")
    parser.add_argument("--no-spacy", action="store_true")
    parser.add_argument("--no-fhir", action="store_true")
    parser.add_argument("--llm", action="store_true",
                        help="Enable the opt-in LLM NER + RE methods (needs an API key).")
    parser.add_argument("--holistic", action="store_true",
                        help="Holistic variant: one LLM step for entities + relations "
                             "(replaces NER+RE; needs an API key).")
    args = parser.parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_structured = pd.DataFrame([{
        "id": RECORD_ID, "text": args.sentence, "text_en": args.sentence,
    }])

    print("\n" + "=" * 70)
    print("SENTENCE")
    print("=" * 70)
    print(" ", args.sentence)

    if args.holistic:
        # Holistic variant: one generative step for entities + relations jointly.
        df_ner, df_rel = holistic.run(
            df_structured,
            use_context=not args.no_context,
            use_umls=not args.no_umls,
        )
    else:
        # --- NER (+ ConText) ---
        df_ner, _ = ner_tasks.run(
            df_structured,
            use_neural=not args.no_neural,
            use_gazetteer=not args.no_gazetteer,
            use_llm=args.llm,
            use_context=not args.no_context,
            use_umls=not args.no_umls,
        )

        # --- RE (+ assertion annotation) ---
        df_rel, _ = re_tasks.run(
            df_ner,
            df_structured,
            biolink_model_name=config.BIOLINK_BERT_MODEL,
            use_biolink=not args.no_biolink,
            use_spacy=not args.no_spacy,
            use_llm=args.llm,
        )

    # --- Save ---
    ner_csv = os.path.join(OUTPUT_DIR, "ner_results.csv")
    re_csv = os.path.join(OUTPUT_DIR, "re_results.csv")
    df_ner.to_csv(ner_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    df_rel.to_csv(re_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # --- Report ---
    print_ner(df_ner)
    print_umls(df_ner)
    print_re(df_rel)

    # --- FHIR graph mapping (reads the CSVs just written) ---
    if not args.no_fhir:
        if "cui" not in df_ner.columns:
            print("\n[FHIR] skipped: needs UMLS columns (run without --no-umls).")
        else:
            patients, concepts, facts, edges = build_fhir_layer(ner_csv, re_csv)
            print_fhir(patients, concepts, facts, edges)

    print(f"\n  Saved -> {ner_csv}")
    print(f"  Saved -> {re_csv}")


if __name__ == "__main__":
    main()
