"""
Demo: what SciSpaCy and medSpaCy add on top of plain spaCy.

Run:
    python testing/demo_scispacy_medspacy.py

Builds ONE shared pipeline on the SciSpaCy biomedical model and layers the
medSpaCy clinical components on top, then showcases four capabilities on
curated English clinical sentences:

  1. SciSpaCy  – biomedical NER + dependency parsing  (en_core_sci_md)
  2. SciSpaCy  – abbreviation detection               (AbbreviationDetector)
  3. medSpaCy  – assertion / ConText                   (negation, uncertainty,
                                                        family, historical)
  4. medSpaCy  – clinical section detection            (Sectionizer)

How this maps to the project NER pipeline (nlp/ner/tasks.py):
  - (1) en_core_sci_md is a drop-in replacement for en_core_web_sm — a better
        parser for the rule-based RE step (nlp/re/spacy.py).
  - (3) ConText is the recommended enrichment layer: it would annotate every
        entity coming out of the neural + gazetteer ensemble with
        is_negated / is_family / is_historical / is_uncertain.
  - (4) Sectionizer can drive FAMILY_HISTORY / HISTORY from context instead of
        the hand-written word lists in nlp/ner/ontology.py.
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import spacy
import medspacy  # noqa: F401 — importing registers the medspacy_* pipe factories
from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
from scispacy.linking import EntityLinker  # noqa: F401 — registers scispacy_linker

SCISPACY_MODEL = "en_core_sci_md"


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_pipeline():
    """SciSpaCy base model + medSpaCy clinical layers, all in one nlp object."""
    nlp = spacy.load(SCISPACY_MODEL)
    nlp.add_pipe("abbreviation_detector")  # SciSpaCy
    nlp.add_pipe("medspacy_context")       # medSpaCy — assertion / ConText
    nlp.add_pipe("medspacy_sectionizer")   # medSpaCy — section detection
    return nlp


# ---------------------------------------------------------------------------
# 1. SciSpaCy — biomedical NER + dependency parsing
# ---------------------------------------------------------------------------
def demo_ner_and_parse(nlp) -> None:
    banner("1. SciSpaCy — biomedical NER + dependency parsing (en_core_sci_md)")

    text = (
        "A 54-year-old man with type 2 diabetes mellitus presented with "
        "progressive dyspnea and bilateral pulmonary infiltrates on chest CT."
    )
    doc = nlp(text)

    print(f"Text: {text}\n")
    print("Detected biomedical entities (single generic 'ENTITY' label):")
    for ent in doc.ents:
        print(f"   - {ent.text}")

    print("\nDependency parse of a causal clause:")
    doc2 = nlp("Aspirin reduces the risk of myocardial infarction.")
    for tok in doc2:
        if tok.dep_ in {"nsubj", "ROOT", "dobj", "pobj"}:
            print(f"   {tok.text:14} dep={tok.dep_:8} head={tok.head.text}")


# ---------------------------------------------------------------------------
# 2. SciSpaCy — abbreviation detection
# ---------------------------------------------------------------------------
def demo_abbreviations(nlp) -> None:
    banner("2. SciSpaCy — abbreviation detection (AbbreviationDetector)")

    text = (
        "The patient was diagnosed with chronic obstructive pulmonary disease "
        "(COPD) and acute myocardial infarction (AMI). COPD was managed with "
        "inhalers, while the AMI required urgent intervention."
    )
    doc = nlp(text)
    print(f"Text: {text}\n")
    print("Short form  ->  long form (resolved from context):")
    seen = set()
    for abbr in doc._.abbreviations:
        key = (abbr.text, str(abbr._.long_form))
        if key in seen:
            continue
        seen.add(key)
        print(f"   {abbr.text:6} -> {abbr._.long_form}")


# ---------------------------------------------------------------------------
# 3. medSpaCy — assertion / ConText (the key feature the project lacks)
# ---------------------------------------------------------------------------
def demo_context(nlp) -> None:
    banner("3. medSpaCy — assertion / ConText (negation, uncertainty, family, history)")

    sentences = [
        "The patient reports severe chest pain.",          # affirmed
        "The patient denies any fever or cough.",          # negated
        "Findings are suggestive of possible pneumonia.",  # uncertain
        "Mother had breast cancer at age 50.",             # family
        "Past medical history of myocardial infarction.",  # historical
        "No evidence of metastatic disease.",              # negated
    ]

    print(f"{'entity':22}{'NEGATED':9}{'UNCERTAIN':11}{'FAMILY':8}{'HISTORICAL':11}")
    print("-" * 70)
    for sent in sentences:
        doc = nlp(sent)
        for ent in doc.ents:
            print(
                f"{ent.text[:21]:22}"
                f"{str(ent._.is_negated):9}"
                f"{str(ent._.is_uncertain):11}"
                f"{str(ent._.is_family):8}"
                f"{str(ent._.is_historical):11}"
            )
    print(
        "\nWhy it matters: the project's neural + gazetteer NER assigns every "
        "match\nscore=1.0 regardless of negation. 'denies fever' and 'has fever'\n"
        "look identical without ConText — a classic precision leak in clinical NER."
    )


# ---------------------------------------------------------------------------
# 4. medSpaCy — clinical section detection
# ---------------------------------------------------------------------------
def demo_sectionizer(nlp) -> None:
    banner("4. medSpaCy — clinical section detection (Sectionizer)")

    note = (
        "Chief Complaint:\n"
        "Chest pain for two days.\n\n"
        "Past Medical History:\n"
        "Hypertension and type 2 diabetes.\n\n"
        "Family History:\n"
        "Father had coronary artery disease.\n\n"
        "Medications:\n"
        "Aspirin 81 mg daily and metformin 500 mg twice daily.\n"
    )
    doc = nlp(note)

    print("Detected sections:")
    for section in doc._.sections:
        if section.title_start is not None and section.title_end is not None:
            title = doc[section.title_start:section.title_end].text.strip()
        else:
            title = "(none)"
        print(f"   [{section.category}]  title={title!r}")

    print("\nEntities tagged with their section (drives FAMILY_HISTORY / HISTORY):")
    for ent in doc.ents:
        if ent._.section_category:
            print(f"   {ent.text[:30]:32} -> {ent._.section_category}")


# ---------------------------------------------------------------------------
# 5. SciSpaCy — entity linking to UMLS (CUI + canonical concept)
# ---------------------------------------------------------------------------
def demo_entity_linking() -> None:
    banner("5. SciSpaCy — entity linking to UMLS (CUI normalisation)")

    print(
        "Loading the UMLS EntityLinker (downloads a ~1 GB knowledge base on the\n"
        "FIRST run; cached afterwards — setup_venv.py pre-downloads it).\n"
    )

    # Linker on its own model: it maps each entity to UMLS concepts (CUIs).
    nlp = spacy.load(SCISPACY_MODEL)
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": True, "linker_name": "umls"},
    )
    linker = nlp.get_pipe("scispacy_linker")

    text = (
        "The patient was diagnosed with myocardial infarction and "
        "type 2 diabetes mellitus, and treated with metformin."
    )
    doc = nlp(text)
    print(f"Text: {text}\n")

    for ent in doc.ents:
        if not ent._.kb_ents:
            print(f"   {ent.text:28} -> (no UMLS match)")
            continue
        cui, score = ent._.kb_ents[0]  # best candidate
        concept = linker.kb.cui_to_entity[cui]
        print(f"   {ent.text:28} -> {cui}  {concept.canonical_name}  (score={score:.2f})")
        # Semantic types map onto the project's ENT classes; one definition line:
        if concept.types:
            print(f"   {'':28}    types={concept.types}")

    print(
        "\nWhy it matters: 'MI', 'heart attack' and 'myocardial infarction' all\n"
        "collapse to the same CUI — semantic dedup the project's string-based\n"
        "dedup (nlp/ner/tasks.py) cannot do. UMLS also contains SNOMED + MeSH."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SciSpaCy + medSpaCy capability demo.")
    parser.add_argument(
        "--no-linking",
        action="store_true",
        help="Skip the UMLS entity-linking section (avoids the ~1 GB KB download).",
    )
    args = parser.parse_args()

    print("Loading SciSpaCy model + medSpaCy components ...")
    nlp = build_pipeline()
    print("Pipeline:", nlp.pipe_names)

    demo_ner_and_parse(nlp)
    demo_abbreviations(nlp)
    demo_context(nlp)
    demo_sectionizer(nlp)

    if args.no_linking:
        print("\n[skipped] entity linking (--no-linking)")
    else:
        try:
            demo_entity_linking()
        except Exception as exc:  # KB not cached / download failed
            print(f"\n[entity linking unavailable] {type(exc).__name__}: {exc}")
            print("Run setup_venv.py (pre-downloads the UMLS KB) or omit --no-linking.")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
