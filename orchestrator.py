"""End-to-end pipeline driver. Flip booleans in STEPS to enable/disable methods."""

from pathlib import Path

import config
from nlp._pipeline import NER_COLUMNS, RE_COLUMNS, load_csv, save_csv
from nlp import holistic
from nlp.ner import tasks as ner_tasks
from nlp.re import tasks as re_tasks
from preprocessing import extract, save, transform


STEPS = {
    "preprocessing":  False,   # parse data.txt + translate PL->EN -> structured.csv
    "ner_neural":     True,   # blaze999/Medical-NER (transformer)
    "ner_gazetteer":  True,   # PhraseMatcher + regex augmentation
    "ner_llm":        True,  # opt-in LLM NER peer method (needs API key) via llm.py
    "ner_context":    True,   # medSpaCy ConText assertion (negation/family/...)
    "ner_umls":       True,   # UMLS entity linking (CUI + semantic group) via linker.py
    "re_biolink":     True,   # BioLinkBERT zero-shot relation classifier
    "re_spacy":       True,   # spaCy dependency-rule extractor
    "re_llm":         True,  # opt-in LLM RE peer method (ontology + score) via llm.py
    "visualization":  True,   # RE results -> Neo4j-format nodes/relationships CSV
    "neo4j_import":   True,  # load CSVs into Neo4j (needs running DB + NEO4J_PASSWORD)
}


def preprocessing_step(input_path: str, structured_csv: str) -> None:
    """Build structured.csv from raw data.txt if it doesn't already exist."""
    if not STEPS["preprocessing"]:
        print("[SKIP] preprocessing"); return
    if Path(structured_csv).is_file():
        print(f"[SKIP] preprocessing — {structured_csv} already exists"); return

    data_format = "csv" if input_path.endswith(".csv") else "txt"

    raw_data = extract.run(input_path)
    transformed_data = transform.run(raw_data, data_format)
    save.run(transformed_data, structured_csv)


def ner_step(input_csv: str, output_csv: str, raw_csv: str) -> None:
    deduped, raw = ner_tasks.run(
        load_csv(input_csv),
        use_neural=STEPS["ner_neural"],
        use_gazetteer=STEPS["ner_gazetteer"],
        use_llm=STEPS["ner_llm"],
        use_context=STEPS["ner_context"],
        use_umls=STEPS["ner_umls"],
    )
    save_csv(raw, raw_csv, columns=NER_COLUMNS)
    save_csv(deduped, output_csv, columns=NER_COLUMNS)


def re_step(ner_csv: str, structured_csv: str, output_csv: str, raw_csv: str) -> None:
    deduped, raw = re_tasks.run(
        load_csv(ner_csv),
        load_csv(structured_csv),
        biolink_model_name=config.BIOLINK_BERT_MODEL,
        use_biolink=STEPS["re_biolink"],
        use_spacy=STEPS["re_spacy"],
        use_llm=STEPS["re_llm"],
    )
    save_csv(raw, raw_csv, columns=RE_COLUMNS)
    save_csv(deduped, output_csv, columns=RE_COLUMNS)


def holistic_step() -> None:
    """Holistic variant: one generative step classifies entities + relations jointly.

    Replaces ner_step + re_step; shares the same enrichment (ConText/UMLS, done
    inside holistic.run) and writes the same CSVs the graph step reads. One method
    only, so raw == deduped.
    """
    df_ner, df_rel = holistic.run(
        load_csv(config.STRUCTURED_CSV),
        use_context=STEPS["ner_context"],
        use_umls=STEPS["ner_umls"],
    )
    save_csv(df_ner, config.NER_RAW_OUTPUT_CSV, columns=NER_COLUMNS)
    save_csv(df_ner, config.NER_OUTPUT_CSV, columns=NER_COLUMNS)
    save_csv(df_rel, config.RE_RAW_OUTPUT_CSV, columns=RE_COLUMNS)
    save_csv(df_rel, config.RE_OUTPUT_CSV, columns=RE_COLUMNS)


def viz_step() -> None:
    """Convert RE results to Neo4j-format nodes.csv / relationships.csv."""
    if not STEPS.get("visualization"):
        print("[SKIP] visualization"); return
    from visualization.pipeline import run_visualization_pipeline
    run_visualization_pipeline()  # paths/threshold from visualization/config.py


def neo4j_step() -> None:
    """Load the generated CSVs into Neo4j (uri/user from config, password from .env)."""
    if not STEPS.get("neo4j_import"):
        print("[SKIP] neo4j_import"); return
    from visualization.config import (
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
        NEO4J_NODES_CSV, NEO4J_RELS_CSV,
    )
    if not NEO4J_PASSWORD:
        print("[SKIP] neo4j_import — brak NEO4J_PASSWORD (ustaw w .env)"); return
    from visualization.neo4j_integration import run_full_import
    try:
        run_full_import(
            uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE, nodes_csv=NEO4J_NODES_CSV, rels_csv=NEO4J_RELS_CSV,
        )
    except Exception as exc:  # most commonly: DB not running / wrong creds
        print(f"[neo4j_import] FAILED: {type(exc).__name__}: {exc}")
        print(f"  -> czy baza Neo4j jest uruchomiona pod {NEO4J_URI}? "
              f"sprawdź też NEO4J_USERNAME/NEO4J_PASSWORD.")


def full_pipeline() -> None:
    print("=" * 60)
    print("PIPELINE CONFIG")
    for k, v in STEPS.items():
        print(f"  {k:20s} = {v}")
    print(f"  {'PIPELINE_MODE':20s} = {config.PIPELINE_MODE}")
    print("=" * 60)

    preprocessing_step(config.INPUT_CSV, config.STRUCTURED_CSV)

    if config.PIPELINE_MODE == "holistic":
        # One generative step does NER + RE jointly (replaces the modular path).
        holistic_step()
    else:
        ner_step(config.STRUCTURED_CSV, config.NER_OUTPUT_CSV, config.NER_RAW_OUTPUT_CSV)
        re_step(config.NER_OUTPUT_CSV, config.STRUCTURED_CSV,
                config.RE_OUTPUT_CSV, config.RE_RAW_OUTPUT_CSV)

    viz_step()
    neo4j_step()


if __name__ == "__main__":
    full_pipeline()
