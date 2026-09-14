"""
UMLS entity linking for NER entities (swappable backend).

Adds UMLS columns to the existing NER entities WITHOUT extracting new ones:
maps each entity to a UMLS CUI + canonical name + linker score. Runs as a
sub-step after dedup/ConText, mirroring the Doc rebuild + char-offset
placement used by ``assertion.enrich``.

Backends (config.UMLS_LINKER):
  - "scispacy" : scispaCy UMLS EntityLinker — in-process, no UMLS license
                 needed (AI2 ships the KB). Alias/TF-IDF matching, no context.
  - "medcat"   : reserved — would read an out-of-process MedCAT output CSV and
                 join by (id, offset). Not implemented until the model pack
                 is available; raises NotImplementedError for now.

The value payoff downstream: entities carrying a CUI can be MERGEd on the CUI
in the graph ("aspirin"/"ASA" -> one node), which string dedup can't do.
"""

from functools import lru_cache

import pandas as pd
import spacy
from tqdm import tqdm

from config import SCISPACY_MODEL, UMLS_LINKER, UMLS_LINKER_THRESHOLD
from nlp.ner.assertion import _span_for_row

# Columns appended to the NER DataFrame (subset of NER_COLUMNS).
LINKER_COLUMNS = ["cui", "canonical_name", "umls_score"]


@lru_cache(maxsize=1)
def _load_scispacy():
    """SciSpaCy tokenizer + UMLS EntityLinker pipe (loads the ~1 GB KB once).

    NER is disabled so the sci model's own entities don't replace the ones we
    inject from the NER pipeline. ``resolve_abbreviations`` is off because we
    link the exact spans blaze999/gazetteer produced (no abbreviation pipe run).
    """
    from scispacy.linking import EntityLinker  # noqa: F401 — registers the factory

    nlp = spacy.load(SCISPACY_MODEL, disable=["ner"])
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": False, "linker_name": "umls",
                "threshold": UMLS_LINKER_THRESHOLD},
    )
    return nlp, nlp.get_pipe("scispacy_linker")


def _init_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    # Force object dtype: these columns hold mixed str + float (umls_score), and
    # a bare `df[col] = ""` can infer pandas' arrow-backed string dtype, which
    # then rejects the float score on assignment.
    for col in LINKER_COLUMNS:
        df[col] = pd.Series([""] * len(df), index=df.index, dtype="object")
    return df


def _link_scispacy(df_ner: pd.DataFrame, df_structured: pd.DataFrame,
                   text_col: str) -> pd.DataFrame:
    df = _init_columns(df_ner)
    if df.empty:
        return df

    nlp, linker = _load_scispacy()
    text_by_id = df_structured.set_index("id")[text_col].to_dict()

    print(f"[UMLS] Linking {len(df)} entities via scispaCy UMLS linker ...")

    for rec_id, grp in tqdm(df.groupby("id"), total=df["id"].nunique(),
                            desc="UMLS", unit="rec"):
        text = text_by_id.get(rec_id)
        if not isinstance(text, str) or not text.strip():
            continue

        doc = nlp.make_doc(text)  # tokenize only; no pipes run yet
        span_to_rows: dict[tuple[int, int], list[int]] = {}
        spans = []
        for idx, row in grp.iterrows():
            span = _span_for_row(doc, row, text)
            if span is None:
                continue
            span_to_rows.setdefault((span.start_char, span.end_char), []).append(idx)
            spans.append(span)

        if not spans:
            continue

        doc.ents = spacy.util.filter_spans(spans)  # linker reads doc.ents
        linker(doc)  # run only the linker component on our injected entities

        for ent in doc.ents:
            if not ent._.kb_ents:
                continue
            cui, score = ent._.kb_ents[0]  # best candidate (already thresholded)
            concept = linker.kb.cui_to_entity[cui]
            for idx in span_to_rows.get((ent.start_char, ent.end_char), []):
                df.at[idx, "cui"] = cui
                df.at[idx, "canonical_name"] = concept.canonical_name
                df.at[idx, "umls_score"] = round(float(score), 4)

    linked = int((df["cui"].astype(str) != "").sum())
    print(f"[UMLS] {linked}/{len(df)} entities linked to a CUI "
          f"({100 * linked / len(df):.1f}%).")
    return df


def link(
    df_ner: pd.DataFrame,
    df_structured: pd.DataFrame,
    text_col: str = "text_en",
    backend: str = UMLS_LINKER,
) -> pd.DataFrame:
    """Append UMLS columns (LINKER_COLUMNS) to *df_ner* using *backend*."""
    if backend == "scispacy":
        return _link_scispacy(df_ner, df_structured, text_col)
    if backend == "medcat":
        raise NotImplementedError(
            "medcat backend not wired yet — the model pack is unavailable "
            "(KCL portal down). Use UMLS_LINKER='scispacy'."
        )
    raise ValueError(f"Unknown UMLS_LINKER backend: {backend!r}")
