"""
medSpaCy ConText assertion enrichment for NER entities.

Adds per-entity assertion flags (negation, uncertainty, family, historical) by
placing the *existing* NER entities back onto a spaCy Doc (by character offset)
and running medSpaCy's ConText algorithm. It extracts no new entities — only
annotates the ones the neural + gazetteer methods already found.

The data is thyroid-ultrasound radiology text: short findings, heavy negation
("no sign of", "without", "not seen"). Default ConText rules catch most pre-posed
negation; a few BACKWARD rules are added for the trailing negation common in
radiology ("X were not found", "X not seen").
"""

from functools import lru_cache

import pandas as pd
import spacy
import medspacy  # noqa: F401 — importing registers the medspacy_* factories
from tqdm import tqdm
from medspacy.context import ConText, ConTextRule

from config import SCISPACY_MODEL, CONTEXT_MAX_SCOPE

# Columns appended to the NER DataFrame (must match the flags in NER_COLUMNS).
ASSERTION_COLUMNS = ["is_negated", "is_uncertain", "is_family", "is_historical"]

# ConText Span extension attributes corresponding to each flag.
_ATTR_BY_COLUMN = {
    "is_negated": "is_negated",
    "is_uncertain": "is_uncertain",
    "is_family": "is_family",
    "is_historical": "is_historical",
}

# Trailing negation triggers (entity appears BEFORE the cue) — common in
# radiology reports, not covered by default ConText rules.
_BACKWARD_NEGATION = [
    "were not found", "not found", "not seen", "not visualized",
    "not visualised", "not identified", "not detected", "not observed",
    "not demonstrated", "not appreciated",
]

# Clause boundaries that TERMINATE a modifier's scope — stops flags leaking past
# a clause in run-on sentences ("denies fever, ECG showed..." -> ECG not negated).
# Deliberately NOT "and": that would break coordinations ("diabetes and hypertension").
_TERMINATORS = [";", ",", "but", "however", "except", "otherwise", "whereas"]


@lru_cache(maxsize=1)
def _load():
    """SciSpaCy parser (for sentence boundaries) + standalone ConText component.

    NER is disabled so the sci model's own entities don't overwrite the ones we
    inject from the NER pipeline. Scope is bounded two ways so assertion flags
    don't leak across a whole run-on sentence: a global max_scope (token distance)
    and clause-boundary TERMINATE rules.
    """
    nlp = spacy.load(SCISPACY_MODEL, disable=["ner"])
    context = ConText(nlp, rules="default", max_scope=CONTEXT_MAX_SCOPE)
    context.add(
        [ConTextRule(lit, "NEGATED_EXISTENCE", direction="BACKWARD")
         for lit in _BACKWARD_NEGATION]
        + [ConTextRule(sep, "TERMINATE", direction="TERMINATE")
           for sep in _TERMINATORS]
    )
    return nlp, context


def _span_for_row(doc, row, text: str):
    """Locate an entity on *doc*: by char offset if present, else by string search."""
    label = str(row.get("label", ""))
    start, end = row.get("start"), row.get("end")
    if pd.notna(start) and pd.notna(end):
        span = doc.char_span(int(start), int(end), label=label, alignment_mode="expand")
        if span is not None:
            return span
    word = str(row.get("word", "")).strip()
    if not word:
        return None
    pos = text.lower().find(word.lower())
    if pos == -1:
        return None
    return doc.char_span(pos, pos + len(word), label=label, alignment_mode="expand")


def enrich(
    df_ner: pd.DataFrame,
    df_structured: pd.DataFrame,
    text_col: str = "text_en",
) -> pd.DataFrame:
    """Append ConText assertion columns (ASSERTION_COLUMNS) to *df_ner*.

    Groups entities by record id, rebuilds the Doc from the record text, places
    the entities by char offset, runs ConText, and maps the assertion attributes
    back onto the rows. Entities that can't be located stay False.
    """
    df = df_ner.copy().reset_index(drop=True)
    for col in ASSERTION_COLUMNS:
        df[col] = False
    if df.empty:
        return df

    nlp, context = _load()
    text_by_id = df_structured.set_index("id")[text_col].to_dict()

    print(f"[CONTEXT] Annotating assertion on {len(df)} entities ...")

    for rec_id, grp in tqdm(df.groupby("id"), total=df["id"].nunique(),
                            desc="ConText", unit="rec"):
        text = text_by_id.get(rec_id)
        if not isinstance(text, str) or not text.strip():
            continue

        doc = nlp(text)
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

        doc.ents = spacy.util.filter_spans(spans)  # ConText needs non-overlapping ents
        context(doc)

        for ent in doc.ents:
            rows = span_to_rows.get((ent.start_char, ent.end_char), [])
            for idx in rows:
                for col, attr in _ATTR_BY_COLUMN.items():
                    df.at[idx, col] = bool(getattr(ent._, attr))

    negated = int(df["is_negated"].sum())
    print(f"[CONTEXT] {negated} entities flagged negated "
          f"({100 * negated / len(df):.1f}%).")
    return df
