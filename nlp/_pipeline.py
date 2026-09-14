"""Shared helpers for pipeline steps: dedup, CSV I/O, output schemas."""

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


NER_COLUMNS = [
    "id", "word", "label", "score", "text_en", "method",
    # char offsets (needed by the ConText assertion step) + assertion flags
    "start", "end",
    "is_negated", "is_uncertain", "is_family", "is_historical",
    # UMLS entity-linking (nlp/ner/linker.py): CUI + normalized name + linker
    # score. Blank if no UMLS match.
    "cui", "canonical_name", "umls_score",
]
RE_COLUMNS = [
    "id", "entity1", "entity1_label", "relation",
    "entity2", "entity2_label", "score", "sentence", "method",
    # per-entity assertion flags + relation-level status (graph edge polarity)
    "entity1_negated", "entity1_uncertain", "entity1_family", "entity1_historical",
    "entity2_negated", "entity2_uncertain", "entity2_family", "entity2_historical",
    "assertion",
    # per-entity UMLS linking (carried from NER) — node identity + type in the graph
    "entity1_cui", "entity1_canonical_name",
    "entity2_cui", "entity2_canonical_name",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV produced by any pipeline step."""
    return pd.read_csv(path)


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> Path:
    """Write *df* to *path* with QUOTE_ALL, restricted/ordered to *columns* if given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return path


def record_entities(df_ner: pd.DataFrame, record_id) -> list[dict]:
    """``{"word", "label"}`` dicts for one record, blank surface forms removed.

    The neural model occasionally spans a lone whitespace character, giving an
    empty ``word`` (read back from CSV as NaN). Such an entity is a substring of
    every sentence, so the RE methods would pair it with everything.
    """
    sub = df_ner[df_ner["id"] == record_id][["word", "label"]]
    sub = sub[sub["word"].notna() & (sub["word"].astype(str).str.strip() != "")]
    return sub.drop_duplicates().to_dict("records")


def dedup(
    df: pd.DataFrame,
    key_cols: list[str],
    method_priority: list[str],
    method_col: str = "method",
) -> pd.DataFrame:
    """
    Drop duplicates by *key_cols*, keeping rows whose method appears first in
    *method_priority*. Methods not in the list are ranked after listed ones.
    """
    if df.empty:
        return df.reset_index(drop=True)

    priority = {m: i for i, m in enumerate(method_priority)}
    fallback = len(method_priority)

    df = df.sort_values(
        method_col,
        key=lambda s: s.map(lambda v: priority.get(v, fallback)),
        kind="stable",
    )
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


# Assertion flags carried from NER onto relations, and the precedence used to
# collapse the two endpoints into one relation-level status (graph edge polarity).
# Order = precedence: negated > uncertain > family > historical > affirmed.
ASSERTION_FLAGS = ["is_negated", "is_uncertain", "is_family", "is_historical"]
_ASSERTION_STATUS = ["negated", "uncertain", "family", "historical"]


def annotate_assertion(
    df_rel: pd.DataFrame,
    df_ner: pd.DataFrame,
    sides: tuple[tuple[str, str], tuple[str, str]] = (
        ("entity1", "entity1_label"),
        ("entity2", "entity2_label"),
    ),
    id_col: str = "id",
) -> pd.DataFrame:
    """Attach per-entity assertion flags + a relation-level ``assertion`` status.

    Each relation endpoint is looked up in *df_ner* by ``(id, word.lower(), label)``
    and tagged with ``<entity>_negated/_family/_historical``. The relation-level
    ``assertion`` collapses both endpoints by precedence
    negated > family > historical > affirmed (status if EITHER endpoint has it).

    Lossless — no relation is dropped. Backward-compatible: if *df_ner* lacks the
    flag columns (e.g. NER run with --no-context), everything is False / affirmed.

    *sides* maps each endpoint to (entity_column, label_column) so the same logic
    serves both the biolink/spaCy schema (entity1_label) and the decoder schema
    (label1).
    """
    df_rel = df_rel.copy()
    out_cols = [f"{ent}_{f.replace('is_', '')}" for ent, _ in sides for f in ASSERTION_FLAGS]

    if df_rel.empty or not all(f in df_ner.columns for f in ASSERTION_FLAGS):
        for c in out_cols:
            df_rel[c] = False
        df_rel["assertion"] = "affirmed"
        return df_rel

    lookup: dict[tuple, dict] = {}
    for _, r in df_ner.iterrows():
        key = (r[id_col], str(r["word"]).lower(), r["label"])
        cur = lookup.setdefault(key, {f: False for f in ASSERTION_FLAGS})
        for f in ASSERTION_FLAGS:
            cur[f] = cur[f] or bool(r.get(f, False))

    def flag(rec_id, word, label, f):
        return lookup.get((rec_id, str(word).lower(), label), {}).get(f, False)

    for ent, lab in sides:
        for f in ASSERTION_FLAGS:
            col = f"{ent}_{f.replace('is_', '')}"
            df_rel[col] = df_rel.apply(
                lambda x: flag(x[id_col], x[ent], x[lab], f), axis=1
            )

    (ent1, _), (ent2, _) = sides

    def relation_assertion(row) -> str:
        # Assertion axes are orthogonal (negated/uncertain/family/historical), so
        # collect ALL that apply rather than only the top-precedence one. Joined in
        # precedence order -> e.g. "negated+family" for "no family history of X".
        statuses = [
            status
            for f, status in zip(ASSERTION_FLAGS, _ASSERTION_STATUS)
            if row[f"{ent1}_{f.replace('is_', '')}"] or row[f"{ent2}_{f.replace('is_', '')}"]
        ]
        return "+".join(statuses) if statuses else "affirmed"

    df_rel["assertion"] = df_rel.apply(relation_assertion, axis=1)
    return df_rel


# UMLS fields carried from NER onto each relation endpoint (node identity).
UMLS_FIELDS = ["cui", "canonical_name"]


def annotate_umls(
    df_rel: pd.DataFrame,
    df_ner: pd.DataFrame,
    sides: tuple[tuple[str, str], tuple[str, str]] = (
        ("entity1", "entity1_label"),
        ("entity2", "entity2_label"),
    ),
    id_col: str = "id",
) -> pd.DataFrame:
    """Attach per-endpoint UMLS columns (``<entity>_cui`` / ``<entity>_canonical_name``).

    Each relation endpoint is looked up in *df_ner* by ``(id, word.lower(), label)``
    — the same key as :func:`annotate_assertion` — so the CUI the linker assigned
    in NER rides through to the graph. Missing/unlinked -> "".

    Lets the visualizer merge graph nodes by CUI (synonyms collapse) instead of by
    raw string. Backward-compatible: if *df_ner* has no ``cui`` column, all "".
    """
    df_rel = df_rel.copy()
    out_cols = [f"{ent}_{f}" for ent, _ in sides for f in UMLS_FIELDS]

    if df_rel.empty or "cui" not in df_ner.columns:
        for c in out_cols:
            df_rel[c] = ""
        return df_rel

    lookup: dict[tuple, dict] = {}
    for _, r in df_ner.iterrows():
        key = (r[id_col], str(r["word"]).lower(), r["label"])
        # first non-empty wins (dedup keeps the higher-priority method's row)
        if key not in lookup:
            lookup[key] = {f: ("" if pd.isna(r.get(f)) else r.get(f, "")) for f in UMLS_FIELDS}

    def field(rec_id, word, label, f):
        return lookup.get((rec_id, str(word).lower(), label), {}).get(f, "")

    for ent, lab in sides:
        for f in UMLS_FIELDS:
            df_rel[f"{ent}_{f}"] = df_rel.apply(
                lambda x: field(x[id_col], x[ent], x[lab], f), axis=1
            )
    return df_rel
