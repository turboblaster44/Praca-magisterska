"""Holistic (joint) extraction variant: one generative step for entities + relations.

The modular pipeline runs NER methods and RE methods separately and cascades them.
This variant instead asks a single LLM, per batch of ``config.HOLISTIC_BATCH_SIZE``
records, to classify BOTH the entities
AND the relations in one pass, strictly constrained by the project's domain
knowledge: the MACCROBAT entity types (``ENT``) and the relation ontology's allowed
``(subject_type, relation, object_type)`` triples (``ENTITY_PAIR_RELATIONS``). It
REPLACES the modular extractors but reuses the exact same downstream: ConText
assertion, UMLS linking, and the FHIR graph build, so both variants are comparable.

``method`` is stamped ``"holistic"``. No API key -> empty frames (graceful no-op),
so the rest of the pipeline still runs. Selected via ``config.PIPELINE_MODE``.
"""

import json

import pandas as pd
from tqdm import tqdm

from config import (
    HOLISTIC_BATCH_SIZE, LLM_RE_SCORE_THRESHOLD, HOLISTIC_LLM_ASSERTIONS,
)
from nlp.llm_client import generate, reset_stats, stats
from nlp.ner import assertion, linker
from nlp.ner.ontology import ENT
from nlp.re.ontology import ENTITY_PAIR_RELATIONS
from nlp._pipeline import annotate_assertion, annotate_umls, dedup

_METHOD = "holistic"
_LABELS = set(ENT.all())
_NER_CORE = ["id", "word", "label", "score", "text_en", "method", "start", "end"]
# Assertion axes the model is asked for when HOLISTIC_LLM_ASSERTIONS is on. Same
# four columns ConText would otherwise fill, so everything downstream is unchanged.
_ASSERTION_KEYS = {
    "negated": "is_negated", "uncertain": "is_uncertain",
    "family": "is_family", "historical": "is_historical",
}
_RE_CORE = [
    "id", "entity1", "entity1_label", "relation",
    "entity2", "entity2_label", "score", "sentence", "method",
]


def _triples_block() -> str:
    """Allowed triples grouped by subject type: 'SUBJECT' + '  -> OBJECT: REL | REL'.

    Grouped rather than 140 flat 'SUBJ REL OBJ' lines so the model can look up an
    entity's type once and see every relation it can take.
    """
    by_subject: dict[str, list[str]] = {}
    for (t1, t2), rels in ENTITY_PAIR_RELATIONS.items():
        by_subject.setdefault(t1, []).append(f"  -> {t2}: {' | '.join(rels)}")
    return "\n".join(f"{t1}\n" + "\n".join(sorted(lines))
                     for t1, lines in sorted(by_subject.items()))


def _assertion_block() -> tuple[str, str, str, str]:
    """Prompt fragments for the four assertion axes; all empty when ConText owns them."""
    if not HOLISTIC_LLM_ASSERTIONS:
        return "", "", "", ""
    reguly = (
        "For EACH entity also decide four independent assertion flags, judged from the "
        "sentence as a whole, not from nearby words:\n"
        '  "negated"    — the sentence states this does NOT exist ("no nodules", '
        '"thyroid not enlarged"). A finding merely described as normal is NOT negated.\n'
        '  "uncertain"  — hedged or under consideration ("possible", "suspected", '
        '"cannot be excluded").\n'
        '  "family"     — it concerns a RELATIVE, not the patient. Applies to EVERY '
        'entity the family statement covers, including each member of a coordination '
        '("family history of breast cancer and ovarian cancer" -> both are family).\n'
        '  "historical" — a past episode rather than the current presentation. A '
        "long-standing condition still being managed is NOT historical.\n"
        "The flags are independent and may combine; default every one to false.\n\n"
    )
    przyklad = (
        ', "negated": false, "uncertain": false, "family": false, "historical": false'
    )
    schemat = (
        ', "negated": false, "uncertain": false, "family": false, "historical": false'
    )
    zasada = (
        " Decide the four assertion flags for every entity from the sentence's meaning;"
        " when a cue covers several coordinated entities, flag all of them."
    )
    return reguly, przyklad, schemat, zasada


def _build_prompt(batch: list[tuple[object, str]]) -> str:
    asercje_reguly, asercje_przyklad, asercje_schemat, asercje_zasada = _assertion_block()
    labels = ", ".join(ENT.all())
    records = json.dumps([{"id": str(rec_id), "sentence": text} for rec_id, text in batch],
                         ensure_ascii=False)
    return (
        "You are a clinical information extraction system. From EACH sentence below, "
        "extract BOTH the named entities AND the relations between them in a single "
        "pass, strictly following the ontology.\n\n"
        f"Records (JSON): {records}\n\n"
        f"Allowed ENTITY TYPES (use these exact labels only):\n{labels}\n\n"
        "Allowed RELATION TRIPLES (subject_type RELATION object_type). A relation is "
        "valid ONLY if its two entities' types and direction match one of these:\n"
        f"{_triples_block()}\n\n"
        "For EACH record work in two steps:\n"
        "  step 1: list every entity in that record's sentence.\n"
        "  step 2: take every ORDERED pair of the entities you just listed, look the "
        "pair's (subject_type, object_type) up in the table above, and emit a relation "
        "whenever the sentence supports one of the relations listed for that pair. Check "
        "both directions of a pair, they are different lookups. Report only step 2 "
        "findings, never the enumeration itself.\n"
        "Most clinical sentences carry at least one relation: a procedure and a finding "
        "in the same sentence are almost always related, so is a medication and the "
        "condition it is given for, and a finding and the anatomy it sits in. Leave "
        '"relations" empty only when no pair of the extracted entities is supported by '
        "both the sentence and the table.\n\n"
        f"{asercje_reguly}"
        "Example of one record object (illustrates the two steps, not a template for "
        'labels): sentence "Ultrasound showed a hypoechoic nodule in the right lobe, and '
        'levothyroxine was continued for hypothyroidism." ->\n'
        '{"id": "X", "entities": [{"text": "Ultrasound", "label": "DIAGNOSTIC_PROCEDURE", '
        f'"confidence": 0.98{asercje_przyklad}}}, '
        '{"text": "hypoechoic nodule", "label": "SIGN_SYMPTOM", '
        '"confidence": 0.95}, {"text": "right lobe", "label": "BIOLOGICAL_STRUCTURE", '
        '"confidence": 0.97}, {"text": "levothyroxine", "label": "MEDICATION", '
        '"confidence": 0.99}, {"text": "hypothyroidism", "label": "DISEASE_DISORDER", '
        '"confidence": 0.98}], "relations": [{"entity1": "Ultrasound", "entity2": '
        '"hypoechoic nodule", "relation": "REVEALS", "confidence": 0.95}, {"entity1": '
        '"hypoechoic nodule", "entity2": "right lobe", "relation": "LOCATED_IN", '
        '"confidence": 0.94}, {"entity1": "levothyroxine", "entity2": "hypothyroidism", '
        '"relation": "TREATS", "confidence": 0.96}]}\n\n'
        "Return ONLY a JSON object, no prose, of the exact shape:\n"
        '{"records": [{"id": "...", "entities": [{"text": "...", "label": "ENTITY_TYPE", '
        f'"confidence": 0.0{asercje_schemat}}}], '
        '"relations": [{"entity1": "...", "entity2": "...", '
        '"relation": "RELATION", "confidence": 0.0}]}]}\n'
        "Rules: return one object per input record, keeping its id unchanged, with both "
        'keys present even when empty; a record\'s entities and relations must come from '
        "that record's own sentence only; copy entity text exactly as it appears; use "
        "only labels from the list (skip anything that does not fit); entity1/entity2 "
        "must be the exact text of two extracted entities and the (entity1_type, "
        "entity2_type, relation) must match an allowed triple; confidence is your "
        f"certainty in [0,1].{asercje_zasada}"
    )


def _extract_json(text: str) -> dict | None:
    """Parse the model's JSON, tolerating code fences / surrounding prose."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _to_float(value) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _offsets(text: str, word: str):
    """First-occurrence char offsets (same lookup as assertion._span_for_row)."""
    pos = text.lower().find(word.lower())
    if pos == -1:
        return None, None
    return pos, pos + len(word)


def _parse_record(rec_id, text: str, payload: dict) -> tuple[list[dict], list[dict]]:
    """Validate one record's entities + relations against the ontology."""
    ent_rows: list[dict] = []
    label_by_text: dict[str, str] = {}  # entity text (lower) -> its accepted label

    for e in payload.get("entities") or []:
        if not isinstance(e, dict):
            continue
        word = str(e.get("text", "")).strip()
        label = str(e.get("label", "")).strip().upper()
        if not word or label not in _LABELS:
            continue
        start, end = _offsets(text, word)
        row = {
            "id": rec_id, "word": word, "label": label,
            "score": _to_float(e.get("confidence")), "text_en": text,
            "method": _METHOD, "start": start, "end": end,
        }
        if HOLISTIC_LLM_ASSERTIONS:
            # Missing key -> False, same default ConText would leave.
            row.update({col: bool(e.get(key)) for key, col in _ASSERTION_KEYS.items()})
        ent_rows.append(row)
        label_by_text.setdefault(word.lower(), label)

    rel_rows: list[dict] = []
    for r in payload.get("relations") or []:
        if not isinstance(r, dict):
            continue
        e1, e2 = str(r.get("entity1", "")).strip(), str(r.get("entity2", "")).strip()
        relation = str(r.get("relation", "")).strip().upper()
        l1, l2 = label_by_text.get(e1.lower()), label_by_text.get(e2.lower())
        if not (l1 and l2 and relation):
            continue
        if relation not in ENTITY_PAIR_RELATIONS.get((l1, l2), []):
            continue
        score = _to_float(r.get("confidence"))
        if score < LLM_RE_SCORE_THRESHOLD:
            continue
        rel_rows.append({
            "id": rec_id, "entity1": e1, "entity1_label": l1, "relation": relation,
            "entity2": e2, "entity2_label": l2, "score": score,
            "sentence": text, "method": _METHOD,
        })

    return ent_rows, rel_rows


def run(
    df_structured: pd.DataFrame,
    use_context: bool = True,
    use_umls: bool = True,
    lang_col: str = "text_en",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint entity+relation extraction via one LLM call per record.

    Returns (df_ner, df_rel) already enriched (ConText/UMLS) and in the canonical
    NER_COLUMNS / RE_COLUMNS schemas, ready to save and feed the FHIR graph.
    """
    print("=" * 60)
    zrodlo = "LLM" if HOLISTIC_LLM_ASSERTIONS else ("ConText" if use_context else "brak")
    print(f"HOLISTIC PIPELINE (asercje={zrodlo}, umls={use_umls}, "
          f"batch={HOLISTIC_BATCH_SIZE})")
    print("=" * 60)

    records = [(rec["id"], rec[lang_col]) for _, rec in df_structured.iterrows()
               if isinstance(rec.get(lang_col), str) and rec[lang_col].strip()]
    batches = [records[i:i + HOLISTIC_BATCH_SIZE]
               for i in range(0, len(records), HOLISTIC_BATCH_SIZE)]

    reset_stats()
    ent_rows: list[dict] = []
    rel_rows: list[dict] = []
    lost = 0  # records the model returned nothing usable for (a batch fails as a whole)
    for batch in tqdm(batches, desc="Holistic", unit="batch"):
        response, _ = generate(_build_prompt(batch))
        payload = _extract_json(response) if response else None
        by_id = {str(r.get("id")): r for r in (payload or {}).get("records") or []
                 if isinstance(r, dict)}
        for rec_id, text in batch:
            rec_payload = by_id.get(str(rec_id))
            if rec_payload is None:
                lost += 1
                continue
            ents, rels = _parse_record(rec_id, text, rec_payload)
            ent_rows.extend(ents)
            rel_rows.extend(rels)

    kolumny = _NER_CORE + (list(_ASSERTION_KEYS.values()) if HOLISTIC_LLM_ASSERTIONS else [])
    df_ner = pd.DataFrame(ent_rows, columns=kolumny)
    df_rel = pd.DataFrame(rel_rows, columns=_RE_CORE)

    if not df_ner.empty:
        keyed = df_ner.copy()
        keyed["_word_lower"] = keyed["word"].astype(str).str.lower()
        df_ner = dedup(keyed, key_cols=["id", "_word_lower", "label"],
                       method_priority=[_METHOD]).drop(columns="_word_lower")
    if not df_rel.empty:
        df_rel = dedup(df_rel, key_cols=["id", "entity1", "entity2", "relation"],
                       method_priority=[_METHOD])

    # Reuse the exact same enrichment as the modular pipeline. The one exception is
    # HOLISTIC_LLM_ASSERTIONS: the model already returned the four flags, so running
    # ConText on top would overwrite them with the rule engine's verdict.
    if use_context and not HOLISTIC_LLM_ASSERTIONS:
        df_ner = assertion.enrich(df_ner, df_structured, text_col=lang_col)
    if use_umls:
        df_ner = linker.link(df_ner, df_structured, text_col=lang_col)
    df_rel = annotate_assertion(df_rel, df_ner)
    df_rel = annotate_umls(df_rel, df_ner)

    print(f"[HOLISTIC] {len(df_ner)} entities, {len(df_rel)} relations.")
    s = stats()
    print(f"[HOLISTIC] {s['calls']} calls, {s['retries']} retries, {s['failed']} failed.")
    if lost:
        print(f"[HOLISTIC] {lost}/{len(records)} records got no usable response.")
    return df_ner, df_rel
