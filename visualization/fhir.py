"""
FHIR-reification graph builder (GRAPH_MODE="fhir").

Turns NER + RE output into a FHIR-shaped Labeled Property Graph:
  - each entity MENTION -> a reified fact resource node (Condition / Observation /
    Procedure / MedicationStatement / FamilyMemberHistory), keyed per occurrence
    (`id:start:end`), so repeated concepts are NOT conflated (docs Q10),
  - the shared CUI stays a :Concept (CodeableConcept) node -> cross-patient huby,
  - provenance rides as `source_id` + char offsets on the fact (no Sentence node):
    the sentence text lives in the source CSVs and is found back via source_id,
  - RE relations are routed (config.RELATION_FHIR_MAP) to fact<->fact edges or to
    attribute fields; attribute values, which never become facts, stay a :Concept
    anchored to their patient by [:MENTIONS] so nothing is lost.

See docs/PatientCentrism.md Część III/IV and the plan.
"""

import pandas as pd
from pathlib import Path

from nlp._pipeline import ASSERTION_FLAGS, _ASSERTION_STATUS
from .pipeline import _node_key, _display_name, load_relations
from .config import (
    NER_OUTPUT_CSV, RE_OUTPUT_CSV,
    FHIR_PATIENTS_CSV, FHIR_CONCEPTS_CSV, FHIR_FACTS_CSV, FHIR_EDGES_CSV,
)
from config import (
    DEMOGRAPHIC_TYPES, FHIR_RESOURCE_MAP,
    RELATION_FHIR_MAP, ATTRIBUTE_FIELDS, RELATION_DIRECTION,
)

_DEMO_PROPS = sorted(label.lower() for label in DEMOGRAPHIC_TYPES)

# MACCROBAT captures a whole family-history clause ("mother had breast cancer")
# as ONE entity of this type, so the inner disease never surfaces as its own
# DISEASE_DISORDER and ConText's family cue (inside the span) never fires. We
# therefore reify these directly as :FamilyMemberHistory facts (see _build_facts).
_FAMILY_HISTORY_TYPES = {"FAMILY_HISTORY"}

# Family-relation terms scanned out of a FAMILY_HISTORY span → the `relative`
# property. Longer/compound terms first so "grandmother" is not read as "mother".
_RELATIVES = (
    "grandmother", "grandfather", "grandparent", "mother", "father", "parent",
    "sister", "brother", "sibling", "son", "daughter", "aunt", "uncle", "cousin",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _relative_of(word: str) -> str:
    """First family-relation term found in a FAMILY_HISTORY span ('' if none)."""
    w = str(word).lower()
    for rel in _RELATIVES:
        if rel in w:
            return rel
    return ""

def _is_true(value) -> bool:
    """Truthy check tolerant of CSV round-tripping (bool True or the string 'True')."""
    return value is True or str(value).strip().lower() == "true"


def _entity_assertion(row) -> str:
    """Collapse one entity's four assertion flags into a status string.

    Same precedence/composite rule as ``relation_assertion`` in nlp/_pipeline.py,
    for a single entity: every axis that fires is joined in precedence order
    (e.g. "negated+family"); no flag -> "affirmed".
    """
    statuses = [
        status
        for f, status in zip(ASSERTION_FLAGS, _ASSERTION_STATUS)
        if _is_true(row.get(f, False))
    ]
    return "+".join(statuses) if statuses else "affirmed"


def _clean(value) -> str:
    """NaN/None -> '' , else stripped string."""
    return "" if (value is None or pd.isna(value)) else str(value).strip()


def _offset(value) -> str:
    """Char offset as int-string, or '' if missing (keeps CSV/Neo4j happy)."""
    return "" if (value is None or pd.isna(value)) else str(int(value))


def _fact_id(rec_id: str, start, end, idx: int) -> str:
    """Per-occurrence reification key. Falls back to the row index if offsets are absent."""
    s, e = _offset(start), _offset(end)
    return f"{rec_id}:{s}:{e}" if (s and e) else f"{rec_id}:na:{idx}"


# The family override rewrites a fact into FamilyMemberHistory, but only a statement
# ABOUT SOMEONE can be a relative's history. A body part or a place of care cannot,
# so those resource types are immune to it.
_NOT_ABOUT_A_PERSON = {"BodyStructure", "Location"}


def _resource_for(label: str, assertion: str) -> str:
    """FHIR resource type (node label). Family-flagged entity -> FamilyMemberHistory."""
    resource = FHIR_RESOURCE_MAP.get(label, "Observation")
    if "family" in assertion and resource not in _NOT_ABOUT_A_PERSON:
        return "FamilyMemberHistory"
    return resource


def _verification_status(assertion: str) -> str:
    """assertion -> FHIR Condition.verificationStatus."""
    if "negated" in assertion:
        return "refuted"
    if "uncertain" in assertion:
        return "provisional"
    return "confirmed"


def _clinical_status(assertion: str) -> str:
    """assertion -> FHIR Condition.clinicalStatus (historical -> inactive)."""
    return "inactive" if "historical" in assertion else "active"


def _canonical_relation(rel: str) -> str:
    """Merge a deprecated duplicate label to its canonical one (else itself)."""
    return RELATION_FHIR_MAP.get(rel, {}).get("canonical", rel)


def _is_converse(relation: str) -> bool:
    """True if the label is its canonical relation stated backwards.

    Needed where :func:`_is_inverted` cannot help: it decides by resource type, so
    it is blind to a pair whose endpoints are the same type (CONTAINS runs
    BodyStructure -> BodyStructure).
    """
    return bool(RELATION_FHIR_MAP.get(relation, {}).get("inverse"))


def _is_inverted(relation: str, label1: str, label2: str) -> bool:
    """True if the RE row runs against the relation's canonical direction.

    Only fires when the object endpoint belongs on the subject side and the
    subject endpoint does not, so a genuinely ambiguous pair (both endpoints
    Procedures) is left exactly as extracted.
    """
    subjects = RELATION_DIRECTION.get(relation)
    if not subjects:
        return False
    res1, res2 = FHIR_RESOURCE_MAP.get(label1), FHIR_RESOURCE_MAP.get(label2)
    return res2 in subjects and res1 not in subjects


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_fhir_layer(
    ner_csv: str = NER_OUTPUT_CSV,
    re_csv: str = RE_OUTPUT_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (patients, concepts, facts, edges) frames for the FHIR graph."""
    ner = pd.read_csv(ner_csv)
    rel = load_relations(re_csv)  # normalized: entity1/label1/entity2/label2/relation/…

    patients_df = _build_patients(ner)
    facts, fact_by_id, fact_lookup = _build_facts(ner)
    concepts_df, context_rows = _build_concepts(ner)
    edges = _build_edges(facts, fact_by_id, fact_lookup, context_rows, rel)

    facts_df = pd.DataFrame(facts, columns=(
        ["key", "resource", "word", "name", "source_id", "start", "end",
         "cui", "code_key", "assertion", "verificationStatus", "clinicalStatus"]
        + ATTRIBUTE_FIELDS + ["relative"]
    ))
    edges_df = pd.DataFrame(edges, columns=(
        [":TYPE", "start_key", "end_key", "relation_score", "assertion",
         "source_id", "start", "end"]
    )).drop_duplicates(subset=[":TYPE", "start_key", "end_key"], keep="first").reset_index(drop=True)

    return patients_df, concepts_df, facts_df, edges_df


def _build_patients(ner: pd.DataFrame) -> pd.DataFrame:
    """One :Patient per record id; demographics (first value) as properties."""
    rows = []
    for pid, grp in ner.groupby("id"):
        row = {"key": str(pid), "id": str(pid)}
        for prop in _DEMO_PROPS:
            row[prop] = ""
        demo = grp[grp["label"].isin(DEMOGRAPHIC_TYPES)]
        for _, ent in demo.iterrows():
            prop = str(ent["label"]).lower()
            if prop in row and not row[prop]:  # first value wins
                row[prop] = str(ent["word"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["key", "id", *_DEMO_PROPS])


def _build_facts(ner: pd.DataFrame):
    """Reify every fact-producing mention. Returns (facts, fact_by_id, fact_lookup)."""
    facts: list[dict] = []
    fact_by_id: dict[str, dict] = {}
    # (id, word.lower(), label) -> [fact_id, ...]  (RE endpoints resolve through this)
    fact_lookup: dict[tuple, list] = {}

    for idx, r in ner.iterrows():
        label = str(r["label"])
        is_family_ent = label in _FAMILY_HISTORY_TYPES
        if label not in FHIR_RESOURCE_MAP and not is_family_ent:
            continue
        rec_id = str(r["id"])
        assertion = _entity_assertion(r)
        if is_family_ent and "family" not in assertion:
            # Force the family axis: the cue sits inside the span so ConText
            # missed it. The relative DID have it, so verificationStatus stays
            # confirmed (unless separately negated: "no family history of ...").
            assertion = "family" if assertion == "affirmed" else f"{assertion}+family"
        fid = _fact_id(rec_id, r.get("start"), r.get("end"), idx)
        fact = {
            "key": fid,
            "resource": _resource_for(label, assertion),
            "word": str(r["word"]),
            "name": _display_name(r["word"], r.get("cui"), r.get("canonical_name")),
            "source_id": rec_id,
            "start": _offset(r.get("start")),
            "end": _offset(r.get("end")),
            "cui": _clean(r.get("cui")),
            "code_key": _node_key(r["word"], r.get("cui")),
            "assertion": assertion,
            "verificationStatus": _verification_status(assertion),
            "clinicalStatus": _clinical_status(assertion),
        }
        for f in ATTRIBUTE_FIELDS:
            fact[f] = ""
        fact["relative"] = _relative_of(r["word"]) if is_family_ent else ""
        facts.append(fact)
        fact_by_id[fid] = fact
        fact_lookup.setdefault((rec_id, str(r["word"]).lower(), label), []).append(fid)

    return facts, fact_by_id, fact_lookup


def _build_concepts(ner: pd.DataFrame):
    """Shared :Concept nodes (codes for every non-demographic mention), merged by CUI-or-string.

    Returns (concepts_df, context_rows) where context_rows are the mentions that
    never become facts (attribute values) and so need the [:MENTIONS] backstop.
    """
    src = ner[~ner["label"].isin(DEMOGRAPHIC_TYPES) & (ner["label"] != "SUBJECT")].copy()
    src["key"] = [_node_key(w, c) for w, c in zip(src["word"], src["cui"])]
    src["cname"] = [_display_name(w, c, cn)
                    for w, c, cn in zip(src["word"], src["cui"], src["canonical_name"])]

    concepts_df = (
        src.groupby("key")
        .agg(
            name=("cname", "first"),
            entity_type=("label", "first"),
            cui=("cui", "first"),
            count=("key", "size"),
        )
        .reset_index()
    )
    concepts_df["cui"] = concepts_df["cui"].fillna("").astype(str)

    # Mentions that never become facts (attribute values) — the MENTIONS backstop.
    # FAMILY_HISTORY is reified as a fact, so keep it out too.
    fact_labels = set(FHIR_RESOURCE_MAP) | _FAMILY_HISTORY_TYPES
    context_rows = src[~src["label"].isin(fact_labels)].copy()
    return concepts_df, context_rows


def _build_edges(facts, fact_by_id, fact_lookup, context_rows, rel):
    """subject + has_code (from facts), then RE-routed edges, then MENTIONS backstop."""
    edges: list[dict] = []

    def emit(typ, start_key, end_key, score="", assertion="", source_id="", start="", end=""):
        edges.append({
            ":TYPE": typ, "start_key": start_key, "end_key": end_key,
            "relation_score": score, "assertion": assertion,
            "source_id": source_id, "start": start, "end": end,
        })

    # Provenance + code + subject, straight from the reified facts.
    for f in facts:
        emit("SUBJECT", f["key"], f["source_id"], source_id=f["source_id"],
             start=f["start"], end=f["end"])
        emit("HAS_CODE", f["key"], f["code_key"], source_id=f["source_id"])

    stats = {"clinical": 0, "attribute_field": 0,
             "demographic": 0, "dissolved": 0, "dropped_unresolved": 0,
             "mentions_backstop": 0, "flipped": 0}

    for _, r in rel.iterrows():
        rec_id = str(r["id"])
        relation = str(r["relation"])
        e1, l1 = str(r["entity1"]), str(r["label1"])
        e2, l2 = str(r["entity2"]), str(r["label2"])
        score = _clean(r.get("relation_score"))
        assertion = _clean(r.get("assertion")) or "affirmed"
        role = RELATION_FHIR_MAP.get(relation, {"role": "clinical"}).get("role", "clinical")

        f1 = fact_lookup.get((rec_id, e1.lower(), l1))  # subject/host fact(s)

        if role == "attribute":  # rule 2: attribute -> property FIELD only (policy A)
            # Which end hosts the field is decided per row, not per label. The
            # ontology deliberately declares both argument orders for some of these
            # ("Reverse: disease/symptom -> duration (passive constructions)"), and
            # names three of them value-first: QUANTIFIES/DURATION_OF/DESCRIBES read
            # "measure quantifies finding", "duration of event", "description
            # describes event". A per-label flag would fix one order and break the
            # other. Deciding by "which end is a fact" is not a guess: attribute
            # value types never become facts, so exactly one end qualifies in 1032
            # of 1065 rows. The 33 ambiguous ones are LAB_VALUE (both a fact type
            # and a measure) and keep the extracted order.
            f2 = fact_lookup.get((rec_id, e2.lower(), l2))
            # One label breaks the rule above: LAB_VALUE is both a fact type and a
            # measurement, so when it meets a finding BOTH ends are facts and being a
            # fact no longer identifies the host. The ontology offers QUANTIFIES in
            # either order, and left alone the subject wins, which writes the finding's
            # name into the measurement ("1.8-2 m/s".value = "acceleration of flow").
            # The label's own definition settles it — "measure quantifies finding" —
            # so the measurement is the value whichever side it was written on.
            if f1 and f2 and l1 == "LAB_VALUE":
                host, value = f2, e1
            elif f1:
                host, value = f1, e2
            elif f2:
                host, value = f2, e1
            else:
                stats["dropped_unresolved"] += 1
                continue
            field = RELATION_FHIR_MAP[relation].get("field")
            # VERBATIM, not the UMLS canonical name: this is FHIR CodeableConcept.text,
            # the phrase as dictated. Canonicalising it destroys the content the field
            # exists for — "1 mg" links to C0024467 and would be stored as "magnesium",
            # "150 mg" as "One Hundred Fifty", "500 mg" and "500 ml" both as "500",
            # "not increased" as "Increase". The numeric+unit split in
            # neo4j_integration._set_numeric then finds no digits and drops the property.
            # The coded form is NOT lost: the mention stays a :Concept with its CUI,
            # reachable via the MENTIONS backstop described below.
            for fid in host:
                if field and not fact_by_id[fid].get(field):
                    fact_by_id[fid][field] = value  # first value wins
            stats["attribute_field"] += 1
            # No edge: FHIR treats an attribute as a
            # coded FIELD on the resource (Condition.severity, dosage, ...), not a
            # shared node/edge (docs Kubełek 3). The value mention still falls through
            # to the (:Patient)-[:MENTIONS]->(:Concept) backstop, so it stays reachable
            # and traceable to its sentence — nothing is lost.
            continue

        if role == "clinical":  # rule 2: fact<->fact edge (anatomical placement included)
            f2 = fact_lookup.get((rec_id, e2.lower(), l2))
            if f1 and f2:
                typ = _canonical_relation(relation)
                subj, obj = f1, f2
                # Two repairs for a row written backwards, disjoint by construction:
                # `inverse` covers a label whose endpoints share a resource type, so
                # RELATION_DIRECTION cannot tell them apart (CONTAINS); the latter
                # covers the rest (IS_VALUE_OF, and DIAGNOSES/TREATS as extracted).
                if _is_converse(relation) or _is_inverted(typ, l1, l2):
                    subj, obj = f2, f1
                    stats["flipped"] += 1
                for a in subj:
                    for b in obj:
                        emit(typ, a, b, score, assertion, rec_id)
                stats["clinical"] += 1
            else:
                stats["dropped_unresolved"] += 1
            continue
        # rule 4: demographic / dissolved / unresolved endpoint -> no edge
        if role == "demographic":
            stats["demographic"] += 1
        elif role == "dissolved":
            stats["dissolved"] += 1
        else:
            stats["dropped_unresolved"] += 1

    # Backstop: attribute mentions never become facts (their value lives in a field
    # on the host), so they reach the graph only as (:Patient)-[:MENTIONS]->(:Concept).
    # Nothing is lost.
    for _, r in context_rows.iterrows():
        emit("MENTIONS", str(r["id"]), r["key"], assertion=_entity_assertion(r),
             source_id=str(r["id"]), start=_offset(r.get("start")), end=_offset(r.get("end")))
        stats["mentions_backstop"] += 1

    print(
        f"[fhir] RE->graf: {len(rel)} relacji | "
        f"clinical={stats['clinical']} "
        f"attribute={stats['attribute_field']} demographic={stats['demographic']} "
        f"dissolved={stats['dissolved']} odrzucone(brak faktu)={stats['dropped_unresolved']}"
    )
    print(f"[fhir] odwrócony kierunek (kanonizacja TREATS/DIAGNOSES/MEASURES/LOCATED_IN): {stats['flipped']}")
    print(f"[fhir] encje tylko przez MENTIONS (niepodpięte relacją): {stats['mentions_backstop']}")

    return edges


# ---------------------------------------------------------------------------
# Save / run
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, path: str, what: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=1)  # QUOTE_ALL
    print(f"[fhir] Saved {len(df)} {what} to {path}")


def run_fhir_pipeline(
    ner_csv: str = NER_OUTPUT_CSV,
    re_csv: str = RE_OUTPUT_CSV,
    patients_csv: str = FHIR_PATIENTS_CSV,
    concepts_csv: str = FHIR_CONCEPTS_CSV,
    facts_csv: str = FHIR_FACTS_CSV,
    edges_csv: str = FHIR_EDGES_CSV,
) -> None:
    """Build the FHIR layer and write the four CSVs."""
    patients_df, concepts_df, facts_df, edges_df = build_fhir_layer(ner_csv, re_csv)
    _save(patients_df, patients_csv, "patients")
    _save(concepts_df, concepts_csv, "concepts")
    _save(facts_df, facts_csv, "facts")
    _save(edges_df, edges_csv, "edges")


if __name__ == "__main__":
    run_fhir_pipeline()
