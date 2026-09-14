"""
spaCy rule-based relation extraction.

Two complementary strategies are applied to each sentence:

1. Lexical patterns  – search for fixed trigger phrases (e.g. "caused by",
   "associated with") and identify the nearest named entities on each side.
   Patterns are split into FORWARD and INVERTED to handle directionality:
     FORWARD  – agent/cause is BEFORE the trigger  ("X causes Y")
     INVERTED – agent/cause is AFTER  the trigger  ("X due to Y" → Y causes X)

2. Dependency patterns – walk the dependency tree looking for predicate verbs
   (cause, treat, diagnose, …) and link their syntactic subject/object to
   extracted entities.

Results from both strategies are combined; the caller (re_tasks.py) is
responsible for merging these with BioLinkBERT results.
"""

from functools import lru_cache

import pandas as pd
import spacy
from tqdm import tqdm

from config import SPACY_MODEL_ENGLISH
from nlp._pipeline import record_entities
from nlp.re.ontology import (
    ENTITY_PAIR_RELATIONS,
    LEXICAL_PATTERNS_FORWARD,
    LEXICAL_PATTERNS_INVERTED,
    ONT,
    VERB_RELATION_MAP,
)


@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load(SPACY_MODEL_ENGLISH)

# Dependency relations considered as syntactic subject / object
_SUBJ_DEPS = {"nsubj", "nsubjpass"}
_OBJ_DEPS  = {"dobj", "attr", "pobj", "oprd"}

# Adjectival / nominal modifier deps: a descriptor attaches to the noun it modifies.
# Used to extract attribute-attachment relations from SYNTAX (exact attachment)
# instead of semantic similarity, which cannot tell which noun a descriptor belongs to.
_MODIFIER_DEPS = {"amod", "compound"}

# Attribute-attachment relations: a modifier grammatically attached to a noun is
# an attribute of that noun, and the relation TYPE is fixed by the entity types
# (severity -> HAS_SEVERITY, colour -> HAS_COLOR, ...). The parse gives exact
# attachment, so these are extracted here and skipped by biolink (see biolink.py
# _SYNTAX_ONLY_RELATIONS). Direction differs per relation: DESCRIBES points
# modifier -> head ("chronic" describes "lesions"), whereas HAS_* points
# head -> modifier ("hypothyroidism" HAS_SEVERITY "severe") — the ontology encodes
# which, so _extract_syntactic tries both orderings and emits whichever it allows.
_ATTACHMENT_RELATIONS = {
    ONT.DESCRIBES,
    ONT.HAS_SEVERITY,
    ONT.HAS_COLOR,
    ONT.HAS_SHAPE,
    ONT.HAS_TEXTURE,
}

# Predicative deps: an adjectival/nominal predicate linked to the subject through a
# copula or linking verb. Copulas ("the lesion was red") parse the predicate as
# `acomp`/`attr`; raising verbs ("the nodule appears/seems hypoechoic") parse it as
# `oprd`. Either way the predicate is an attribute of the verb's subject, so it
# feeds the same _ATTACHMENT_RELATIONS logic.
_PREDICATE_DEPS = {"acomp", "attr", "oprd"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sent_entities(sent_text: str, entities: list[dict]) -> list[dict]:
    """Return entities whose surface form appears in *sent_text* (case-insensitive)."""
    return [e for e in entities if e["word"].lower() in sent_text.lower()]


def _emit(e1: dict, relation: str, e2: dict, sent_text: str, trigger: str, method: str) -> dict:
    """Build a result row dict."""
    return {
        "entity1":       e1["word"],
        "entity1_label": e1["label"],
        "relation":      relation,
        "entity2":       e2["word"],
        "entity2_label": e2["label"],
        "score":         1.0,
        "sentence":      sent_text,
        "method":        method,
        "trigger":       trigger,
    }


def _extract_lexical(doc, entities: list[dict]) -> list[dict]:
    """
    Lexical trigger-phrase extraction.

    FORWARD  patterns: entity_before --relation--> entity_after
    INVERTED patterns: entity_after  --relation--> entity_before
    """
    results: list[dict] = []

    for sent in doc.sents:
        sent_text = sent.text
        sent_ents = _sent_entities(sent_text, entities)
        if len(sent_ents) < 2:
            continue

        # FORWARD: agent/cause is before the trigger
        for relation, triggers in LEXICAL_PATTERNS_FORWARD.items():
            for trigger in triggers:
                pos = sent_text.lower().find(trigger.lower())
                if pos == -1:
                    continue
                before = sent_text[:pos].lower()
                after  = sent_text[pos + len(trigger):].lower()
                ents_before = [e for e in sent_ents if e["word"].lower() in before]
                ents_after  = [e for e in sent_ents if e["word"].lower() in after]
                for e1 in ents_before:
                    for e2 in ents_after:
                        if e1["word"] == e2["word"]:
                            continue
                        if relation not in ENTITY_PAIR_RELATIONS.get((e1["label"], e2["label"]), []):
                            continue
                        results.append(_emit(e1, relation, e2, sent_text, trigger, "spacy_lexical"))

        # INVERTED: agent/cause is after the trigger → swap entity1/entity2
        for relation, triggers in LEXICAL_PATTERNS_INVERTED.items():
            for trigger in triggers:
                pos = sent_text.lower().find(trigger.lower())
                if pos == -1:
                    continue
                before = sent_text[:pos].lower()
                after  = sent_text[pos + len(trigger):].lower()
                ents_before = [e for e in sent_ents if e["word"].lower() in before]
                ents_after  = [e for e in sent_ents if e["word"].lower() in after]
                # Inverted: after-entity is the agent → e1=after, e2=before
                for e1 in ents_after:
                    for e2 in ents_before:
                        if e1["word"] == e2["word"]:
                            continue
                        if relation not in ENTITY_PAIR_RELATIONS.get((e1["label"], e2["label"]), []):
                            continue
                        results.append(_emit(e1, relation, e2, sent_text, trigger, "spacy_lexical"))

    return results


def _noun_chunk_text(token, doc) -> str:
    """Return the noun-chunk text that contains *token*, or the token text."""
    for chunk in doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token.text


def _entity_for_token(token, sent_ents: list[dict], doc) -> dict | None:
    """Map a parse token to the NER entity it belongs to (by surface/noun-chunk)."""
    tl = token.text.lower()
    for e in sent_ents:                      # exact single-token match first
        if e["word"].lower() == tl:
            return e
    chunk = _noun_chunk_text(token, doc).lower()
    for e in sent_ents:                      # else overlap with the token's noun chunk
        w = e["word"].lower()
        if tl in w.split() or w == chunk or w in chunk or chunk in w:
            return e
    return None


def _emit_attachment(e_a: dict, e_b: dict, sent_text: str, trigger: str, results: list[dict]) -> None:
    """Emit any _ATTACHMENT_RELATIONS the ontology allows between e_a and e_b.

    Tries both orderings, because direction depends on the relation: DESCRIBES
    points modifier -> head, whereas HAS_* points head -> modifier. The ontology
    encodes which pair is legal, so only the correct direction fires.
    """
    for subj, obj in ((e_a, e_b), (e_b, e_a)):
        for r in ENTITY_PAIR_RELATIONS.get((subj["label"], obj["label"]), []):
            if r in _ATTACHMENT_RELATIONS:
                results.append(_emit(subj, r, obj, sent_text, trigger, "spacy_dependency"))


def _extract_syntactic(doc, entities: list[dict]) -> list[dict]:
    """Attribute attachment from the dependency tree.

    Two grammatical patterns feed the same _ATTACHMENT_RELATIONS logic:
      1. direct modifiers (amod/compound) — "chronic lesions", "red lesion";
      2. predicative complements (acomp/attr via a copula) — "the lesion was red",
         "the nodule appears hypoechoic".
    Both give exact attachment (the parse says which noun the descriptor belongs
    to), so these relations are extracted ONLY here; biolink skips them, because
    direction-blind similarity would attach a descriptor to every entity.
    """
    results: list[dict] = []
    for sent in doc.sents:
        sent_text = sent.text
        sent_ents = _sent_entities(sent_text, entities)
        if len(sent_ents) < 2:
            continue
        for token in sent:
            # (1) direct modifier: descriptor attaches to its head noun.
            if token.dep_ in _MODIFIER_DEPS:
                e_head = _entity_for_token(token.head, sent_ents, doc)
                if not e_head:
                    continue
                # The modifier + any coordinated modifiers, so "chronic and
                # inflammatory lesions" attaches BOTH to "lesions" (spaCy hangs the
                # 2nd adjective off the 1st via `conj`, not off the noun).
                for mod in (token, *token.conjuncts):
                    e_mod = _entity_for_token(mod, sent_ents, doc)
                    if e_mod and e_mod["word"] != e_head["word"]:
                        _emit_attachment(e_mod, e_head, sent_text, token.dep_, results)

            # (2) predicative complement: the subject noun and the adjectival
            # predicate share the copula/linking verb, so the predicate is an
            # attribute of the subject ("the lesion was red").
            elif token.dep_ in _PREDICATE_DEPS:
                subjects = [c for c in token.head.children if c.dep_ in _SUBJ_DEPS]
                for pred in (token, *token.conjuncts):
                    e_pred = _entity_for_token(pred, sent_ents, doc)
                    if not e_pred:
                        continue
                    for subj in subjects:
                        e_subj = _entity_for_token(subj, sent_ents, doc)
                        if e_subj and e_subj["word"] != e_pred["word"]:
                            _emit_attachment(e_subj, e_pred, sent_text, token.dep_, results)
    return results


def _with_conjuncts(tokens: list) -> list:
    """Expand tokens with their coordinated siblings (`conj`), de-duplicated.

    So "X and Y cause Z" links BOTH X->Z and Y->Z, and "affects the sinuses and
    the nose" links both objects — spaCy hangs later conjuncts off the first, not
    off the verb.
    """
    out, seen = [], set()
    for t in tokens:
        for x in (t, *t.conjuncts):
            if x.i not in seen:
                seen.add(x.i)
                out.append(x)
    return out


def _extract_dependency(doc, entities: list[dict]) -> list[dict]:
    """Dependency-tree verb-pattern extraction."""
    results: list[dict] = []

    for sent in doc.sents:
        sent_text = sent.text
        sent_ents = _sent_entities(sent_text, entities)
        if len(sent_ents) < 2:
            continue

        for token in sent:
            rel = VERB_RELATION_MAP.get(token.lemma_.lower())
            if rel is None:
                continue

            # Follow coordination so all conjoined subjects/objects are linked.
            subjects = _with_conjuncts([c for c in token.children if c.dep_ in _SUBJ_DEPS])
            objects  = _with_conjuncts([c for c in token.children if c.dep_ in _OBJ_DEPS])

            for subj in subjects:
                subj_span = _noun_chunk_text(subj, doc).lower()
                for obj in objects:
                    obj_span = _noun_chunk_text(obj, doc).lower()

                    e1_matches = [
                        e for e in sent_ents
                        if e["word"].lower() in subj_span
                        or subj_span in e["word"].lower()
                    ]
                    e2_matches = [
                        e for e in sent_ents
                        if e["word"].lower() in obj_span
                        or obj_span in e["word"].lower()
                    ]

                    for e1 in e1_matches:
                        for e2 in e2_matches:
                            if e1["word"] == e2["word"]:
                                continue
                            if rel not in ENTITY_PAIR_RELATIONS.get((e1["label"], e2["label"]), []):
                                continue
                            results.append(
                                _emit(e1, rel, e2, sent_text, token.text, "spacy_dependency")
                            )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(df_ner: pd.DataFrame, df_structured: pd.DataFrame) -> pd.DataFrame:
    """
    Run spaCy rule-based relation extraction.

    Parameters
    ----------
    df_ner        : DataFrame with columns [id, word, label, score, text_en]
    df_structured : DataFrame with columns [id, text, text_en, ...]

    Returns
    -------
    DataFrame with columns:
        id, entity1, entity1_label, relation, entity2, entity2_label,
        score, sentence, method, trigger
    """
    nlp = _load_spacy()

    results: list[dict] = []
    id_col = df_structured.set_index("id")

    for record_id in tqdm(df_ner["id"].unique(), desc="spaCy-RE", unit="rec"):
        if record_id not in id_col.index:
            continue
        text = id_col.loc[record_id, "text_en"]
        if not isinstance(text, str) or not text.strip():
            continue

        entities = record_entities(df_ner, record_id)

        doc = nlp(text)

        for row in (_extract_lexical(doc, entities)
                    + _extract_dependency(doc, entities)
                    + _extract_syntactic(doc, entities)):
            row["id"] = record_id
            results.append(row)

    if not results:
        return pd.DataFrame(
            columns=[
                "id", "entity1", "entity1_label", "relation",
                "entity2", "entity2_label", "score", "sentence", "method", "trigger",
            ]
        )

    df = pd.DataFrame(results)
    print(f"[spaCy RE] Found {len(df)} relations.")
    return df
