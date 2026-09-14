"""BioLinkBERT zero-shot relation extraction via entity markers + prototype embeddings."""

import re as _re
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from config import SPACY_MODEL_ENGLISH, RE_PAIRING, RE_BIOLINK_SCORE_THRESHOLD
from nlp._pipeline import record_entities
from nlp.re import candidates as candidate_gen
from nlp.re.ontology import ENTITY_PAIR_RELATIONS, RELATION_PROTOTYPES

SCORE_THRESHOLD = RE_BIOLINK_SCORE_THRESHOLD

# Relations that are attachment/modifier relations — extracted from the syntactic
# parse (spaCy amod/compound), NOT from semantic similarity. biolink would
# over-generate them (a descriptor pairs with every allowed entity in the
# sentence) or mis-direct them, so it must skip them; the dependency extractor
# (spacy.py _extract_syntactic / _ATTACHMENT_RELATIONS) owns them.
_SYNTAX_ONLY_RELATIONS = {
    "DESCRIBES", "HAS_SEVERITY", "HAS_COLOR", "HAS_SHAPE", "HAS_TEXTURE",
}


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Load tokenizer + model with entity-marker special tokens registered."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BioLinkBERT RE] Loading '{model_name}' on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    )
    model = AutoModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device).eval()
    return tokenizer, model, device


@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load(SPACY_MODEL_ENGLISH)


def _get_embedding(text: str, tokenizer, model, device: str) -> np.ndarray:
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding=True,
    ).to(device)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].cpu().numpy()


def _mark_entities(sentence: str, e1: str, e2: str) -> str:
    def _replace_once(text, word, open_tag, close_tag):
        pattern = _re.compile(_re.escape(word), _re.IGNORECASE)
        return pattern.sub(f"{open_tag} {word} {close_tag}", text, count=1)

    marked = _replace_once(sentence, e1, "[E1]", "[/E1]")
    return _replace_once(marked, e2, "[E2]", "[/E2]")


def _raw_pairs(doc, entities: list[dict]) -> list[tuple]:
    """(sentence_text, e1, e2) tuples — from the syntactic candidate generator
    (pruned dependency paths) or the legacy all-pairs enumeration, per RE_PAIRING."""
    if RE_PAIRING == "syntactic":
        return candidate_gen.generate_pairs(doc, entities)
    raw: list[tuple] = []
    for sent in doc.sents:
        sent_ents = [e for e in entities if e["word"].lower() in sent.text.lower()]
        for e1, e2 in combinations(sent_ents, 2):
            raw.append((sent.text, e1, e2))
    return raw


def _entity_pairs_in_sentences(text: str, entities: list[dict], nlp_model) -> list[dict]:
    doc = nlp_model(text)
    pairs: list[dict] = []
    seen: set[tuple] = set()

    for sent_text, e1, e2 in _raw_pairs(doc, entities):
        type_key = (e1["label"], e2["label"])
        if type_key not in ENTITY_PAIR_RELATIONS:
            continue
        # Drop syntax-only relations (e.g. DESCRIBES) — the dependency parser
        # owns them; biolink would attach a descriptor to every entity.
        candidates = [r for r in ENTITY_PAIR_RELATIONS[type_key]
                      if r not in _SYNTAX_ONLY_RELATIONS]
        if not candidates:
            continue
        dedup_key = (sent_text, tuple(sorted([e1["word"], e2["word"]])))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        pairs.append({
            "sentence": sent_text,
            "entity1": e1["word"],
            "entity1_label": e1["label"],
            "entity2": e2["word"],
            "entity2_label": e2["label"],
            "candidate_relations": candidates,
        })
    return pairs


def run(df_ner: pd.DataFrame, df_structured: pd.DataFrame, biolink_model_name: str) -> pd.DataFrame:
    """Score every valid entity pair against relation prototype embeddings."""
    tokenizer, model, device = _load_model(biolink_model_name)
    nlp = _load_spacy()

    print("[BioLinkBERT RE] Computing prototype embeddings …")
    proto_embs = {
        rel: _get_embedding(text, tokenizer, model, device)
        for rel, text in RELATION_PROTOTYPES.items()
    }

    results: list[dict] = []
    id_col = df_structured.set_index("id")

    for record_id in tqdm(df_ner["id"].unique(), desc="biolink-RE", unit="rec"):
        if record_id not in id_col.index:
            continue
        text = id_col.loc[record_id, "text_en"]
        if not isinstance(text, str) or not text.strip():
            continue

        entities = record_entities(df_ner, record_id)

        for pair in _entity_pairs_in_sentences(text, entities, nlp):
            marked = _mark_entities(pair["sentence"], pair["entity1"], pair["entity2"])
            pair_emb = _get_embedding(marked, tokenizer, model, device)

            best_rel, best_score = "NO_RELATION", 0.0
            for rel in pair["candidate_relations"]:
                score = float(cosine_similarity(pair_emb, proto_embs[rel])[0][0])
                if score > best_score:
                    best_score, best_rel = score, rel

            if best_rel != "NO_RELATION" and best_score >= SCORE_THRESHOLD:
                results.append({
                    "id": record_id,
                    "entity1": pair["entity1"],
                    "entity1_label": pair["entity1_label"],
                    "relation": best_rel,
                    "entity2": pair["entity2"],
                    "entity2_label": pair["entity2_label"],
                    "score": round(best_score, 4),
                    "sentence": pair["sentence"],
                    "method": "biolink_bert",
                })

    print(f"[BioLinkBERT RE] Found {len(results)} relations.")
    return pd.DataFrame(results)
