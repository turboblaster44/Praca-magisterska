"""Gazetteer NER: PhraseMatcher lexicons + regex extractors on translated text."""

from functools import lru_cache
from typing import Iterable

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

from config import SPACY_MODEL_ENGLISH
from nlp.ner.ontology import LEXICONS, REGEX_PATTERNS


@lru_cache(maxsize=1)
def _load_spacy():
    return spacy.load(SPACY_MODEL_ENGLISH, disable=["ner", "lemmatizer"])


def _build_matcher(nlp) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for label, terms in LEXICONS.items():
        matcher.add(label, [nlp.make_doc(t) for t in terms])
    return matcher


def _phrase_matches(nlp, matcher: PhraseMatcher, text: str) -> Iterable[tuple[str, str, int, int]]:
    doc = nlp(text)
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        yield span.text, nlp.vocab.strings[match_id], span.start_char, span.end_char


def _regex_matches(text: str) -> Iterable[tuple[str, str, int, int]]:
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            yield m.group(0), label, m.start(), m.end()


def run(df_structured: pd.DataFrame) -> pd.DataFrame:
    """Extract gazetteer + regex entities from translated text."""
    print("=" * 60)
    print("NER GAZETTEER EXTRACTION")
    print("=" * 60)

    nlp = _load_spacy()
    matcher = _build_matcher(nlp)

    rows: list[dict] = []
    for _, row in df_structured.iterrows():
        text = row.get("text_en")
        if not isinstance(text, str) or not text.strip():
            continue
        for word, label, start, end in _phrase_matches(nlp, matcher, text):
            rows.append({"id": row["id"], "word": word, "label": label,
                         "score": 1.0, "text_en": text, "method": "gazetteer",
                         "start": start, "end": end})
        for word, label, start, end in _regex_matches(text):
            rows.append({"id": row["id"], "word": word, "label": label,
                         "score": 1.0, "text_en": text, "method": "gazetteer",
                         "start": start, "end": end})

    df = pd.DataFrame(rows, columns=["id", "word", "label", "score", "text_en",
                                     "method", "start", "end"])
    print(f"[GAZETTEER] Extracted {len(df)} entities.")
    return df
