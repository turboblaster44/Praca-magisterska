"""LLM-based NER peer method (opt-in).

Adds entities the neural / gazetteer methods miss, constrained to the ENT ontology
and with a graded confidence. Emits the same 8-column contract as the other NER
methods (``id, word, label, score, text_en, method, start, end``) so the ConText
assertion and UMLS linking sub-steps in ``nlp/ner/tasks.py`` enrich its rows exactly
like the rest. Needs an LLM API key (see ``nlp/llm_client``); without one it returns
an empty frame (no-op).
"""

import pandas as pd
from tqdm import tqdm

from config import NER_LLM_BATCH_SIZE
from nlp.llm_client import generate, reset_stats, stats
from nlp.ner.ontology import ENT

_METHOD = "llm"
_LABELS = set(ENT.all())
_COLUMNS = ["id", "word", "label", "score", "text_en", "method", "start", "end"]


def _build_prompt(batch: list[tuple[object, str]]) -> str:
    labels = ", ".join(ENT.all())
    records = "\n".join(f"{rec_id}: {text}" for rec_id, text in batch)
    return (
        f"Sentences, one per line as 'record_id: sentence':\n{records}\n"
        "Extract every medical named entity from EACH sentence. For each entity, "
        "output exactly one line formatted as:\n"
        "record_id || entity_text || LABEL || confidence\n"
        f"LABEL must be one of exactly these types: {labels}.\n"
        "confidence is your certainty between 0 and 1. Copy the entity text exactly "
        "as it appears in its own sentence and keep the record_id unchanged. Skip "
        "anything that does not fit one of the listed types. Do not add commentary.\n"
    )


def _parse(text: str) -> list[tuple[str, str, str, float]]:
    """Parse 'id || entity || LABEL || confidence' lines; drop labels outside the ontology."""
    out: list[tuple[str, str, str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if "||" not in line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) < 4:
            continue
        rec_id = parts[0]
        word = parts[1].strip('"').strip()
        label = parts[2].strip().upper()
        if not word or label not in _LABELS:
            continue
        try:
            score = float(parts[3])
        except ValueError:
            score = 0.0
        out.append((rec_id, word, label, round(score, 4)))
    return out


def _offsets(text: str, word: str):
    """First-occurrence char offsets (same lookup as assertion._span_for_row)."""
    pos = text.lower().find(word.lower())
    if pos == -1:
        return None, None
    return pos, pos + len(word)


def run(df_structured: pd.DataFrame, lang_col: str = "text_en") -> pd.DataFrame:
    """Extract ontology-constrained entities via the LLM; empty frame if no API key."""
    records = [(rec["id"], rec[lang_col]) for _, rec in df_structured.iterrows()
               if isinstance(rec.get(lang_col), str) and rec[lang_col].strip()]
    batches = [records[i:i + NER_LLM_BATCH_SIZE]
               for i in range(0, len(records), NER_LLM_BATCH_SIZE)]
    reset_stats()

    rows: list[dict] = []
    for batch in tqdm(batches, desc="NER-LLM", unit="batch"):
        by_str_id = {str(rec_id): (rec_id, text) for rec_id, text in batch}
        response, _ = generate(_build_prompt(batch))
        if not response:
            continue
        for rec_id_str, word, label, score in _parse(response):
            if rec_id_str not in by_str_id:  # id the model invented or mangled
                continue
            rec_id, text = by_str_id[rec_id_str]
            start, end = _offsets(text, word)
            rows.append({
                "id": rec_id, "word": word, "label": label, "score": score,
                "text_en": text, "method": _METHOD, "start": start, "end": end,
            })

    s = stats()
    print(f"[LLM NER] Found {len(rows)} entities "
          f"({s['calls']} calls, {s['retries']} retries, {s['failed']} failed).")
    if s["failed"]:
        print(f"[LLM NER] WARNING: {s['failed']} of {len(batches)} batches produced "
              f"nothing — up to {s['failed'] * NER_LLM_BATCH_SIZE} records unprocessed.")
    return pd.DataFrame(rows, columns=_COLUMNS)
