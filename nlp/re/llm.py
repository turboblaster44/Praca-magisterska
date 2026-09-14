"""LLM-based relation extraction peer method (opt-in).

Reuses biolink's candidate-pair generation (pruned-SDP pairing + ontology lookup +
syntax-only filtering) to get, per sentence, the entity pairs and the exact set of
ontology relations valid for each pair's type combination; then asks an LLM to pick
the single best directed relation (or NO_RELATION) per pair with a confidence. Emits
the core ``RE_COLUMNS`` schema (``method="llm"``) so dedup, ConText assertion and
UMLS annotation in ``nlp/re/tasks.py`` apply to it like the other methods. Needs an
LLM API key (see ``nlp/llm_client``); without one it returns an empty frame (no-op).
"""

import pandas as pd
from tqdm import tqdm

from config import LLM_RE_SCORE_THRESHOLD, RE_LLM_BATCH_SIZE
from nlp._pipeline import record_entities
from nlp.llm_client import generate, reset_stats, stats
from nlp.re.biolink import _entity_pairs_in_sentences, _load_spacy

_METHOD = "llm"
_NO_RELATION = "NO_RELATION"
_COLUMNS = [
    "id", "entity1", "entity1_label", "relation",
    "entity2", "entity2_label", "score", "sentence", "method",
]


def _build_prompt(units: list[tuple]) -> str:
    """One prompt for a batch of (record_id, sentence, pairs) units, pairs numbered
    continuously across the whole batch."""
    lines = []
    i = 0
    for _, sentence, pairs in units:
        lines.append(f'Sentence: "{sentence}"')
        for p in pairs:
            i += 1
            allowed = ", ".join(p["candidate_relations"] + [_NO_RELATION])
            lines.append(
                f'{i}. "{p["entity1"]}" ({p["entity1_label"]}) -> '
                f'"{p["entity2"]}" ({p["entity2_label"]}) | allowed: {allowed}'
            )
    pairs_block = "\n".join(lines)
    return (
        "For each numbered directed entity pair below, choose the single best relation "
        "from THAT pair's own allowed list (or NO_RELATION if none holds), and give a "
        "confidence between 0 and 1. Judge a pair only against the sentence it is "
        "listed under.\n\n"
        f"{pairs_block}\n\n"
        "Respond with exactly one line per pair, keeping its number, formatted as:\n"
        "<number>. <RELATION> | <confidence>\n"
        "Use only a label from that pair's allowed list. Do not add commentary.\n"
    )


def _parse(text: str, n: int) -> list[tuple[str, float]]:
    """Parse '<i>. <RELATION> | <conf>' lines into a list indexed by pair number.

    Keyed on each line's own number rather than its position, so one skipped or extra
    line cannot shift every later answer — a real risk once a prompt carries the pairs
    of several sentences. Falls back to positional order for an unnumbered response.
    """
    parsed = [(_NO_RELATION, 0.0)] * n
    unnumbered: list[tuple[str, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        left, right = line.rsplit("|", 1)
        try:
            conf = float(right.strip())
        except ValueError:
            conf = 0.0
        head, sep, rest = left.partition(".")  # leading "N." carries the pair number
        idx = int(head.strip()) - 1 if sep and head.strip().isdigit() else None
        label = (rest if idx is not None else left).strip().strip('"').upper()
        if idx is not None and 0 <= idx < n:
            parsed[idx] = (label, round(conf, 4))
        elif idx is None:
            unnumbered.append((label, round(conf, 4)))

    if unnumbered and all(p == (_NO_RELATION, 0.0) for p in parsed):
        for i, item in enumerate(unnumbered[:n]):
            parsed[i] = item
    return parsed


def run(df_ner: pd.DataFrame, df_structured: pd.DataFrame) -> pd.DataFrame:
    """Classify ontology-valid candidate pairs via the LLM; empty frame if no API key."""
    nlp = _load_spacy()
    id_col = df_structured.set_index("id")
    results: list[dict] = []

    # A unit = one sentence with its candidate pairs (a record may span several).
    units: list[tuple] = []
    for record_id in tqdm(df_ner["id"].unique(), desc="RE-LLM pairs", unit="rec"):
        if record_id not in id_col.index:
            continue
        text = id_col.loc[record_id, "text_en"]
        if not isinstance(text, str) or not text.strip():
            continue

        entities = record_entities(df_ner, record_id)
        by_sentence: dict[str, list[dict]] = {}
        for p in _entity_pairs_in_sentences(text, entities, nlp):
            by_sentence.setdefault(p["sentence"], []).append(p)
        units.extend((record_id, s, ps) for s, ps in by_sentence.items())

    batches = [units[i:i + RE_LLM_BATCH_SIZE]
               for i in range(0, len(units), RE_LLM_BATCH_SIZE)]
    reset_stats()

    for batch in tqdm(batches, desc="RE-LLM", unit="batch"):
        flat = [(record_id, p) for record_id, _, pairs in batch for p in pairs]
        response, _ = generate(_build_prompt(batch))
        if not response:
            continue
        for (record_id, p), (relation, score) in zip(flat, _parse(response, len(flat))):
            if relation == _NO_RELATION or relation not in set(p["candidate_relations"]):
                continue
            if score < LLM_RE_SCORE_THRESHOLD:
                continue
            results.append({
                "id": record_id,
                "entity1": p["entity1"],
                "entity1_label": p["entity1_label"],
                "relation": relation,
                "entity2": p["entity2"],
                "entity2_label": p["entity2_label"],
                "score": score,
                "sentence": p["sentence"],
                "method": _METHOD,
            })

    s = stats()
    print(f"[LLM RE] Found {len(results)} relations "
          f"({s['calls']} calls, {s['retries']} retries, {s['failed']} failed).")
    if s["failed"]:
        print(f"[LLM RE] WARNING: {s['failed']} of {len(batches)} batches produced "
              f"nothing — up to {s['failed'] * RE_LLM_BATCH_SIZE} sentences unclassified.")
    return pd.DataFrame(results, columns=_COLUMNS)
