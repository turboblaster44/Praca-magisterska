"""
TEMPORARY helper for the thesis figure (fig:re_candidates).

Runs NER on one sentence, parses it with spaCy and then dumps, in a redrawable
form:
  1. the dependency tree, with entity tokens marked,
  2. the collapsed graph actually used by nlp/re/candidates.py
     (prepositions bypassed, det/punct/cc dropped, conjuncts re-attached),
  3. EVERY entity pair with accept / reject + the reason and the path,
  4. the candidate list as really returned by candidates.generate_pairs().

Usage:
    python -m testing.dep_tree_demo
    python -m testing.dep_tree_demo "CT revealed a mass in the left lung."
    python -m testing.dep_tree_demo "..." --entities "CT:DIAGNOSTIC_PROCEDURE,mass:DISEASE_DISORDER"
    python -m testing.dep_tree_demo "..." --max-path 2

--entities skips the NER models entirely (fast iteration on the figure).
"""

import argparse
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import pandas as pd
import spacy

from config import SPACY_MODEL_ENGLISH
from nlp.re import candidates as cg

DEFAULT_SENTENCE = (
    "CT revealed a severe lesion in the left lung, "
    "and metformin was given for diabetes and hypertension."
)

BAR = "=" * 78


def get_entities(sentence: str, manual: str | None, use_context: bool) -> list[dict]:
    """Entity dicts (word / label / score) from the real NER step, or from --entities."""
    if manual:
        out = []
        for item in manual.split(","):
            word, _, label = item.partition(":")
            out.append({"word": word.strip(), "label": (label or "UNKNOWN").strip(),
                        "score": 1.0, "method": "manual"})
        return out

    from nlp.ner import tasks as ner_tasks
    df_structured = pd.DataFrame([{"id": 0, "text": sentence, "text_en": sentence}])
    df_ner, _ = ner_tasks.run(
        df_structured, use_neural=True, use_gazetteer=True, use_llm=False,
        use_context=use_context, use_umls=False,
    )
    return df_ner.to_dict("records")


def print_tree(sent, ent_by_idx: dict[int, dict]) -> None:
    """ASCII dependency tree of one sentence, entity tokens tagged with their label."""
    def walk(tok, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            branch = ""
            child_prefix = ""
        else:
            branch = "`-- " if is_last else "|-- "
            child_prefix = "    " if is_last else "|   "
        tag = ""
        if tok.i in ent_by_idx:
            tag = f"   <<{ent_by_idx[tok.i]['label']}>>"
        dep = "ROOT" if is_root else tok.dep_
        print(f"  {prefix}{branch}{tok.text}  ({dep}, {tok.pos_}){tag}")
        kids = list(tok.children)
        for n, kid in enumerate(kids):
            walk(kid, prefix + child_prefix, n == len(kids) - 1, False)

    walk(sent.root, "", True, True)


def print_collapsed(sent, adj: dict[int, set], doc, ent_by_idx: dict[int, dict]) -> None:
    """Edges of the collapsed graph — the one the candidate generator walks."""
    def node(i: int) -> str:
        mark = "*" if i in ent_by_idx else " "
        return f"{mark}{doc[i].text}"

    seen = set()
    for a, neighbours in sorted(adj.items()):
        for b in sorted(neighbours):
            if (b, a) in seen:
                continue
            seen.add((a, b))
            print(f"    {node(a):>18}  ---  {node(b)}")


def decide(doc, adj, a: int, b: int, ent_idx: set, max_path: int):
    """Repeat the candidates.generate_pairs decision, but keep the reason."""
    tok_a, tok_b = doc[a], doc[b]
    if tok_b in tok_a.conjuncts or tok_a in tok_b.conjuncts:
        return False, "koordynacja (elementy wyliczenia)", None
    path = cg._shortest_path(adj, a, b)
    if path is None:
        return False, "brak polaczenia w grafie", None
    if len(path) - 1 > max_path:
        return False, f"sciezka za dluga (dlugosc {len(path) - 1})", path
    blockers = [doc[n].text for n in path[1:-1] if n in ent_idx]
    if blockers:
        return False, f"rozdzielona encja '{blockers[0]}'", path
    return True, f"dlugosc {len(path) - 1}", path


def fmt_path(doc, path, ent_idx: set) -> str:
    if not path:
        return "-"
    return " - ".join(f"[{doc[i].text}]" if i in ent_idx else doc[i].text for i in path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Dependency tree + RE candidate pairs.")
    parser.add_argument("sentence", nargs="?", default=DEFAULT_SENTENCE)
    parser.add_argument("--entities", default=None,
                        help='Skip NER, e.g. "CT:DIAGNOSTIC_PROCEDURE,lesion:DISEASE_DISORDER"')
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--max-path", type=int, default=2)
    args = parser.parse_args(argv)

    entities = get_entities(args.sentence, args.entities, not args.no_context)

    print("\n" + BAR)
    print("SENTENCE")
    print(BAR)
    print(" ", args.sentence)

    print("\n" + BAR)
    print(f"ENTITIES ({len(entities)})")
    print(BAR)
    for e in entities:
        print(f"  {str(e['word'])[:30]:<32}{e['label']}")

    nlp = spacy.load(SPACY_MODEL_ENGLISH)
    doc = nlp(args.sentence)

    for s_no, sent in enumerate(doc.sents, 1):
        sent_ents = [e for e in entities if e["word"].lower() in sent.text.lower()]
        roots: dict[int, dict] = {}
        for e in sent_ents:
            r = cg._entity_root(sent, doc, e["word"])
            if r is not None and r.i not in roots:
                roots[r.i] = e

        print("\n" + BAR)
        print(f"SENTENCE {s_no}: {sent.text}")
        print(BAR)

        print("\n  DEPENDENCY TREE (* = entity head token)")
        print_tree(sent, roots)

        if len(roots) < 2:
            print("\n  < 2 entities -> no pairs")
            continue

        adj = cg._collapsed_adjacency(sent)
        print("\n  COLLAPSED GRAPH (prepositions bypassed, det/punct/cc dropped;"
              " * = entity)")
        print_collapsed(sent, adj, doc, roots)

        ent_idx = set(roots)
        idxs = sorted(roots)
        print(f"\n  PAIRS (max_path_len={args.max_path})")
        print(f"    {'':3}{'pair':<40}{'reason':<34}path")
        print("    " + "-" * 72)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                ok, reason, path = decide(doc, adj, a, b, ent_idx, args.max_path)
                pair = f"{roots[a]['word']} -> {roots[b]['word']}"
                mark = "OK " if ok else "NO "
                print(f"    {mark}{pair[:38]:<40}{reason:<34}"
                      f"{fmt_path(doc, path, ent_idx)}")

    print("\n" + BAR)
    print("CANDIDATES RETURNED BY candidates.generate_pairs()")
    print(BAR)
    pairs = cg.generate_pairs(doc, entities, max_path_len=args.max_path)
    if not pairs:
        print("  (none)")
    for _, e1, e2 in pairs:
        print(f"  [{e1['label']}] {e1['word']}   <-->   [{e2['label']}] {e2['word']}")
    print()


if __name__ == "__main__":
    main()
