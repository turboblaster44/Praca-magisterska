"""
Visualize the spaCy dependency parse of a sentence — the syntactic graph the
rule-based RE walks (amod/compound -> DESCRIBES, conj coordination, verb
subj/obj). Companion to test_sentence.py, but for the *parse* rather than the
NER/RE output.

Usage:
    python -m testing.show_parse "Chronic and inflammatory lesions affect the sinuses."
    python -m testing.show_parse                     # built-in example
    python -m testing.show_parse "..." --sci         # use en_core_sci_md instead
    python -m testing.show_parse "..." --serve        # live server at :5000

Console: one row per token (text, POS, dep, head) + the RE-relevant dependencies
highlighted. HTML: a displaCy dependency diagram saved under testing/output/.
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import spacy
from spacy import displacy

from config import SPACY_MODEL_ENGLISH, SCISPACY_MODEL

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

DEFAULT_SENTENCE = (
    "Chronic and inflammatory lesions affect the maxillary sinuses and the "
    "nasal septum, and diabetes and hypertension cause peripheral neuropathy."
)

# Dependencies the syntactic RE keys on — flagged in the console for quick reading.
_RE_DEPS = {
    "amod": "DESCRIBES (modifier)",
    "compound": "DESCRIBES (modifier)",
    "conj": "coordination (and/or)",
    "cc": "coordination marker",
    "nsubj": "subject", "nsubjpass": "subject",
    "dobj": "object", "pobj": "prep-object", "attr": "object", "oprd": "object",
    "prep": "preposition (-> LOCATED_IN, planned)",
}


def print_tokens(doc) -> None:
    print("\n" + "=" * 78)
    print("DEPENDENCY PARSE (token -> head)")
    print("=" * 78)
    print(f"  {'i':>3}  {'token':<16}{'POS':<7}{'dep':<12}{'head':<16}RE?")
    print("  " + "-" * 74)
    for tok in doc:
        note = _RE_DEPS.get(tok.dep_, "")
        print(f"  {tok.i:>3}  {tok.text[:15]:<16}{tok.pos_:<7}{tok.dep_:<12}"
              f"{tok.head.text[:15]:<16}{note}")


def _label(tok) -> str:
    note = _RE_DEPS.get(tok.dep_, "")
    tag = f"  <- {note}" if note else ""
    return f"{tok.text} [{tok.dep_}] ({tok.pos_}){tag}"


def _subtree(tok, prefix: str, is_last: bool, out: list) -> None:
    out.append(prefix + ("`-- " if is_last else "|-- ") + _label(tok))
    kids = sorted(tok.children, key=lambda c: c.i)
    child_prefix = prefix + ("    " if is_last else "|   ")
    for i, kid in enumerate(kids):
        _subtree(kid, child_prefix, i == len(kids) - 1, out)


def print_tree(doc) -> None:
    """Indented dependency tree per sentence: children sit under their head, so a
    verb's whole subtree (its subjects/objects/modifiers) is visually grouped."""
    print("\n" + "=" * 78)
    print("DEPENDENCY TREE (children nested under their head)")
    print("=" * 78)
    for sent in doc.sents:
        out = [_label(sent.root)]  # ROOT has no connector
        kids = sorted(sent.root.children, key=lambda c: c.i)
        for i, kid in enumerate(kids):
            _subtree(kid, "", i == len(kids) - 1, out)
        print("\n".join(out))
        print()


def print_coordination(doc) -> None:
    """Show conjunct groups explicitly — the thing the RE now follows."""
    groups = []
    for tok in doc:
        conj = list(tok.conjuncts)
        if conj and all(tok.i <= c.i for c in conj):  # print once per group (from head)
            groups.append([tok.text] + [c.text for c in conj])
    if groups:
        print("\n" + "=" * 78)
        print("COORDINATION GROUPS (token.conjuncts) - RE links all members")
        print("=" * 78)
        for g in groups:
            print("   " + "  <->  ".join(g))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Render a sentence's spaCy dependency graph.")
    parser.add_argument("sentence", nargs="?", default=DEFAULT_SENTENCE)
    parser.add_argument("--sci", action="store_true",
                        help="Use en_core_sci_md instead of en_core_web_sm.")
    parser.add_argument("--serve", action="store_true",
                        help="Serve the diagram live at http://localhost:5000 (Ctrl+C to stop).")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "parse.html"))
    args = parser.parse_args(argv)

    model = SCISPACY_MODEL if args.sci else SPACY_MODEL_ENGLISH
    print(f"Loading spaCy model: {model}")
    nlp = spacy.load(model)
    doc = nlp(args.sentence)

    print("\nSENTENCE:\n ", args.sentence)
    print_tree(doc)
    print_tokens(doc)
    print_coordination(doc)

    if args.serve:
        print("\nServing at http://localhost:5000  (Ctrl+C to stop) …")
        displacy.serve(doc, style="dep", options={"compact": True, "distance": 110})
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = displacy.render(doc, style="dep", page=True,
                           options={"compact": True, "distance": 110})
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\n[parse] Dependency diagram saved -> {args.output}")
    print("        Open it in a browser to see the graph.")


if __name__ == "__main__":
    main()
