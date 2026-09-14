"""
Renders the fig:re_candidates figure: the collapsed dependency graph with the
candidate-pair verdicts drawn on top.

Companion to dep_tree_demo.py, which prints the same information as text. Reuses
nlp/re/candidates.py directly, so the picture cannot drift from the implementation.

Usage:
    python -m testing.dep_tree_figure
    python -m testing.dep_tree_figure "CT revealed a mass in the left lung." \
        --entities "CT:DIAGNOSTIC_PROCEDURE,mass:DISEASE_DISORDER,left lung:BIOLOGICAL_STRUCTURE"
    python -m testing.dep_tree_figure "..." --name re_candidates --format pdf

Writes testing/output/<name>.pdf (and .png with --format png).
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import graphviz
import spacy

from nlp.re import candidates as cg
from testing.dep_tree_demo import decide

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# The two cases the figure has to show: a pair rejected on path length, and a pair
# rejected on syntax (coordination).
EXAMPLES = {
    "odleglosc": (
        "CT revealed a mass in the left lung.",
        "CT:DIAGNOSTIC_PROCEDURE,mass:DISEASE_DISORDER,"
        "left lung:BIOLOGICAL_STRUCTURE",
        "re_candidates_odleglosc",
    ),
    "skladnia": (
        "Diabetes and hypertension cause neuropathy.",
        "Diabetes:DISEASE_DISORDER,hypertension:DISEASE_DISORDER,"
        "neuropathy:DISEASE_DISORDER",
        "re_candidates_skladnia",
    ),
    # The one case where "entity on the interior" fires on its own: the path is
    # short enough, but it runs through the middle entity of an anatomical chain.
    "encja": (
        "A lesion in the left lobe of the thyroid.",
        "lesion:DISEASE_DISORDER,left lobe:BIOLOGICAL_STRUCTURE,"
        "thyroid:BIOLOGICAL_STRUCTURE",
        "re_candidates_encja",
    ),
}

ENTITY_FILL = "#dce9f5"
ACCEPT_COLOR = "#1a7f37"
REJECT_COLOR = "#b3261e"

# dep_tree_demo keeps its reasons ASCII for the console; the figure goes into a
# Polish thesis, so restore the diacritics here.
_PL = {
    "dlugosc": "długość",
    "sciezka za dluga": "ścieżka za długa",
    "rozdzielona encja": "rozdzielona encją",
    "brak polaczenia w grafie": "brak połączenia w grafie",
}


def pl(reason: str) -> str:
    for a, b in _PL.items():
        reason = reason.replace(a, b)
    return reason


def parse_entities(spec: str) -> list[dict]:
    """'word:LABEL,word:LABEL' -> the entity dicts the RE layer works with."""
    out = []
    for item in spec.split(","):
        word, _, label = item.rpartition(":")
        if word.strip() and label.strip():
            out.append({"word": word.strip(), "label": label.strip().upper()})
    return out


def build(sentence: str, entities: list[dict], max_path: int) -> graphviz.Digraph:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    doc = nlp(sentence)
    sent = list(doc.sents)[0]

    roots = {}
    for e in entities:
        r = cg._entity_root(sent, doc, e["word"])
        if r is not None:
            roots[r.i] = e
    adj = cg._collapsed_adjacency(sent)
    ent_idx = set(roots)

    # Verdicts first: they decide which tokens the figure needs at all.
    idxs = sorted(ent_idx)
    verdicts = []
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            a, b = idxs[i], idxs[j]
            ok, reason, path = decide(doc, adj, a, b, ent_idx, max_path)
            verdicts.append((a, b, ok, reason, path))

    # Only tokens lying on a path between two entities carry information here;
    # bypassed prepositions and words inside an entity span would be noise.
    keep = set(ent_idx)
    for _, _, _, _, path in verdicts:
        if path:
            keep.update(path)

    g = graphviz.Digraph("re_candidates", format="pdf")
    # pad keeps edge labels and the topmost node off the image border; without it
    # graphviz trims the drawing flush and the tallest glyphs get shaved.
    g.attr(rankdir="TB", splines="spline", nodesep="0.5", ranksep="0.7",
           pad="0.4", dpi="200")
    g.attr("node", fontname="Helvetica", fontsize="11", shape="box",
           style="rounded", color="#888888")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#888888",
           arrowhead="none")

    for i in sorted(keep):
        if i in ent_idx:
            g.node(str(i), f"{roots[i]['word']}\\n{roots[i]['label']}",
                   style="rounded,filled", fillcolor=ENTITY_FILL, color="#4a7ea8")
        else:
            g.node(str(i), doc[i].text, shape="plaintext", style="")

    # Skeleton of the collapsed graph, one edge per pair.
    drawn = set()
    for a, neighbours in adj.items():
        for b in neighbours:
            if a not in keep or b not in keep:
                continue
            key = tuple(sorted((a, b)))
            if key in drawn:
                continue
            drawn.add(key)
            g.edge(str(key[0]), str(key[1]))

    # Verdicts, laid over the skeleton without disturbing the layout.
    for a, b, ok, reason, _ in verdicts:
        g.edge(
            str(a), str(b),
            label=f"  kandydat, {pl(reason)}  " if ok
                  else f"  odrzucona: {pl(reason)}  ",
            color=ACCEPT_COLOR if ok else REJECT_COLOR,
            fontcolor=ACCEPT_COLOR if ok else REJECT_COLOR,
            style="bold" if ok else "dotted",
            constraint="false", arrowhead="none",
        )
    return g


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sentence", nargs="?",
                    help="own sentence; without it both presets are rendered")
    ap.add_argument("--entities", help="'word:LABEL,word:LABEL' — omits the NER models")
    ap.add_argument("--max-path", type=int, default=2)
    ap.add_argument("--name", default="re_candidates")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.sentence:
        jobs = [(args.sentence, args.entities or "", args.name)]
    else:
        jobs = list(EXAMPLES.values())

    for sentence, entities, name in jobs:
        g = build(sentence, parse_entities(entities), args.max_path)
        g.format = args.format
        path = g.render(filename=name, directory=OUTPUT_DIR, cleanup=True)
        print(f"[figure] {path}")


if __name__ == "__main__":
    main()
