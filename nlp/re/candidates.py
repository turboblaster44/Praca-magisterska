"""
Syntactic candidate-pair generation for relation extraction.

Instead of enumerating ALL entity pairs in a sentence (which floods the
classifier with syntactically-unrelated "cousin" pairs and yields false
positives), keep only pairs connected by a SHORT dependency path — the pruned
shortest-dependency-path idea (Bunescu & Mooney 2005; Zhang, Qi & Manning 2018),
adapted here to *select* candidate pairs rather than to build classifier features.

A pair (A, B) is a candidate iff, in the dependency graph with function words
collapsed (prepositions / conjunctions / determiners made transparent), the
shortest path between A and B:
  - passes through NO other entity (else A relates to that middle entity, not B),
  - is at most `max_path_len` edges — 1 = direct modifier ("chronic"->"lesions"),
    2 = through one content hub (verb/preposition: "lesions"-affect-"sinuses"),
  - and A, B are not coordinated with each other (conjuncts are parallel — e.g.
    "diabetes and hypertension" do not relate to each other, each relates across
    the verb to the object).

Returns pairs in surface (sentence) order so the caller's type-pair lookup and
directionality behave exactly as with the previous all-pairs enumeration.
"""

from collections import deque

# Deps dropped entirely — they are never relation endpoints.
_SKIP_DEPS = {"det", "punct", "cc"}


def _entity_root(sent, doc, word: str):
    """Representative token (span root) for an entity's surface form in *sent*."""
    pos = sent.text.lower().find(word.lower())
    if pos == -1:
        return None
    start = sent.start_char + pos
    span = doc.char_span(start, start + len(word), alignment_mode="expand")
    return span.root if span is not None else None


def _collapsed_adjacency(sent) -> dict[int, set]:
    """Undirected adjacency over content tokens, function words collapsed.

    - det / punct / cc            : dropped (not nodes)
    - preposition                 : head links directly to each pobj (prep bypassed)
    - conjunct                    : links to its head's head (parallel attachment)
    - everything else             : token <-> its head
    """
    adj: dict[int, set] = {}

    def link(a, b):
        if a is None or b is None or a.i == b.i:
            return
        adj.setdefault(a.i, set()).add(b.i)
        adj.setdefault(b.i, set()).add(a.i)

    for tok in sent:
        dep = tok.dep_
        if dep in _SKIP_DEPS:
            continue
        if dep == "prep":
            for pobj in (c for c in tok.children if c.dep_ == "pobj"):
                link(tok.head, pobj)
            continue
        if dep == "conj":
            # Attach in parallel with the first conjunct — i.e. to whatever that
            # conjunct hangs off AFTER collapsing. Walking past prepositions
            # matters: in "given for diabetes and hypertension" the first
            # conjunct is a pobj, so its raw head is the (bypassed) preposition;
            # linking there would strand the second conjunct behind a dead node.
            anchor = tok.head.head
            while anchor.dep_ == "prep" and anchor.head is not anchor:
                anchor = anchor.head
            link(tok, anchor)
            continue
        if tok.head is not tok:
            link(tok, tok.head)
    return adj


def _shortest_path(adj: dict[int, set], start_i: int, goal_i: int):
    """BFS shortest path (list of token indices), or None if disconnected."""
    if start_i == goal_i:
        return [start_i]
    prev = {start_i: None}
    q = deque([start_i])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == goal_i:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(nxt)
    return None


def generate_pairs(doc, entities: list[dict], max_path_len: int = 2) -> list[tuple]:
    """Candidate (sentence_text, entity1, entity2) tuples, in surface order.

    max_path_len=2 enforces "at most one content hub" between the two entities
    (1 = modifier edge, 2 = through a single verb/preposition); >=3 means two
    hubs / cross-clause and is treated as an unrelated cousin.
    """
    out: list[tuple] = []
    for sent in doc.sents:
        sent_ents = [e for e in entities if e["word"].lower() in sent.text.lower()]
        if len(sent_ents) < 2:
            continue

        roots: dict[int, dict] = {}     # token index -> entity dict
        for e in sent_ents:
            r = _entity_root(sent, doc, e["word"])
            if r is not None and r.i not in roots:
                roots[r.i] = e
        if len(roots) < 2:
            continue

        adj = _collapsed_adjacency(sent)
        entity_idx = set(roots)
        idxs = sorted(roots)            # sorted => surface order (a before b)

        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                tok_a, tok_b = doc[a], doc[b]
                # coordinated with each other -> parallel, not a relation
                if tok_b in tok_a.conjuncts or tok_a in tok_b.conjuncts:
                    continue
                path = _shortest_path(adj, a, b)
                if path is None or len(path) - 1 > max_path_len:
                    continue
                # another entity on the interior -> A relates to it, not to B
                if any(node in entity_idx for node in path[1:-1]):
                    continue
                e_a, e_b = roots[a], roots[b]
                if str(e_a["word"]) == str(e_b["word"]):
                    continue
                out.append((sent.text, e_a, e_b))
    return out
