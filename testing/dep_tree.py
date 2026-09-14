"""
Visualise spaCy dependency tree for a sentence.

Usage:
    python testing/dep_tree.py "The patient was diagnosed with hypertension."
    python testing/dep_tree.py   # uses built-in example
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import spacy

SENTENCE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "The 45-year-old male patient was diagnosed with hypertension in 2018 and was treated with Atenolol for 6 months."
)

nlp = spacy.load("en_core_web_sm")
doc = nlp(SENTENCE)

print(f"\nSentence: {SENTENCE}\n")
print(f"{'TOKEN':<20} {'LEMMA':<20} {'POS':<10} {'DEP':<15} {'HEAD'}")
print("-" * 80)
for token in doc:
    print(f"{token.text:<20} {token.lemma_:<20} {token.pos_:<10} {token.dep_:<15} {token.head.text}")

print("\n" + "=" * 80)
print("DEPENDENCY TREE (indented)")
print("=" * 80)

def print_tree(token, indent=0):
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    print(f"{prefix}{token.text!r:<20} [{token.dep_}]  pos={token.pos_}  lemma={token.lemma_!r}")
    for child in token.children:
        print_tree(child, indent + 1)

for token in doc:
    if token.dep_ == "ROOT":
        print_tree(token)

print("\n" + "=" * 80)
print("NOUN CHUNKS")
print("=" * 80)
for chunk in doc.noun_chunks:
    print(f"  {chunk.text!r:<30} root={chunk.root.text!r}  dep={chunk.root.dep_}")
