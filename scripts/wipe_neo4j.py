"""
Empty the Neo4j database — and nothing else.

Usage:
    python scripts/wipe_neo4j.py

`visualization/neo4j_integration.py --wipe` also clears the graph, but then
immediately imports the CSVs; this is the clean-slate half on its own, for when
you want an empty database before showing something, or after a demo.

Connection settings come from .env, the same ones every other entry point reads.
DETACH DELETE removes every node and relationship — the database itself, its
users and its indexes stay. The delete runs in batches so a full-corpus graph
does not have to fit in one transaction.

It does not ask for confirmation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase

from visualization.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
)


def _count(session, query: str) -> int:
    return session.run(query).single()["c"]


def main() -> None:
    if not NEO4J_PASSWORD:
        print("[wipe] no NEO4J_PASSWORD — set it in .env")
        sys.exit(1)

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            with driver.session(database=NEO4J_DATABASE) as session:
                nodes = _count(session, "MATCH (n) RETURN count(n) AS c")
                rels = _count(session, "MATCH ()-[r]->() RETURN count(r) AS c")
                if not nodes:
                    print(f"[wipe] {NEO4J_DATABASE} at {NEO4J_URI} is already empty")
                    return

                print(f"[wipe] deleting {nodes} nodes and {rels} relationships "
                      f"from {NEO4J_DATABASE} at {NEO4J_URI} ...")
                # Autocommit (session.run) — CALL {} IN TRANSACTIONS cannot run
                # inside an explicit transaction.
                session.run("CALL () { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
                left = _count(session, "MATCH (n) RETURN count(n) AS c")
                print(f"[wipe] done - {left} nodes left")
    except Exception as exc:
        print(f"[wipe] FAILED: {type(exc).__name__}: {exc}")
        print(f"  -> is Neo4j running at {NEO4J_URI}? check NEO4J_USERNAME/NEO4J_PASSWORD too.")
        sys.exit(1)


if __name__ == "__main__":
    main()
