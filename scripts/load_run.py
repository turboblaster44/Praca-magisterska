"""
Load an archived run from prev_runs/<run name>/ into Neo4j.

Usage:
    python scripts/load_run.py                 # list what is in prev_runs/
    python scripts/load_run.py "run name"      # wipe the graph, then import that run

The counterpart of scripts/save_run.py: that one copies a finished run out of
data/ and visualization/, this one puts one back into the database, so an old
run can be queried again without re-extracting anything.

The database is WIPED first — an archived run is a complete graph, and merging
it into whatever is already loaded would mix two corpora. Use
scripts/wipe_neo4j.py if you only want the database emptied.

Which CSVs are read depends on GRAPH_MODE in config.py: the four fhir_*.csv in
"fhir" mode, nodes.csv + relationships.csv in "concept" mode. A run archived in
one mode cannot be loaded in the other, so the script checks before connecting.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visualization.config import (
    GRAPH_MODE,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    NEO4J_NODES_CSV, NEO4J_RELS_CSV,
    FHIR_PATIENTS_CSV, FHIR_CONCEPTS_CSV, FHIR_FACTS_CSV, FHIR_EDGES_CSV,
)

PREV_RUNS = ROOT / "prev_runs"

# The graph CSVs each mode imports, by basename — save_run.py copies them flat.
REQUIRED = {
    "fhir": [Path(p).name for p in
             (FHIR_PATIENTS_CSV, FHIR_CONCEPTS_CSV, FHIR_FACTS_CSV, FHIR_EDGES_CSV)],
    "concept": [Path(p).name for p in (NEO4J_NODES_CSV, NEO4J_RELS_CSV)],
}


def list_runs() -> None:
    """Print every archived run and the graph mode it can be loaded in."""
    runs = sorted(d for d in PREV_RUNS.iterdir() if d.is_dir()) if PREV_RUNS.is_dir() else []
    if not runs:
        print(f"[load] no archived runs in {PREV_RUNS}")
        return
    print(f"Runs in {PREV_RUNS}:")
    for d in runs:
        modes = [m for m, files in REQUIRED.items()
                 if all((d / f).is_file() for f in files)]
        print(f"  {d.name:<28}{', '.join(modes) if modes else 'no graph CSVs'}")
    print('\nLoad one with: python scripts/load_run.py "<run name>"')


def main() -> None:
    if len(sys.argv) < 2:
        list_runs()
        sys.exit(0)

    run_dir = PREV_RUNS / sys.argv[1]
    if not run_dir.is_dir():
        print(f"[load] {run_dir} is not a directory\n")
        list_runs()
        sys.exit(1)

    missing = [f for f in REQUIRED[GRAPH_MODE] if not (run_dir / f).is_file()]
    if missing:
        print(f"[load] {run_dir.name} has no {GRAPH_MODE}-mode graph: missing "
              f"{', '.join(missing)}")
        other = "concept" if GRAPH_MODE == "fhir" else "fhir"
        if all((run_dir / f).is_file() for f in REQUIRED[other]):
            print(f"  -> it was archived in {other!r} mode; set GRAPH_MODE = {other!r} "
                  f"in config.py to load it.")
        sys.exit(1)

    if not NEO4J_PASSWORD:
        print("[load] no NEO4J_PASSWORD — set it in .env")
        sys.exit(1)

    print(f"[load] {run_dir.name} ({GRAPH_MODE} mode) -> {NEO4J_URI}")
    from visualization.neo4j_integration import run_full_import
    try:
        run_full_import(
            uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE, wipe=True, run_dir=str(run_dir),
        )
    except Exception as exc:
        print(f"[load] FAILED: {type(exc).__name__}: {exc}")
        print(f"  -> is Neo4j running at {NEO4J_URI}? check NEO4J_USERNAME/NEO4J_PASSWORD too.")
        sys.exit(1)


if __name__ == "__main__":
    main()
