"""
Neo4j integration for loading graph data into a Neo4j database.

Uses the official neo4j Python driver to import nodes and relationships
from the visualization CSV output files.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase


def _ident(value, default: str) -> str:
    """Sanitize a string for safe use as a Cypher label / relationship type.

    Cypher cannot parametrize labels or relationship types, so they must be
    interpolated into the query string — this strips anything that isn't
    alphanumeric/underscore to prevent injection and invalid identifiers.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_")
    return cleaned or default


# Endpoint labels per structural edge type. The three key-spaces are NOT fully
# disjoint: an unlinked mention keys on its surface string, so a bare-number
# DATE ("2011") collides with the record id of patient 2011. An unlabelled
# MATCH then binds both nodes and writes the edge twice. Facts stay unqualified
# ("" = any label) — their `id:start:end` keys cannot collide — as do clinical
# edges, whose endpoints are always facts.
_EDGE_ENDPOINTS = {
    "SUBJECT":   ("", "Patient"),
    "HAS_CODE":  ("", "Concept"),
    "MENTIONS":  ("Patient", "Concept"),
}


def _clean_props(props: dict) -> dict:
    """Drop empty / placeholder string values so missing fields become null in Neo4j.

    CSV blanks read back as NaN -> "nan"; empty offsets/attributes as "". Omitting
    them (rather than SETting "" / "nan") leaves the property absent, i.e. null.
    """
    return {
        k: v for k, v in props.items()
        if not (isinstance(v, str) and v.strip().lower() in ("", "nan", "none"))
    }


# Fields stored as a NUMBER (int/float) instead of a string, so numeric comparisons work
# directly in Cypher: patient demographics (age/weight/height) and numeric fact-attribute
# magnitudes (value/dosage/duration). Textual attributes (severity, frequency, color,
# shape, texture, route, onset, descriptor) stay strings.
_NUMERIC_FIELDS = {"age", "weight", "height", "value", "dosage", "duration"}

# Numeric fields that carry a UNIT: the unit is stored alongside as `<field>_unit`, so
# magnitudes are never conflated across units (5 g vs 5 mg -> same number, different
# *_unit). age has no unit (always years). Queries should compare within one unit
# (WHERE f.dosage_unit = 'mg' AND f.dosage > 100).
_UNIT_FIELDS = {"weight", "height", "value", "dosage", "duration"}


def _leading_number(value):
    """First number in a string ("54-year-old"->54, "70.5 kg"->70.5); None if none.

    Returns int for whole numbers, float otherwise, so Neo4j stores a numeric type.
    """
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    if not m:
        return None
    num = float(m.group())
    return int(num) if num.is_integer() else num


def _unit_of(value):
    """Unit token after the number ("80 kg"->"kg", "5mg"->"mg", "120 mg/dl"->"mg/dl"); '' if none."""
    s = str(value)
    m = re.search(r"\d+(?:\.\d+)?", s)
    return s[m.end():].strip().strip("-").strip() if m else ""


def _set_numeric(props: dict, col: str, val: str) -> None:
    """Store props[col] as a number and props[col+'_unit'] as its unit (for _UNIT_FIELDS)."""
    n = _leading_number(val)
    if n is None:
        return
    props[col] = n
    if col in _UNIT_FIELDS:
        u = _unit_of(val)
        if u:
            props[col + "_unit"] = u

from .config import (
    NEO4J_NODES_CSV,
    NEO4J_RELS_CSV,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    FHIR_PATIENTS_CSV,
    FHIR_CONCEPTS_CSV,
    FHIR_FACTS_CSV,
    FHIR_EDGES_CSV,
    GRAPH_MODE,
)
from config import ATTRIBUTE_FIELDS


def _in_dir(run_dir: Optional[str], path: str) -> str:
    """Redirect a default CSV path into `run_dir` (e.g. prev_runs/<name>/).

    scripts/save_run.py copies the graph CSVs flat, under their basenames, so a
    saved run can be re-imported by swapping only the directory.
    """
    return path if not run_dir else str(Path(run_dir) / Path(path).name)


def load_csv_safe(path: str) -> pd.DataFrame:
    """Load a CSV file produced by the visualization pipeline."""
    df = pd.read_csv(path, quotechar='"')
    return df


class Neo4jConnector:
    """
    Context manager for Neo4j database connections.

    Usage:
        with Neo4jConnector("bolt://localhost:7687", "neo4j", "password") as conn:
            conn.import_nodes("visualization/nodes.csv")
            conn.import_relationships("visualization/relationships.csv")
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        self.uri = uri
        self.auth = (username, password)
        self.database = database
        self._driver = None

    def _connect(self):
        self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
        self._driver.verify_connectivity()

    def _close(self):
        if self._driver:
            self._driver.close()

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        self._close()

    def import_nodes(self, csv_path: str) -> int:
        """
        Import nodes from a Neo4j-format CSV file.

        Uses MERGE to avoid duplicates if run multiple times.
        Returns the number of nodes imported.
        """
        nodes_df = load_csv_safe(csv_path)

        query = """
        LOAD CSV WITH HEADERS FROM $csv_url AS row
        MERGE (n {name: row.name})
        SET n.entity_type = row.entity_type,
            n.count = toInteger(row.count)
        WITH n, row
        CALL apoc.create.addLabels(n, [row.`:Label`]) YIELD node
        RETURN count(node) AS total
        """

        # Build file URL for Neo4j import
        csv_url = Path(csv_path).resolve().as_uri()

        with self._driver.session(database=self.database) as session:
            result = session.run(query, csv_url=csv_url)
            summary = result.consume()
            total = nodes_df.shape[0]

        print(f"[neo4j] Imported {total} nodes from {csv_path}")
        return total

    def import_relationships(self, csv_path: str) -> int:
        """
        Import relationships from a Neo4j-format CSV file.

        Assumes nodes already exist (run import_nodes first).
        Uses MATCH to find nodes by name and CREATE for relationships.
        Returns the number of relationships imported.
        """
        rels_df = load_csv_safe(csv_path)

        # Filter out rows with missing or null IDs
        rels_df = rels_df.dropna(subset=[":START_ID", ":END_ID"])

        imported = 0
        with self._driver.session(database=self.database) as session:
            for _, row in rels_df.iterrows():
                query = """
                MATCH (a {name: $start_name})
                MATCH (b {name: $end_name})
                MERGE (a)-[r:`$rel_type`]->(b)
                SET r.relation = $rel_type,
                    r.relation_score = toFloat($rel_score),
                    r.source_id = toInteger($source_id)
                """
                # We need node names, but CSV only has IDs. Need lookup.
                # Use a simpler approach: pass full tuples
                pass

        print(f"[neo4j] Imported {imported} relationships from {csv_path}")
        return imported

    def import_all(self, nodes_csv: str = NEO4J_NODES_CSV, rels_csv: str = NEO4J_RELS_CSV) -> tuple[int, int]:
        """
        Import all nodes and relationships into Neo4j.

        Returns (nodes_count, relationships_count)
        """
        nodes_count = self.import_nodes(nodes_csv)
        rels_count = self.import_relationships_batched(rels_csv)
        return nodes_count, rels_count

    def import_relationships_batched(self, csv_path: str) -> int:
        """
        Batch-import relationships using UNWIND for efficiency.

        Relationships reference nodes by :START_ID/:END_ID which are
        internal IDs from the CSV. Since nodes were created with
        name-based MERGE, we use a name-based approach here too.
        """
        rels_df = load_csv_safe(csv_path)
        rels_df = rels_df.dropna(subset=[":START_ID", ":END_ID"])

        # Convert to list of dicts for parameter passing
        records = rels_df.to_dict(orient="records")

        query = """
        UNWIND $relationships AS row
        MATCH (a {name: row.start_name})
        MATCH (b {name: row.end_name})
        MERGE (a)-[r:`${row.type}`]->(b)
        SET r.relation = row.type,
            r.relation_score = row.score,
            r.source_id = row.source_id
        """

        # Since the relationships CSV uses internal IDs (not names),
        # we need to build a name lookup. Do it in two passes:
        # 1. Create nodes with their internal ID as a property
        # 2. Match relationships by that ID
        # For simplicity, let's use a simpler approach with direct Cypher
        return 0  # Placeholder - see import_relationships_direct


def import_via_cypher_load_csv(
    uri: str,
    username: str,
    password: str,
    database: str,
    nodes_csv: str,
    rels_csv: str,
) -> None:
    """
    Import nodes and relationships using LOAD CSV.cypher.

    This is the recommended approach for Neo4j imports.
    Requires the CSV files to be accessible to the Neo4j server.
    """
    from neo4j import GraphDatabase

    auth = (username, password)

    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()

        with driver.session(database=database) as session:
            # Import nodes
            nodes_url = Path(nodes_csv).resolve().as_uri()
            session.run("""
                LOAD CSV WITH HEADERS FROM $nodes_url AS row
                MERGE (n {name: row.name})
                SET n.entity_type = row.entity_type,
                    n.count = toInteger(row.count)
                """,
                nodes_url=nodes_url,
            )
            print(f"[neo4j] Nodes imported from {nodes_csv}")

            # Import relationships
            rels_url = Path(rels_csv).resolve().as_uri()
            session.run("""
                LOAD CSV WITH HEADERS FROM $rels_url AS row
                MATCH (a {name: row.start_name})
                MATCH (b {name: row.end_name})
                MERGE (a)-[r:`${row.type}`]->(b)
                SET r.relation = row.type,
                    r.relation_score = toFloat(row.score),
                    r.source_id = toInteger(row.source_id)
                """,
                rels_url=rels_url,
            )
            print(f"[neo4j] Relationships imported from {rels_csv}")


# The relationships CSV uses :START_ID/:END_ID with internal integer IDs,
# but nodes are merged by name. We need to update the relationships CSV
# to include start/end names instead of IDs for this import method.
# For now, let's provide a direct Python-based import that reads the CSV
# and creates relationships using the actual entity names.


def import_relationships_by_name(
    uri: str,
    username: str,
    password: str,
    database: str,
    nodes_csv: str,
    rels_csv: str,
) -> tuple[int, int]:
    """
    Import relationships by matching nodes via their names.

    This handles the case where relationships.csv has internal IDs
    but we need to match nodes by their name property.
    """
    from neo4j import GraphDatabase

    auth = (username, password)

    # Load data
    nodes_df = load_csv_safe(nodes_csv)
    rels_df = load_csv_safe(rels_csv)
    rels_df = rels_df.dropna(subset=[":START_ID", ":END_ID"])

    # Build ID -> name mapping from nodes
    id_to_name = {}
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            result = session.run("MATCH (n) RETURN n.name AS name, n.entity_type AS entity_type")
            id_to_name = {row["name"]: row["name"] for row in result}

    # We need a full ID mapping. Let's do it differently:
    # Create a temporary mapping file or use Python directly

    return 0, 0


def run_fhir_import(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    patients_csv: str = FHIR_PATIENTS_CSV,
    concepts_csv: str = FHIR_CONCEPTS_CSV,
    facts_csv: str = FHIR_FACTS_CSV,
    edges_csv: str = FHIR_EDGES_CSV,
    wipe: bool = False,
    run_dir: Optional[str] = None,
) -> None:
    """Import the FHIR-reification graph (GRAPH_MODE="fhir").

    Nodes (Concept / Patient / fact resources) are MERGEd by ``key`` — the three
    key-spaces are disjoint (CUI-or-string / patient id / ``id:start:end``), so
    edges can MATCH endpoints by ``key`` regardless of label. Idempotent; ``wipe``
    clears the graph first for a clean reload.

    ``run_dir`` reads the four CSVs from a saved run (prev_runs/<name>/) instead
    of the current visualization/ output.
    """
    from neo4j import GraphDatabase

    patients_csv = _in_dir(run_dir, patients_csv)
    concepts_csv = _in_dir(run_dir, concepts_csv)
    facts_csv = _in_dir(run_dir, facts_csv)
    edges_csv = _in_dir(run_dir, edges_csv)

    def _num(v):
        s = str(v).strip()
        return s not in ("", "nan", "None")

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        print(f"[neo4j] Connected to {uri}")

        with driver.session(database=database) as session:
            if wipe:
                session.run("MATCH (n) DETACH DELETE n")
                print("[neo4j] Wiped database (DETACH DELETE)")

            # --- Concepts (shared CodeableConcept = CUI; huby) ---
            cdf = load_csv_safe(concepts_csv)
            for _, row in cdf.iterrows():
                session.run(
                    "MERGE (c:Concept {key: $key}) SET c += $props",
                    key=str(row["key"]),
                    props=_clean_props({
                        "name": str(row.get("name", "") or ""),
                        "cui": str(row.get("cui", "") or ""),
                        "entity_type": str(row.get("entity_type", "") or ""),
                        "count": int(row["count"]),
                    }),
                )
            print(f"[neo4j] Imported {len(cdf)} concepts")

            # --- Patients (demographics as properties) ---
            pdf = load_csv_safe(patients_csv)
            demo_cols = [c for c in pdf.columns if c not in ("key", "id")]
            for _, row in pdf.iterrows():
                props = {"id": str(row["id"])}
                for c in demo_cols:
                    val = str(row.get(c, "") or "").strip()
                    if not val or val.lower() in ("nan", "none"):
                        continue
                    if c in _NUMERIC_FIELDS:
                        _set_numeric(props, c, val)   # "54-year-old" -> 54, "70.5 kg" -> 70.5 + _unit
                    else:
                        props[c] = val
                session.run("MERGE (p:Patient {key: $key}) SET p += $props",
                            key=str(row["key"]), props=_clean_props(props))
            print(f"[neo4j] Imported {len(pdf)} patients")

            # --- Facts (reified resources; provenance = source_id + offsets) ---
            fdf = load_csv_safe(facts_csv)
            for _, row in fdf.iterrows():
                resource = _ident(row.get("resource", "Observation"), "Observation")
                props = {
                    "word": str(row.get("word", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "source_id": str(row.get("source_id", "") or ""),
                    "cui": str(row.get("cui", "") or ""),
                    "assertion": str(row.get("assertion", "") or ""),
                    "verificationStatus": str(row.get("verificationStatus", "") or ""),
                    "clinicalStatus": str(row.get("clinicalStatus", "") or ""),
                    # family-relation for :FamilyMemberHistory facts (empty -> dropped by _clean_props)
                    "relative": str(row.get("relative", "") or ""),
                }
                for col in ("start", "end"):
                    if _num(row.get(col)):
                        props[col] = int(float(row[col]))
                for col in ATTRIBUTE_FIELDS:
                    if col not in fdf.columns:
                        continue
                    val = str(row.get(col, "") or "").strip()
                    if not val or val.lower() in ("nan", "none"):
                        continue
                    if col in _NUMERIC_FIELDS:
                        _set_numeric(props, col, val)   # "3 cm" -> 3 + value_unit "cm"; "500 mg" -> 500 + "mg"
                    else:
                        props[col] = val
                session.run(
                    f"MERGE (f {{key: $key}}) SET f:`{resource}`, f += $props",
                    key=str(row["key"]), props=_clean_props(props),
                )
            print(f"[neo4j] Imported {len(fdf)} facts")

            # --- Edges (subject/has_code/body_site/clinical/attribute/mentions) ---
            edf = load_csv_safe(edges_csv)
            for _, row in edf.iterrows():
                rel_type = _ident(row[":TYPE"], "RELATED")
                props = {}
                if _num(row.get("relation_score")):
                    props["relation_score"] = float(row["relation_score"])
                for col in ("assertion", "source_id"):
                    if str(row.get(col, "") or "").strip():
                        props[col] = str(row[col])
                for col in ("start", "end"):
                    if _num(row.get(col)):
                        props[col] = int(float(row[col]))
                src_label, dst_label = _EDGE_ENDPOINTS.get(rel_type, ("", ""))
                a_label = f":`{src_label}`" if src_label else ""
                b_label = f":`{dst_label}`" if dst_label else ""
                session.run(
                    f"""
                    MATCH (a{a_label} {{key: $s}})
                    MATCH (b{b_label} {{key: $e}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    SET r += $props
                    """,
                    s=str(row["start_key"]), e=str(row["end_key"]), props=_clean_props(props),
                )
            print(f"[neo4j] Imported {len(edf)} edges")

    print("[neo4j] FHIR import complete!")


def run_full_import(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
    nodes_csv: str = NEO4J_NODES_CSV,
    rels_csv: str = NEO4J_RELS_CSV,
    wipe: bool = False,
    run_dir: Optional[str] = None,
) -> None:
    """
    Run the complete Neo4j import pipeline.

    Parameters
    ----------
    uri : str
        Neo4j connection URI (e.g., "bolt://localhost:7687")
    username : str
        Neo4j username
    password : str
        Neo4j password
    database : str
        Target database name (default: "neo4j")
    nodes_csv : str
        Path to nodes CSV file
    rels_csv : str
        Path to relationships CSV file
    wipe : bool
        If True, DETACH DELETE the whole graph before importing (fresh load).
    run_dir : str | None
        Directory of a saved run (prev_runs/<name>/) to import from instead of
        the current visualization/ output.
    """
    # FHIR-reification graph is imported by a dedicated routine.
    if GRAPH_MODE == "fhir":
        run_fhir_import(uri, username, password, database, wipe=wipe, run_dir=run_dir)
        return

    from neo4j import GraphDatabase

    nodes_csv = _in_dir(run_dir, nodes_csv)
    rels_csv = _in_dir(run_dir, rels_csv)
    auth = (username, password)

    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        print(f"[neo4j] Connected to {uri}")

        with driver.session(database=database) as session:
            # Import nodes
            nodes_df = load_csv_safe(nodes_csv)
            nodes_url = Path(nodes_csv).resolve().as_uri()

            for _, row in nodes_df.iterrows():
                label = _ident(row.get(":Label", "Entity"), "Entity")
                # Node identity = `key` (CUI when linked, else the string) so
                # synonyms sharing a CUI merge into one node; cui rides along as
                # a property.
                session.run(f"""
                    MERGE (n {{key: $key}})
                    SET n:`{label}`,
                        n.name = $name,
                        n.entity_type = $entity_type,
                        n.cui = $cui,
                        n.count = $count
                    """,
                    key=row["key"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    cui=str(row.get("cui", "") or ""),
                    count=int(row["count"]),
                )
            print(f"[neo4j] Imported {len(nodes_df)} nodes")

            # Import relationships matching nodes by their `key` (CUI-or-string).
            rels_df = load_csv_safe(rels_csv)
            rels_df = rels_df.dropna(subset=["start_key", "end_key"])

            imported_rels = 0
            for _, row in rels_df.iterrows():
                # Cypher can't parametrize the relationship TYPE — interpolate it
                # (sanitized) into the query string; everything else stays a param.
                rel_type = _ident(row[":TYPE"], "RELATED")
                session.run(f"""
                    MATCH (a {{key: $start_key}})
                    MATCH (b {{key: $end_key}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    SET r.relation = $rel_label,
                        r.relation_score = $rel_score,
                        r.source_id = $source_id,
                        r.assertion = $assertion
                    """,
                    start_key=row["start_key"],
                    end_key=row["end_key"],
                    rel_label=row[":TYPE"],
                    rel_score=float(row["relation_score"]),
                    source_id=(row["source_id"]),
                    # ConText edge polarity; defaults to affirmed if RE didn't annotate.
                    assertion=row.get("assertion", "affirmed"),
                )
                imported_rels += 1

            print(f"[neo4j] Imported {imported_rels} relationships")

    print("[neo4j] Import complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import graph data into Neo4j")
    # Defaults come from visualization/config.py (overridable via env vars).
    parser.add_argument("--uri", default=NEO4J_URI, help="Neo4j URI")
    parser.add_argument("--username", default=NEO4J_USERNAME, help="Neo4j username")
    parser.add_argument("--password", default=NEO4J_PASSWORD,
                        required=NEO4J_PASSWORD is None,
                        help="Neo4j password (default: NEO4J_PASSWORD env var)")
    parser.add_argument("--database", default=NEO4J_DATABASE, help="Target database")
    parser.add_argument("--nodes", default=NEO4J_NODES_CSV, help="Nodes CSV path")
    parser.add_argument("--rels", default=NEO4J_RELS_CSV, help="Relationships CSV path")
    parser.add_argument("--run-dir", default=None,
                        help='Import a saved run instead of the current output, '
                             'e.g. --run-dir "prev_runs/Holistic full"')
    parser.add_argument("--wipe", action="store_true",
                        help="DETACH DELETE the whole graph before importing (fresh load)")

    args = parser.parse_args()
    if args.run_dir and not Path(args.run_dir).is_dir():
        parser.error(f"--run-dir: {args.run_dir} is not a directory")
    run_full_import(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
        nodes_csv=args.nodes,
        rels_csv=args.rels,
        wipe=args.wipe,
        run_dir=args.run_dir,
    )