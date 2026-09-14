"""
Visualization pipeline: converts relation extraction output to Neo4j import format.

Neo4j accepts CSV files for bulk import via `LOAD CSV` or `neo4j-admin db_import`.
This pipeline produces two CSV files:
  - nodes.csv    → all unique entities as nodes
  - relationships.csv → entity pairs with relation type as edges
"""

import pandas as pd
from pathlib import Path

from .config import (
    RE_OUTPUT_CSV, NEO4J_NODES_CSV, NEO4J_RELS_CSV, ENTITY_LABEL_MAP,
    RELATION_THRESHOLD, GRAPH_MODE,
)


def _normalize_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any RE output schema to what the visualizer expects.

    Accepts both:
      - biolink/spaCy (nlp.re.tasks): entity1_label / entity2_label / score
      - decoder (Gemini):             label1 / label2 / confidence
    Produces canonical columns: entity1, label1, entity2, label2, relation,
    relation_score, id (+ assertion if present).
    """
    df = df.copy()
    rename = {}
    if "label1" not in df.columns and "entity1_label" in df.columns:
        rename["entity1_label"] = "label1"
    if "label2" not in df.columns and "entity2_label" in df.columns:
        rename["entity2_label"] = "label2"
    df = df.rename(columns=rename)

    if "relation_score" not in df.columns:
        if "score" in df.columns:
            df["relation_score"] = df["score"]
        elif "confidence" in df.columns:
            df["relation_score"] = df["confidence"]
        else:
            df["relation_score"] = 1.0

    # UMLS per-endpoint columns (from RE annotate_umls) — default "" if absent
    # (e.g. RE run without linking) so downstream keys fall back to the string.
    for side in ("entity1", "entity2"):
        for f in ("cui", "canonical_name"):
            col = f"{side}_{f}"
            if col not in df.columns:
                df[col] = ""
    return df


def _node_key(entity: str, cui) -> str:
    """Graph node identity: the UMLS CUI when linked, else the raw string.

    This is what collapses synonyms ("aspirin"/"ASA" -> C0004057) while keeping
    unlinked entities as distinct string-keyed nodes.
    """
    cui = "" if (cui is None or pd.isna(cui)) else str(cui).strip()
    return cui if cui else str(entity)


def _display_name(entity: str, cui, canonical) -> str:
    """Human-readable node name: canonical UMLS name when linked, else the string."""
    cui = "" if (cui is None or pd.isna(cui)) else str(cui).strip()
    if cui and isinstance(canonical, str) and canonical.strip():
        return canonical.strip()
    return str(entity)


def load_relations(csv_path: str) -> pd.DataFrame:
    """Load the relation extraction results and normalize the schema."""
    return _normalize_relations(pd.read_csv(csv_path))


def _endpoint_frame(df: pd.DataFrame, ent: str, lab: str) -> pd.DataFrame:
    """One endpoint (entity1 or entity2) as: key, name, entity_type, cui."""
    sub = pd.DataFrame({
        "entity": df[ent].astype(str),
        "entity_type": df[lab],
        "cui": df[f"{ent}_cui"],
        "canonical_name": df[f"{ent}_canonical_name"],
    })
    sub["key"] = [_node_key(e, c) for e, c in zip(sub["entity"], sub["cui"])]
    sub["name"] = [_display_name(e, c, cn)
                   for e, c, cn in zip(sub["entity"], sub["cui"], sub["canonical_name"])]
    return sub


def extract_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique entities as Neo4j nodes, MERGED BY CUI (fallback: string).

    Returns columns: entity_id, key, name, label, entity_type, cui, count.
    `key` is the node identity (CUI when linked, else the string) — synonyms sharing
    a CUI collapse into one node; unlinked entities stay string-keyed.
    """
    all_entities = pd.concat(
        [_endpoint_frame(df, "entity1", "label1"),
         _endpoint_frame(df, "entity2", "label2")],
        ignore_index=True,
    )

    # Aggregate by node identity. name/type taken as the first (most common
    # would be nicer, but first is stable and cheap); count = mentions.
    node_df = (
        all_entities
        .groupby("key")
        .agg(
            name=("name", "first"),
            entity_type=("entity_type", "first"),
            cui=("cui", "first"),
            count=("key", "size"),
        )
        .reset_index()
    )
    node_df.insert(0, "entity_id", range(1, len(node_df) + 1))
    node_df["label"] = node_df["entity_type"].map(lambda t: ENTITY_LABEL_MAP.get(t, t))
    node_df["cui"] = node_df["cui"].fillna("").astype(str)

    return node_df[["entity_id", "key", "name", "label", "entity_type",
                    "cui", "count"]]


def extract_relationships(df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract entity pairs as Neo4j relationships, referencing nodes by CUI-or-string key.

    Returns: start_id, end_id, start_key, end_key, relation, relation_score,
             source_id (+ assertion when present).
    """
    filtered = df[df["relation_score"] >= RELATION_THRESHOLD].copy()

    # Endpoint keys must match the node identity used in extract_nodes.
    filtered["start_key"] = [_node_key(e, c)
                             for e, c in zip(filtered["entity1"], filtered["entity1_cui"])]
    filtered["end_key"] = [_node_key(e, c)
                           for e, c in zip(filtered["entity2"], filtered["entity2_cui"])]

    key_to_id = dict(zip(nodes_df["key"], nodes_df["entity_id"]))
    filtered["start_id"] = filtered["start_key"].map(key_to_id)
    filtered["end_id"] = filtered["end_key"].map(key_to_id)
    filtered = filtered.dropna(subset=["start_id", "end_id"])
    filtered["start_id"] = filtered["start_id"].astype(int)
    filtered["end_id"] = filtered["end_id"].astype(int)

    cols = ["start_id", "end_id", "start_key", "end_key", "relation", "relation_score", "id"]
    if "assertion" in filtered.columns:
        cols.append("assertion")
    rel_df = filtered[cols].copy()
    rel_df.rename(columns={"id": "source_id"}, inplace=True)
    return rel_df


def save_nodes_csv(nodes_df: pd.DataFrame, output_path: str):
    """
    Save nodes in Neo4j-compatible CSV format.

    Neo4j expects: :ID, name, :Label, entity_type, count
    The :ID column is the node identifier, :Label marks the node label.
    """
    out = pd.DataFrame({
        ":ID": nodes_df["entity_id"],
        "key": nodes_df["key"],            # CUI-or-string identity (MERGE key)
        "name": nodes_df["name"],          # display (canonical UMLS name or string)
        ":Label": nodes_df["label"],
        "entity_type": nodes_df["entity_type"],
        "cui": nodes_df["cui"],
        "count": nodes_df["count"],
    })
    out.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL to handle special chars
    print(f"[visualization] Saved {len(out)} nodes to {output_path}")


def save_rels_csv(rel_df: pd.DataFrame, output_path: str):
    """
    Save relationships in Neo4j-compatible CSV format.

    Neo4j expects: :START_ID, :END_ID, :TYPE, start_name, end_name, relation,
                   relation_score, source_id
    """
    out = pd.DataFrame({
        ":START_ID": rel_df["start_id"],
        ":END_ID": rel_df["end_id"],
        ":TYPE": rel_df["relation"],
        "start_key": rel_df["start_key"],
        "end_key": rel_df["end_key"],
        "relation": rel_df["relation"],
        "relation_score": rel_df["relation_score"],
        "source_id": rel_df["source_id"],
    })
    # ConText edge assertion (graph edge polarity), when present.
    if "assertion" in rel_df.columns:
        out["assertion"] = rel_df["assertion"]
    out.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL to handle special chars
    print(f"[visualization] Saved {len(out)} relationships to {output_path}")


def run_visualization_pipeline(
    relations_csv: str = RE_OUTPUT_CSV,
    nodes_csv: str = NEO4J_NODES_CSV,
    rels_csv: str = NEO4J_RELS_CSV,
) -> tuple[str, str]:
    """
    Run the full visualization pipeline.

    Parameters
    ----------
    relations_csv : str
        Path to the relation extraction output (entity1, entity2, relation, ...)
    nodes_csv : str
        Output path for nodes CSV
    rels_csv : str
        Output path for relationships CSV

    Returns
    -------
    tuple[str, str]
        Paths to the generated nodes and relationships CSV files.
    """
    # FHIR-reification graph (GRAPH_MODE="fhir") is built from NER + RE by a
    # separate module; the concept-graph CSVs below are not produced in that mode.
    if GRAPH_MODE == "fhir":
        from .fhir import run_fhir_pipeline  # lazy: fhir imports helpers from here
        run_fhir_pipeline(re_csv=relations_csv)
        return nodes_csv, rels_csv

    # Ensure output directory exists
    Path(nodes_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load relation data
    df = load_relations(relations_csv)
    print(f"[visualization] Loaded {len(df)} relation records")

    # Extract nodes
    nodes_df = extract_nodes(df)
    save_nodes_csv(nodes_df, nodes_csv)

    # Extract relationships
    rel_df = extract_relationships(df, nodes_df)
    save_rels_csv(rel_df, rels_csv)

    return nodes_csv, rels_csv


if __name__ == "__main__":
    nodes_path, rels_path = run_visualization_pipeline()
    print("\n=== Neo4j Import Instructions ===")
    print(f"Nodes file:   {nodes_path}")
    print(f"Relationships file: {rels_path}")
    print("\nOption 1 - USING CYPHER (small datasets):")
    print("  LOAD CSV WITH HEADERS FROM 'file:///{nodes_path}' AS row")
    print("  CREATE (n:{row.Label} {name: row.name});")
    print("\nOption 2 - USING NEO4J-ADMIN (large datasets):")
    print("  neo4j-admin db_import --nodes={nodes_path} --relationships={rels_path}")