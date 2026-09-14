"""
Configuration for visualization output paths.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so NEO4J_PASSWORD / NEO4J_URI etc. are picked up from the file.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import (  # noqa: F401  (re-exported for the visualization package)
    DATA_DIR,
    RE_OUTPUT_CSV,       # re_results.csv (actual RE output)
    NER_OUTPUT_CSV,      # ner_results.csv (source of facts + provenance)
    GRAPH_MODE,
    GRAPH_RELATION_THRESHOLD,
)

# Neo4j import output files
NEO4J_NODES_CSV = "visualization/nodes.csv"
NEO4J_RELS_CSV = "visualization/relationships.csv"

# FHIR-reification graph outputs (GRAPH_MODE="fhir").
FHIR_PATIENTS_CSV = "visualization/fhir_patients.csv"
FHIR_CONCEPTS_CSV = "visualization/fhir_concepts.csv"
FHIR_FACTS_CSV    = "visualization/fhir_facts.csv"
FHIR_EDGES_CSV    = "visualization/fhir_edges.csv"

# Neo4j connection settings (overridable via environment variables).
# Password is read from the environment only — never hardcode secrets here.
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")  # set: export/$env NEO4J_PASSWORD=...
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# Minimum relation confidence threshold (concept mode only) — value lives in config.py
RELATION_THRESHOLD = GRAPH_RELATION_THRESHOLD

# Entity label mapping to Neo4j-friendly names
ENTITY_LABEL_MAP = {
    "BIOLOGICAL_STRUCTURE": "BiologicalStructure",
    "DISEASE_DISORDER": "DiseaseDisorder",
    "SIGN_SYMPTOM": "SignSymptom",
    "DETAILED_DESCRIPTION": "DetailedDescription",
    "MEDICAL_DEVICE": "MedicalDevice",
    "MEDICATION": "Medication",
    "PROCEDURE": "Procedure",
    "LAB_VALUE": "LabValue",
}