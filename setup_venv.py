#!/usr/bin/env python3
import subprocess
import sys
import os
from config import MARIANMT_MODEL_PL_EN, BIOLINK_BERT_MODEL, SPACY_MODEL_ENGLISH

# -------------------------------
# Configuration
# -------------------------------
VENV_DIR = ".venv"
PYTHON = sys.executable

# -------------------------------
# Helper Functions
# -------------------------------
def run(cmd_list):
    """Run a subprocess command safely as a list."""
    print(">>> Running:", " ".join(cmd_list))
    subprocess.check_call(cmd_list)

# -------------------------------
# 1️⃣ Create virtual environment
# -------------------------------
if not os.path.exists(VENV_DIR):
    print(f"Creating virtual environment at {VENV_DIR}")
    run([PYTHON, "-m", "venv", VENV_DIR])
else:
    print(f"Virtual environment {VENV_DIR} already exists")

# Determine pip and python paths inside venv
pip_bin = os.path.join(VENV_DIR, "Scripts", "pip") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "pip")
python_bin = os.path.join(VENV_DIR, "Scripts", "python") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")

# -------------------------------
# 2️⃣ Upgrade pip first
# -------------------------------

run([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

# -------------------------------
# 3️⃣ Install critical packages first to avoid conflicts
# -------------------------------
run([pip_bin, "install", "pydantic>=2.10.1,<3.0.0"])
run([pip_bin, "install", "typer>=0.16.0,<0.22.0"])
run([pip_bin, "install", "spacy==3.7.5"])
# -------------------------------
# 4️⃣ Install SciSpacy + medSpaCy (clinical NLP layers on top of spaCy)
# -------------------------------
run([pip_bin, "install", "scispacy==0.6.2"])
run([pip_bin, "install", "medspacy==1.3.1"])

# SciSpaCy model en_core_sci_md. Its package pins spacy<3.8, which would
# downgrade spaCy and break medspacy/pyrush (they need >=3.8). Install with
# --no-deps: the model binaries load fine under spaCy 3.8 (v3 format is stable).
SCISPACY_SCI_MODEL_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
    "v0.5.4/en_core_sci_md-0.5.4.tar.gz"
)
run([pip_bin, "install", "--no-deps", SCISPACY_SCI_MODEL_URL])

# The published 0.5.4 model stores booleans as strings. spaCy 3.8, selected
# by medSpaCy on Python 3.12, rejects those values during model loading.
if sys.version_info >= (3, 12):
    run([
        python_bin,
        "-c",
        """
from pathlib import Path
import en_core_sci_md

config_path = next(Path(en_core_sci_md.__file__).parent.rglob("config.cfg"))
config = config_path.read_text()
config_path.write_text(config.replace(
    'include_static_vectors = "True"',
    "include_static_vectors = true",
))
""",
    ])

# Pre-download the UMLS knowledge base for the scispacy EntityLinker (~1 GB).
# There is no pip package for the KB — instantiating the linker once is the
# official way to fetch it. Doing it here (eager) means the pipeline never
# blocks on a surprise download at runtime; later runs hit the local cache.
run([
    python_bin,
    "-c",
    """
import spacy, scispacy
from scispacy.linking import EntityLinker
nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
print("UMLS knowledge base cached")
""",
])

# -------------------------------
# 5️⃣ Install the packages the optional steps need
# -------------------------------
# Not needed by the core NER/RE path, but every one of them is imported by some
# step that can be switched on in orchestrator.py: .env loading (config.py), the
# Gemini-backed LLM/holistic steps, the Neo4j import, and the dependency-graph
# figure in testing/. graphviz here is only the Python wrapper — rendering also
# needs the Graphviz binaries on PATH (graphviz.org/download).
run([pip_bin, "install", "python-dotenv", "google-genai", "neo4j", "graphviz"])

# -------------------------------
# 6️⃣ Install remaining packages
# -------------------------------
run([pip_bin, "install", "pandas", "numpy", "torch", "transformers", "sentencepiece", "scikit-learn", "datasets", "matplotlib", "sacremoses", "tqdm"])

# -------------------------------
# 7️⃣ Download spaCy English model (pipeline runs on translated text)
# -------------------------------
run([python_bin, "-m", "spacy", "download", SPACY_MODEL_ENGLISH])

# -------------------------------
# 8️⃣ Pre-download MarianMT model (Polish -> English)
# -------------------------------
print(f"Downloading MarianMT model {MARIANMT_MODEL_PL_EN}...")

run([
    python_bin,
    "-c",
    f"""
from transformers import MarianTokenizer, MarianMTModel
MarianTokenizer.from_pretrained('{MARIANMT_MODEL_PL_EN}')
MarianMTModel.from_pretrained('{MARIANMT_MODEL_PL_EN}')
print("Model downloaded")
"""
])

# -------------------------------
# 9️⃣ Pre-download BioLinkBERT model (relation extraction)
# -------------------------------
print(f"Downloading BioLinkBERT model {BIOLINK_BERT_MODEL}...")

run([
    python_bin,
    "-c",
    f"""
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('{BIOLINK_BERT_MODEL}')
AutoModel.from_pretrained('{BIOLINK_BERT_MODEL}')
print("Model downloaded")
"""
])

# -------------------------------
# 🔟 Freeze environment for reproducibility
# -------------------------------
requirements_path = os.path.join(os.getcwd(), "requirements.txt")
# Use shell=True here for redirection to file
subprocess.check_call(f"{pip_bin} freeze > {requirements_path}", shell=True)
print(f"\n✅ Pinned requirements saved to {requirements_path}")

# -------------------------------
# 10️⃣ Activation instructions
# -------------------------------
print("\n✅ Setup complete! Activate your venv with:")
if os.name == "nt":
    print(f"{VENV_DIR}\\Scripts\\activate")
else:
    print(f"source {VENV_DIR}/bin/activate")
