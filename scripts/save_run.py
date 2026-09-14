"""
Save the outputs of the last pipeline run into prev_runs/<run name>/.

Usage:
    python scripts/save_run.py "run name"

Copies the run input (structured.csv), the NER/RE CSVs from data/ and the FHIR
graph CSVs from visualization/.
Fails if prev_runs/<run name>/ already exists, so an earlier run is never overwritten.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from visualization.config import (
    FHIR_PATIENTS_CSV, FHIR_CONCEPTS_CSV, FHIR_FACTS_CSV, FHIR_EDGES_CSV,
)

RUN_FILES = [
    config.STRUCTURED_CSV,
    config.NER_OUTPUT_CSV,
    config.NER_RAW_OUTPUT_CSV,
    config.RE_OUTPUT_CSV,
    config.RE_RAW_OUTPUT_CSV,
    FHIR_PATIENTS_CSV,
    FHIR_CONCEPTS_CSV,
    FHIR_FACTS_CSV,
    FHIR_EDGES_CSV,
]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/save_run.py "run name"')
        sys.exit(1)

    dst = ROOT / "prev_runs" / sys.argv[1]
    if dst.exists():
        print(f"{dst} already exists — pick another run name.")
        sys.exit(1)
    dst.mkdir(parents=True)

    copied = 0
    for rel in RUN_FILES:
        src = ROOT / rel
        if not src.is_file():
            print(f"[MISS] {rel}")
            continue
        shutil.copy2(src, dst / src.name)
        print(f"[OK]   {rel}")
        copied += 1

    print(f"Copied {copied}/{len(RUN_FILES)} files -> {dst}")


if __name__ == "__main__":
    main()
