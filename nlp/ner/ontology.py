"""
NER ontology — entity type labels + gazetteer lexicons + regex extractors.

Exports:
  - ENT             : entity type constants (41 classes)
  - LEXICONS        : closed-class term lists per label  (used by gazetteer.py)
  - REGEX_PATTERNS  : regex per label for number+unit patterns (gazetteer.py)
"""

import re


class ENT:
    # Condition
    DISEASE_DISORDER       = "DISEASE_DISORDER"       # CHOROBA_ZABURZENIE
    SIGN_SYMPTOM           = "SIGN_SYMPTOM"           # OBJAW
    SEVERITY               = "SEVERITY"               # NASILENIE

    # Treatment
    MEDICATION             = "MEDICATION"             # LEK
    DOSAGE                 = "DOSAGE"                 # DAWKOWANIE
    FREQUENCY              = "FREQUENCY"              # CZĘSTOTLIWOŚĆ
    ADMINISTRATION         = "ADMINISTRATION"         # SPOSÓB_PODANIA
    THERAPEUTIC_PROCEDURE  = "THERAPEUTIC_PROCEDURE"  # PROCEDURA_TERAPEUTYCZNA

    # Diagnostics
    DIAGNOSTIC_PROCEDURE   = "DIAGNOSTIC_PROCEDURE"   # PROCEDURA_DIAGNOSTYCZNA
    LAB_VALUE              = "LAB_VALUE"              # WARTOŚĆ_LABORATORYJNA

    # Anatomy
    BIOLOGICAL_STRUCTURE   = "BIOLOGICAL_STRUCTURE"   # STRUKTURA_BIOLOGICZNA
    BIOLOGICAL_ATTRIBUTE   = "BIOLOGICAL_ATTRIBUTE"   # ATRYBUT_BIOLOGICZNY
    AREA                   = "AREA"                   # OBSZAR

    # Patient
    AGE                    = "AGE"                    # WIEK
    SEX                    = "SEX"                    # PŁEĆ
    WEIGHT                 = "WEIGHT"                 # MASA_CIAŁA
    HEIGHT                 = "HEIGHT"                 # WZROST
    OCCUPATION             = "OCCUPATION"             # ZAWÓD
    PERSONAL_BACKGROUND    = "PERSONAL_BACKGROUND"    # TŁO_OSOBISTE
    FAMILY_HISTORY         = "FAMILY_HISTORY"         # WYWIAD_RODZINNY

    # Time
    DATE                   = "DATE"                   # DATA
    DURATION               = "DURATION"               # CZAS_TRWANIA
    TIME                   = "TIME"                   # GODZINA

    # Measurements
    MASS                   = "MASS"                   # MASA
    VOLUME                 = "VOLUME"                 # OBJĘTOŚĆ
    DISTANCE               = "DISTANCE"               # ODLEGŁOŚĆ
    QUANTITATIVE_CONCEPT   = "QUANTITATIVE_CONCEPT"   # POJĘCIE_ILOŚCIOWE
    QUALITATIVE_CONCEPT    = "QUALITATIVE_CONCEPT"    # POJĘCIE_JAKOŚCIOWE

    # Context
    CLINICAL_EVENT         = "CLINICAL_EVENT"         # ZDARZENIE_KLINICZNE
    HISTORY                = "HISTORY"                # HISTORIA
    OUTCOME                = "OUTCOME"                # WYNIK
    DETAILED_DESCRIPTION   = "DETAILED_DESCRIPTION"   # SZCZEGÓŁOWY_OPIS
    OTHER_ENTITY           = "OTHER_ENTITY"           # INNA_JEDNOSTKA
    OTHER_EVENT            = "OTHER_EVENT"            # INNE_ZDARZENIE
    ACTIVITY               = "ACTIVITY"               # AKTYWNOŚĆ

    # Misc
    COLOR                  = "COLOR"                  # KOLOR
    SHAPE                  = "SHAPE"                  # KSZTAŁT
    TEXTURE                = "TEXTURE"                # TEKSTURA
    SUBJECT                = "SUBJECT"                # PODMIOT
    COREFERENCE            = "COREFERENCE"            # KOREFERENCJA
    NONBIOLOGICAL_LOCATION = "NONBIOLOGICAL_LOCATION" # LOKALIZACJA_NIEBIOLOGICZNA
    
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all entity type strings."""
        return [
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        ]


# ---------------------------------------------------------------------------
# Gazetteer resources — closed-class term lists and regex patterns.
# Consumed by nlp/ner/gazetteer.py.
# ---------------------------------------------------------------------------

LEXICONS: dict[str, list[str]] = {
    ENT.SUBJECT: [
        "patient", "subject", "individual", "person", "case",
    ],
    ENT.SEX: [
        "male", "female", "man", "woman", "boy", "girl",
    ],
    ENT.COLOR: [
        "red", "blue", "yellow", "white", "black", "green",
        "brown", "pink", "orange", "purple", "gray", "grey",
    ],
    ENT.SHAPE: [
        "round", "oval", "square", "rectangular", "triangular",
        "irregular", "spherical", "elongated",
    ],
    ENT.TEXTURE: [
        "rough", "smooth", "hard", "soft", "firm", "granular",
    ],
    ENT.THERAPEUTIC_PROCEDURE: [
        "excised", "excision", "removed", "removal",
        "resected", "resection",
        "drained", "drainage",
        "aspirated", "aspiration",
        "ablated", "ablation",
        "amputated", "amputation",
        "transplanted", "transplant",
        "ligated", "ligation",
        "sutured", "cauterized",
        "surgery", "operation",
    ],
    ENT.DIAGNOSTIC_PROCEDURE: [
        "MRI", "CT scan", "ultrasound", "X-ray", "xray",
        "endoscopy", "colonoscopy", "biopsy",
        "ECG", "EKG", "EEG",
        "blood test", "urinalysis",
        "examined", "examination",
        "palpated", "palpation",
        "auscultated", "auscultation",
    ],
}


REGEX_PATTERNS: dict[str, re.Pattern] = {
    ENT.AGE:      re.compile(r"\b\d{1,3}\s*(?:years?|y\.?o\.?|months?\s+old)\b", re.IGNORECASE),
    ENT.DURATION: re.compile(r"\b\d{1,3}\s*(?:years?|months?|weeks?|days?|hours?)\b", re.IGNORECASE),
    ENT.WEIGHT:   re.compile(r"\b\d{1,3}(?:\.\d+)?\s*kg\b", re.IGNORECASE),
    ENT.HEIGHT:   re.compile(r"\b\d{2,3}(?:\.\d+)?\s*cm\b", re.IGNORECASE),
    ENT.MASS:     re.compile(r"\b\d{1,4}(?:\.\d+)?\s*(?:g|grams?)\b", re.IGNORECASE),
    ENT.VOLUME:   re.compile(r"\b\d{1,4}(?:\.\d+)?\s*(?:ml|cc|litres?|liters?)\b", re.IGNORECASE),
}
