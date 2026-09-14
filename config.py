"""
Central configuration for all ML / NLP models used in the pipeline.
"""

# Load .env once, here, so secrets (GEMINI_API_KEY / NEO4J_PASSWORD / ...) reach
# os.environ on EVERY entrypoint — everything imports this module (orchestrator,
# test_sentence, nlp.ner/re.tasks, llm_client). Guarded so a missing python-dotenv
# never breaks config import.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# HuggingFace transformer model for medical NER (token classification).
# Both options share the Maccrobat label family (DISEASE_DISORDER, SIGN_SYMPTOM,
# MEDICATION, ...), so they're drop-in interchangeable in _neural_pipeline().
#   "blaze999/Medical-NER"        — DeBERTa, 41 labels (uppercase)
#   "d4data/biomedical-ner-all"   — DistilBERT, ~107 labels (Capitalized)
HF_MEDICAL_NER_MODEL = "blaze999/Medical-NER"

# Aggregation strategy for the HF token-classification pipeline — how per-token
# predictions are merged into entity spans. Options:
#   "none"    — no merging: one row per subword token (par, ##otide, ##ectomy),
#               each with its own label/score. Rawest; you aggregate yourself.
#   "simple"  — groups consecutive tokens by their B/I tags. PRESERVES spaces in
#               multi-word spans, but if the model tags a subword B- (not I-) it
#               starts a new entity -> subword fragments (par / ##otide / ##ectomy).
#   "first"   — WORD level: each word gets the label of its FIRST subword; merges
#               subwords into whole words. Fast but the first subword can mislabel
#               (bad for abbreviations like R-ICA).
#   "average" — WORD level: averages the subword score distributions, label =
#               argmax of the average. Smoother, less sensitive to one odd subword.
#   "max"     — WORD level: the word takes the label/score of its HIGHEST-scoring
#               subword. Robust when one subword is very confident.
# Trade-off on DeBERTa (blaze999): word-level modes ("first"/"average"/"max") merge
# subwords but its SentencePiece tokenizer drops spaces ("type2diabetesmellitus");
# "simple" keeps spaces but fragments. WordPiece models (d4data) keep spaces either
# way. Rebuilding `word` from char offsets would make this choice cosmetic.
HF_NER_AGGREGATION = "simple"

# MarianMT model for Polish -> English translation
MARIANMT_MODEL_PL_EN = "Helsinki-NLP/opus-mt-pl-en"


# BioLinkBERT model for relation extraction
BIOLINK_BERT_MODEL = "michiyasunaga/BioLinkBERT-base"

# How biolink picks entity pairs to classify (nlp/re/candidates.py):
#   "syntactic" — only pairs connected by a short dependency path (pruned SDP;
#                 cuts "cousin" false positives). Recall bounded by parse quality.
#   "all_pairs" — every in-sentence pair (legacy); robust to bad parses, noisier.
RE_PAIRING = "syntactic"

# spaCy English model (for RE dependency parsing on translated text)
SPACY_MODEL_ENGLISH = "en_core_web_sm"

# SciSpaCy biomedical model — sentence segmentation for the medSpaCy ConText
# assertion step (better clinical sentence boundaries than en_core_web_sm)
SCISPACY_MODEL = "en_core_sci_md"

# medSpaCy ConText scope control — stops assertion flags (negated/uncertain/
# family/historical) from leaking across a whole run-on sentence. max_scope =
# max tokens a modifier reaches from its cue (None = whole sentence, causes the
# leak). Clause punctuation / adversatives also terminate scope (see assertion.py).
CONTEXT_MAX_SCOPE = 5

# UMLS entity-linking backend (nlp/ner/linker.py). "scispacy" = in-process,
# no UMLS license (AI2 ships the KB). "medcat" = reserved (out-of-process,
# needs a model pack), not wired yet.
UMLS_LINKER = "scispacy"

# LLM backend for the optional LLM-based NER / RE peer methods (nlp/ner/llm.py,
# nlp/re/llm.py). Provider-abstracted (nlp/llm_client.py) so the model can be
# swapped/upgraded later. The API key stays in the environment, never here:
# GEMINI_API_KEY (preferred) or GOOGLE_API_KEY. These methods are opt-in
# (STEPS["ner_llm"]/["re_llm"] = False); with no key they degrade to a no-op.
LLM_BACKEND = "gemini"             # future: "anthropic", "openai", "local", ...
LLM_MODEL   = "gemini-3.7-flash"   # env GENAI_MODEL overrides at runtime

# How many records (ner_llm) / sentences (re_llm) go into ONE call of the opt-in
# LLM peer methods. Same trade-off as HOLISTIC_BATCH_SIZE: fewer calls and less
# repeated prompt scaffolding, but one unparseable response now costs the whole
# batch. RE batches are smaller because each sentence drags its candidate-pair
# list along. Set to 1 for one call per record / per sentence.
NER_LLM_BATCH_SIZE = 10
RE_LLM_BATCH_SIZE  = 5

# How many times nlp/llm_client.py retries a failed call before giving the batch up.
# A full-corpus run is ~1000 sequential calls, and every one of them can die on a
# per-minute rate limit, a timeout, a 5xx or an empty (safety-filtered) response —
# none of which has anything to do with having tokens left. Retries wait 2s, 4s, ...
# Only transport failures are retried; an unparseable answer is not, to avoid paying
# twice for the same output.
LLM_MAX_RETRIES = 2

# Processing variant: "pipeline" = modular NER + RE (default), "holistic" = one
# generative step that classifies entities AND relations jointly (nlp/holistic.py),
# ontology-constrained; it REPLACES the modular NER/RE extractors but shares the
# same enrichment (ConText + UMLS) and FHIR graph build.
PIPELINE_MODE = "holistic"

# How many records go into ONE holistic LLM call. The prompt's ontology block is
# ~2000 tokens and identical every time, so batching amortises it across N records
# (10 records/call ≈ 10x less input billed, 10x fewer round trips). Downside: an
# unparseable response now loses N records instead of 1. Set to 1 for one call per
# record (the original behaviour).
HOLISTIC_BATCH_SIZE = 10

# Who decides the four assertion axes in the holistic variant: the LLM, or ConText.
#
# False (default) keeps ConText, so the holistic run shares its whole downstream with
# the modular pipeline and any difference between the two isolates NER+RE. That is
# what makes the two variants comparable, and it is why this defaults off.
#
# True asks the LLM for negated/uncertain/family/historical per entity and SKIPS
# ConText. Worth trying because ConText is a token-window rule engine: it flags
# whatever falls within CONTEXT_MAX_SCOPE of a cue, so it mislabels coordinations
# ("family history of breast cancer and ovarian cancer" flags only the first) and
# fires on incidental cues. A model reading the sentence should do better — but
# switching this on means a graph difference no longer isolates extraction quality.
HOLISTIC_LLM_ASSERTIONS = True

# ---------------------------------------------------------------------------
# Score cutoffs — every confidence filter in the pipeline, in one place.
# Each is applied by exactly one module; the comment says where.
# ---------------------------------------------------------------------------

# NER: minimum entity confidence, applied in nlp/ner/tasks.py BEFORE dedup, to
# every method. Only the neural model is actually affected — the gazetteer emits
# 1.0 and the LLM 0.9+.
#
# WARNING: the right value depends on STEPS["ner_llm"]. Measured against the
# reference annotation of test_data/structured_easy.csv (63 entities):
#
#            with ner_llm            without ner_llm
#   0.00     P .78  R .97  F1 .87    P .77  R .78  F1 .77
#   0.40     P .92  R .94  F1 .93    P .91  R .51  F1 .65
#   0.60     P .94  R .94  F1 .94    P .93  R .43  F1 .59
#
# With the LLM peer on, cutting the neural model costs almost no recall because
# the LLM independently finds the same entities at 0.9+; without it, recall
# collapses. Set to 0.0 when running neural-only.
NER_SCORE_THRESHOLD = 0.40

# RE / BioLinkBERT: minimum cosine similarity to accept the best-scoring relation
# for a candidate pair. Applied in nlp/re/biolink.py.
RE_BIOLINK_SCORE_THRESHOLD = 0.70

# RE / LLM: minimum LLM confidence to keep a relation. Applied in nlp/re/llm.py
# and nlp/holistic.py. Effectively inert — the LLM returns 0.9+ in practice.
LLM_RE_SCORE_THRESHOLD = 0.1

# UMLS linking: minimum scispaCy candidate score. Applied in nlp/ner/linker.py.
# 0.70 is also the scispaCy default; set here so it is explicit and tunable.
UMLS_LINKER_THRESHOLD = 0.70

# Graph build: minimum relation score for an edge. Applied in
# visualization/pipeline.py — CONCEPT MODE ONLY. The FHIR builder does not filter.
GRAPH_RELATION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Graph model (docs/PatientCentrism.md, Część III/IV).
#   "concept" — legacy concept-only graph (nodes/edges from re_results only).
#   "fhir"    — FHIR-reification graph: every entity MENTION becomes a reified
#               fact resource (Condition/Observation/Procedure/...), the shared
#               CUI stays a CodeableConcept (:Concept) node, provenance rides as
#               source_id+offsets on the fact. Built from ner_results + re_results.
# ---------------------------------------------------------------------------
GRAPH_MODE = "concept"

# Demographic entities → properties on :Patient (never their own nodes / facts).
# Property name = label lower-cased (AGE→age, SEX→sex, ...).
DEMOGRAPHIC_TYPES = {"AGE", "SEX", "WEIGHT", "HEIGHT", "OCCUPATION", "PERSONAL_BACKGROUND"}

# NER label → FHIR resource type (node label of the reified fact). Fidelity per
# doc Q8: SIGN_SYMPTOM→Condition, DIAGNOSTIC_PROCEDURE→Procedure. Only these
# labels become facts; demographics → Patient props, the rest (attributes) →
# fields on a fact, reachable via the [:MENTIONS] backstop.
#
# Anatomy and care settings are resources in their own right rather than embedded
# codes: FHIR has BodyStructure for a patient's anatomical structure and Location
# for a place of care (Procedure.location is already a Reference in R4). Keeping
# them as facts is what lets "lymph node LOCATED_IN level IVa" survive at all —
# a reference into the shared concept layer has nowhere to hang a relation.
FHIR_RESOURCE_MAP = {
    "DISEASE_DISORDER":       "Condition",
    "SIGN_SYMPTOM":           "Condition",
    "MEDICATION":             "MedicationStatement",
    "THERAPEUTIC_PROCEDURE":  "Procedure",
    "DIAGNOSTIC_PROCEDURE":   "Procedure",
    "LAB_VALUE":              "Observation",
    "OUTCOME":                "Observation",
    # Social history / functional status ("walking", "lifting a heavy suitcase"):
    # a statement about the patient, and the subject of real CAUSES edges.
    "ACTIVITY":               "Observation",
    # Admissions, visits, referrals — organisational events, not observations.
    "CLINICAL_EVENT":         "Encounter",
    "BIOLOGICAL_STRUCTURE":   "BodyStructure",
    "AREA":                   "BodyStructure",
    "NONBIOLOGICAL_LOCATION": "Location",
}

# Relation → role under FHIR reification (docs/PatientCentrism.md Część III.4).
# Keys mirror nlp.re.ontology.ONT (kept as literals so this foundational module
# stays import-light). Roles:
#   "clinical"    — survives as a fact↔fact edge (edge TYPE = canonical label).
#                   Anatomical relations live here too: since a body part is a
#                   :BodyStructure fact, "lesion LOCATED_IN lobe" is an ordinary
#                   fact↔fact edge and needs no role of its own.
#   "attribute"   — object is a value/descriptor → `field` on the host fact
#                   ONLY (policy A / Kubełek 3: a coded value, not a node/edge);
#                   the value stays reachable via the [:MENTIONS] backstop.
#   "demographic" — becomes a :Patient property (handled directly, not an edge).
#   "dissolved"   — no edge: becomes clinicalStatus/onset or is redundant.
# `deprecated` marks labels that duplicate another under FHIR ("mapuj teraz,
# przytnij później") — merged to `canonical` / `field`, flagged for future pruning.
RELATION_FHIR_MAP = {
    # --- clinical fact↔fact edges ---
    "CAUSES":          {"role": "clinical"},
    "TREATS":          {"role": "clinical"},
    "DIAGNOSES":       {"role": "clinical"},
    "REVEALS":         {"role": "clinical"},
    "INDICATES":       {"role": "clinical"},
    "ASSOCIATED_WITH": {"role": "clinical"},
    "HAS_SYMPTOM":     {"role": "clinical"},
    "RESULTS_IN":      {"role": "clinical"},
    # Where a procedure took place: the object is a care setting (:Location),
    # never anatomy, so this is an ordinary clinical edge.
    "PERFORMED_IN":    {"role": "clinical"},
    # Procedure → the value it produced. Both ends are already facts, so an edge
    # says more than copying the value onto the procedure as a string would.
    "MEASURES":        {"role": "clinical"},
    "GIVEN_FOR":       {"role": "clinical", "canonical": "TREATS",      "deprecated": True},
    "LEADS_TO":        {"role": "clinical", "canonical": "RESULTS_IN",  "deprecated": True},
    "IS_VALUE_OF":     {"role": "clinical", "canonical": "MEASURES",    "deprecated": True},
    # --- anatomical placement: object is a :BodyStructure fact ---
    "LOCATED_IN":      {"role": "clinical"},
    "PERFORMED_ON":    {"role": "clinical"},
    "AFFECTS":         {"role": "clinical"},
    # "level IVa CONTAINS lymph node" is "lymph node LOCATED_IN level IVa" read
    # backwards, and both come out of the same sentence. RELATION_DIRECTION cannot
    # sort this one out because both endpoints are BodyStructure, hence "inverse".
    "CONTAINS":        {"role": "clinical", "canonical": "LOCATED_IN",
                        "inverse": True, "deprecated": True},
    # --- attribute fields (Kubełek 3) ---
    "HAS_SEVERITY":       {"role": "attribute", "field": "severity"},
    "HAS_DOSAGE":         {"role": "attribute", "field": "dosage"},
    "HAS_FREQUENCY":      {"role": "attribute", "field": "frequency"},
    "HAS_ADMINISTRATION": {"role": "attribute", "field": "route"},
    "HAS_COLOR":          {"role": "attribute", "field": "color"},
    "HAS_SHAPE":          {"role": "attribute", "field": "shape"},
    "HAS_TEXTURE":        {"role": "attribute", "field": "texture"},
    "HAS_MASS":           {"role": "attribute", "field": "value"},
    "HAS_VOLUME":         {"role": "attribute", "field": "value"},
    "DURATION_OF":        {"role": "attribute", "field": "duration"},
    "HAS_DURATION":       {"role": "attribute", "field": "duration"},
    "OCCURS_AT":          {"role": "attribute", "field": "onset"},
    "DESCRIBES":          {"role": "attribute", "field": "descriptor"},
    "ADMINISTERED_AS":    {"role": "attribute", "field": "dosage", "deprecated": True},
    "QUANTIFIES":         {"role": "attribute", "field": "value",  "deprecated": True},
    # --- demographics → :Patient property (not an edge) ---
    "HAS_AGE":                 {"role": "demographic"},
    "HAS_SEX":                 {"role": "demographic"},
    "HAS_WEIGHT":              {"role": "demographic"},
    "HAS_HEIGHT":              {"role": "demographic"},
    "HAS_OCCUPATION":          {"role": "demographic"},
    "HAS_PERSONAL_BACKGROUND": {"role": "demographic"},
    # --- dissolved (temporal/status/redundant) ---
    "HISTORY_OF":          {"role": "dissolved"},
    "PREVIOUSLY_OCCURRED": {"role": "dissolved"},
    "RISK_FACTOR_OF":      {"role": "dissolved"},
    "RELATED_TO":          {"role": "dissolved"},
}

# Attribute field names a fact can carry (derived from RELATION_FHIR_MAP).
ATTRIBUTE_FIELDS = sorted({m["field"] for m in RELATION_FHIR_MAP.values()
                           if m["role"] == "attribute"})

# Canonical subject side for asymmetric clinical relations, keyed by the
# CANONICAL relation name. ENTITY_PAIR_RELATIONS licenses both orientations of
# these pairs and the extractors disagree in practice (BioLinkBERT reads
# "diabetes treated with an insulin pump" as diabetes TREATS pump), while the RE
# dedup key is order-sensitive so both survive. A therapy treats a condition and
# a test diagnoses one, never the reverse, so when only the OBJECT endpoint sits
# on the subject side the edge is emitted with its endpoints swapped.
# Resource names, matched against FHIR_RESOURCE_MAP of the entity labels.
RELATION_DIRECTION = {
    "TREATS":    {"Procedure", "MedicationStatement"},
    "DIAGNOSES": {"Procedure"},
    # IS_VALUE_OF is MEASURES written backwards ("Bethesda II IS_VALUE_OF biopsy"),
    # so once it canonicalises the endpoints need flipping back.
    "MEASURES":  {"Procedure"},
}

# Data directory — flip to True to use the small test subset in test_data/
USE_TEST_DATA = False
DATA_DIR = "test_data" if USE_TEST_DATA else "data"

# Data format switch - set to "csv" to use corpus_summary_all.csv, "txt" for data.txt
DATA_FORMAT = "csv"  # or "csv"

# Data file paths
INPUT_TXT          = f"{DATA_DIR}/data.txt"
INPUT_CSV          = f"{DATA_DIR}/corpus_summary_all.csv"
STRUCTURED_CSV     = f"{DATA_DIR}/structured.csv"
NER_OUTPUT_CSV     = f"{DATA_DIR}/ner_results.csv"
NER_RAW_OUTPUT_CSV = f"{DATA_DIR}/ner_results_raw.csv"  # pre-dedup, both methods kept
RE_OUTPUT_CSV      = f"{DATA_DIR}/re_results.csv"
RE_RAW_OUTPUT_CSV  = f"{DATA_DIR}/re_results_raw.csv"  # pre-dedup, both methods kept
CORPUS_SUMMARY_CSV = f"{DATA_DIR}/corpus_summary_all.csv"

# Input file selection based on format
INPUT_FILE = INPUT_CSV if DATA_FORMAT == "csv" else INPUT_TXT