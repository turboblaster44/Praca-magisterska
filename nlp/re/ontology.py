"""
Relation extraction ontology — single source of truth for all RE definitions.

Exports:
  - ONT                        : relation label constants
  - LEXICAL_PATTERNS_FORWARD   : trigger phrases where cause/agent is BEFORE the trigger
  - LEXICAL_PATTERNS_INVERTED  : trigger phrases where cause/agent is AFTER the trigger
  - VERB_RELATION_MAP          : verb lemmas → relation label  (re_spacy)
  - RELATION_PROTOTYPES        : natural-language descriptions (re_biolink)
  - ENTITY_PAIR_RELATIONS      : valid relations per entity-type pair

Directionality:
  FORWARD  → entity_before --relation--> entity_after   ("X causes Y")
  INVERTED → entity_after  --relation--> entity_before  ("X due to Y" → Y causes X)
"""

from nlp.ner.ontology import ENT


class ONT:
    # --- Core clinical [both] ---
    CAUSES              = "CAUSES"
    TREATS              = "TREATS"
    INDICATES           = "INDICATES"

    # --- Coarse clinical [v1] ---
    ASSOCIATED_WITH     = "ASSOCIATED_WITH"
    AFFECTS             = "AFFECTS"
    DIAGNOSES           = "DIAGNOSES"
    QUANTIFIES          = "QUANTIFIES"      # lab / weight / mass → sign / disease
    DURATION_OF         = "DURATION_OF"     # duration / date / time → event / symptom
    ADMINISTERED_AS     = "ADMINISTERED_AS" # medication → dosage / frequency / route
    RISK_FACTOR_OF      = "RISK_FACTOR_OF"  # age / family history → disease

    # --- Anatomy [both] ---
    LOCATED_IN          = "LOCATED_IN"      # disease/symptom/structure → area/structure

    # --- Fine-grained condition [v2] ---
    HAS_SYMPTOM         = "HAS_SYMPTOM"     # disease has_symptom sign/symptom
    HAS_SEVERITY        = "HAS_SEVERITY"    # disease/symptom has_severity severity

    # --- Fine-grained treatment [v2] ---
    GIVEN_FOR           = "GIVEN_FOR"       # medication given_for disease
    HAS_DOSAGE          = "HAS_DOSAGE"      # medication has_dosage dosage
    HAS_FREQUENCY       = "HAS_FREQUENCY"   # medication has_frequency frequency
    HAS_ADMINISTRATION  = "HAS_ADMINISTRATION"  # medication has_administration route

    # --- Fine-grained procedures & events [v2] ---
    PERFORMED_ON        = "PERFORMED_ON"    # procedure performed_on structure/area
    OCCURS_AT           = "OCCURS_AT"       # procedure/event occurs_at date/time
    HAS_DURATION        = "HAS_DURATION"    # procedure/event has_duration duration
    RESULTS_IN          = "RESULTS_IN"      # procedure results_in outcome
    LEADS_TO            = "LEADS_TO"        # event leads_to outcome
    PERFORMED_IN        = "PERFORMED_IN"    # event performed_in location

    # --- Fine-grained diagnostics [v2] ---
    REVEALS             = "REVEALS"         # diagnostic procedure reveals disease/symptom
    MEASURES            = "MEASURES"        # diagnostic procedure measures lab value
    IS_VALUE_OF         = "IS_VALUE_OF"     # lab value is_value_of diagnostic procedure

    # --- Fine-grained anatomy [v2] ---
    CONTAINS            = "CONTAINS"        # area contains biological structure

    # --- Appearance [v2] ---
    HAS_COLOR           = "HAS_COLOR"
    HAS_SHAPE           = "HAS_SHAPE"
    HAS_TEXTURE         = "HAS_TEXTURE"
    HAS_MASS            = "HAS_MASS"            # structure/lesion → mass value
    HAS_VOLUME          = "HAS_VOLUME"          # structure/lesion → volume value

    # --- History & narrative [v2] ---
    PREVIOUSLY_OCCURRED = "PREVIOUSLY_OCCURRED" # history previously_occurred event
    DESCRIBES           = "DESCRIBES"           # description describes event
    RELATED_TO          = "RELATED_TO"          # activity related_to event
    HISTORY_OF          = "HISTORY_OF"          # subject history_of disease

    # --- Subject demographics [v2] ---
    HAS_AGE                 = "HAS_AGE"
    HAS_SEX                 = "HAS_SEX"
    HAS_WEIGHT              = "HAS_WEIGHT"
    HAS_HEIGHT              = "HAS_HEIGHT"
    HAS_OCCUPATION          = "HAS_OCCUPATION"
    HAS_PERSONAL_BACKGROUND = "HAS_PERSONAL_BACKGROUND"
    HAS_FAMILY_HISTORY      = "HAS_FAMILY_HISTORY"


# ---------------------------------------------------------------------------
# FORWARD lexical triggers  (agent/cause BEFORE trigger word)
# "X [trigger] Y"  →  X --relation--> Y
# ---------------------------------------------------------------------------
LEXICAL_PATTERNS_FORWARD: dict[str, list[str]] = {
    ONT.CAUSES: [
        "leads to",
        "leading to",
        "causes",
        "triggers",
        "induces",
        "produces",
        "results in",
    ],
    ONT.ASSOCIATED_WITH: [
        "associated with",
        "in the context of",
        "related to",
        "linked to",
        "in association with",
        "coexisting with",
        "concurrent with",
        "concomitant with",
        "accompanied by",
        "coexistent with",
        "in combination with",
        "together with",
    ],
    ONT.INDICATES: [
        "consistent with",
        "indicating",
        "indicates",
        "suggesting",
        "suggests",
        "reveals",
        "showing",
        "positive for",
        "revealing",
        "reflects",
        "elevated in",
    ],
    ONT.AFFECTS: [
        "affecting",
        "involves",
        "involving",
        "damages",
        "damaging",
        "impairs",
        "impairing",
        "spreads to",
        "metastasized to",
    ],
    ONT.RISK_FACTOR_OF: [
        "increases risk of",
        "increased risk of",
        "predisposes to",
        "predisposed to",
    ],
    ONT.LOCATED_IN: [
        "located in",
        "found in",
        "present in",
        "within the",
        "in the region of",
        "in the area of",
    ],
    ONT.QUANTIFIES: [
        "measuring",
        "measured at",
        "of",
    ],
    ONT.DURATION_OF: [
        "ago",
        "since",
        "since diagnosis",
        "years ago",
        "months ago",
        "weeks ago",
        "days ago",
    ],
    ONT.HAS_SYMPTOM: [
        "presents with",
        "presenting with",
        "manifests as",
        "manifesting as",
        "characterized by",
        "accompanied by",
        "has symptom",
    ],
    ONT.HAS_SEVERITY: [
        "graded as",
        "rated as",
        "classified as",
        "of severity",
        "with severity",
    ],
    ONT.TREATS: [
        "treats",
        "used to treat",
        "therapy for",
        "treatment for",
        "manages",
        "reduces",
        "controls",
        "alleviates",
    ],
    ONT.GIVEN_FOR: [
        "given for",
        "prescribed for",
        "indicated for",
        "administered for",
        "used for",
    ],
    ONT.HAS_DOSAGE: [
        "at a dose of",
        "at dose",
        "dosed at",
        "given at",
        "administered at",
        "taken at",
    ],
    ONT.HAS_FREQUENCY: [
        "once daily",
        "twice daily",
        "three times daily",
        "per day",
        "every day",
        "every other day",
        "weekly",
        "monthly",
    ],
    ONT.HAS_ADMINISTRATION: [
        "administered",
        "given",
        "taken",
        "orally",
        "intravenously",
        "subcutaneously",
        "intramuscularly",
        "via",
    ],
    ONT.PERFORMED_ON: [
        "performed on",
        "done on",
        "carried out on",
        "conducted on",
        "applied to",
    ],
    ONT.OCCURS_AT: [
        "occurred at",
        "scheduled at",
        "performed at",
        "happened at",
    ],
    ONT.HAS_DURATION: [
        "lasting",
        "lasting for",
        "for a duration of",
        "over a period of",
        "duration of",
    ],
    ONT.RESULTS_IN: [
        "results in",
        "resulted in",
        "led to outcome",
    ],
    ONT.LEADS_TO: [
        "leads to",
        "led to",
        "culminated in",
    ],
    ONT.PERFORMED_IN: [
        "performed in",
        "done in",
        "carried out in",
        "conducted in",
    ],
    ONT.REVEALS: [
        "reveals",
        "revealed",
        "detects",
        "detected",
        "demonstrates",
        "demonstrates presence of",
    ],
    ONT.MEASURES: [
        "measures",
        "measuring",
        "quantifies",
        "quantifying",
    ],
    ONT.IS_VALUE_OF: [
        "is value of",
        "is the result of",
        "corresponds to",
        "obtained from",
    ],
    ONT.CONTAINS: [
        "contains",
        "includes",
        "comprises",
    ],
    ONT.PREVIOUSLY_OCCURRED: [
        "previously occurred",
        "occurred previously",
        "past occurrence of",
    ],
    ONT.DESCRIBES: [
        "describes",
        "describing",
        "details",
        "elaborates on",
    ],
    ONT.RELATED_TO: [
        "related to",
        "linked to",
        "connected to",
        "pertaining to",
    ],
    ONT.HISTORY_OF: [
        "history of",
        "past history of",
        "prior history of",
        "known history of",
    ],
    ONT.HAS_AGE: [
        "aged",
        "age of",
        "years old",
        "year-old",
    ],
    ONT.HAS_SEX: [
        "is male",
        "is female",
        "male patient",
        "female patient",
        "sex is",
    ],
    ONT.HAS_WEIGHT: [
        "weighs",
        "weight of",
        "body weight of",
    ],
    ONT.HAS_HEIGHT: [
        "height of",
        "stands",
    ],
    ONT.HAS_OCCUPATION: [
        "works as",
        "employed as",
        "occupation is",
    ],
    ONT.HAS_PERSONAL_BACKGROUND: [
        "background of",
        "with a background",
        "social history of",
    ],
    ONT.HAS_FAMILY_HISTORY: [
        "family history of",
        "family member with",
        "father had",
        "mother had",
        "sibling with",
    ],
    ONT.HAS_MASS: [
        "with mass",
        "of mass",
        "with a mass of",
        "weighing",
        "mass of",
    ],
    ONT.HAS_VOLUME: [
        "with volume",
        "of volume",
        "with a volume of",
        "volume of",
        "measuring",
    ],
}

# ---------------------------------------------------------------------------
# INVERTED lexical triggers  (agent/cause AFTER trigger word)
# "X [trigger] Y"  →  Y --relation--> X
# ---------------------------------------------------------------------------
LEXICAL_PATTERNS_INVERTED: dict[str, list[str]] = {
    ONT.CAUSES: [
        "due to",
        "caused by",
        "secondary to",
        "as a result of",
        "resulting from",
        "triggered by",
        "induced by",
        "attributed to",
    ],
    ONT.TREATS: [
        "treated with",
        "treatment with",
        "therapy with",
        "managed with",
        "management with",
        "underwent",
        "prescribed",
        "treated by",
        "received",
        "on therapy with",
    ],
    ONT.DIAGNOSES: [
        "diagnosed with",
        "diagnosis of",
        "identified as",
        "confirmed as",
        "detected by",
        "found to have",
        "presenting with",
        "presenting as",
    ],
    ONT.INDICATES: [
        "confirmed by",
        "revealed by",
        "shown by",
        "evidenced by",
    ],
    ONT.DURATION_OF: [
        "for the last",
        "for the past",
        "over the last",
        "over the past",
        "in the last",
        "in the past",
        "during the last",
        "during the past",
        "over",
        "for",
    ],
    ONT.ADMINISTERED_AS: [
        "at a dose of",
        "at dose",
        "dosed at",
        "given at",
        "administered at",
        "taken at",
        "once daily",
        "twice daily",
        "three times daily",
        "per day",
        "mg daily",
        "mg twice",
        "orally",
        "intravenously",
        "subcutaneously",
        "intramuscularly",
    ],
    ONT.RISK_FACTOR_OF: [
        "risk factor for",
        "history of",
        "family history of",
        "prior history of",
    ],
    ONT.HAS_SEVERITY: [
        "severity of",
        "grade of",
        "rated as",
    ],
    ONT.HAS_DURATION: [
        "for the last",
        "for the past",
        "over the last",
        "over the past",
        "in the last",
        "in the past",
        "during the last",
        "during the past",
    ],
    ONT.OCCURS_AT: [
        "on the date of",
        "at the time of",
        "during",
    ],
    ONT.REVEALS: [
        "diagnosed with",
        "diagnosis of",
        "shown by",
        "detected by",
        "confirmed by",
        "found to have",
    ],
    ONT.PERFORMED_IN: [
        "in the setting of",
        "at the facility of",
    ],
}

# ---------------------------------------------------------------------------
# Predicate verb lemmas → relation label  (dependency-based extraction)
# ---------------------------------------------------------------------------
VERB_RELATION_MAP: dict[str, str] = {
    # CAUSES
    "cause":      ONT.CAUSES,
    "lead":       ONT.CAUSES,
    "result":     ONT.CAUSES,
    "trigger":    ONT.CAUSES,
    "induce":     ONT.CAUSES,
    "produce":    ONT.CAUSES,
    "develop":    ONT.CAUSES,
    "provoke":    ONT.CAUSES,

    # TREATS
    "treat":      ONT.TREATS,
    "manage":     ONT.TREATS,
    "cure":       ONT.TREATS,
    "prescribe":  ONT.TREATS,
    "administer": ONT.TREATS,
    "receive":    ONT.TREATS,
    "undergo":    ONT.TREATS,
    "reduce":     ONT.TREATS,
    "alleviate":  ONT.TREATS,
    "control":    ONT.TREATS,

    # ASSOCIATED_WITH
    "associate":  ONT.ASSOCIATED_WITH,
    "correlate":  ONT.ASSOCIATED_WITH,
    "relate":     ONT.ASSOCIATED_WITH,
    "coexist":    ONT.ASSOCIATED_WITH,
    "accompany":  ONT.ASSOCIATED_WITH,

    # INDICATES
    "indicate":   ONT.INDICATES,
    "suggest":    ONT.INDICATES,
    "reflect":    ONT.INDICATES,
    "confirm":    ONT.INDICATES,

    # AFFECTS
    "affect":     ONT.AFFECTS,
    "involve":    ONT.AFFECTS,
    "damage":     ONT.AFFECTS,
    "impair":     ONT.AFFECTS,
    "spread":     ONT.AFFECTS,
    "invade":     ONT.AFFECTS,
    "compromise": ONT.AFFECTS,

    # DIAGNOSES
    "diagnose":   ONT.DIAGNOSES,
    "detect":     ONT.DIAGNOSES,
    "identify":   ONT.DIAGNOSES,
    "find":       ONT.DIAGNOSES,
    "present":    ONT.DIAGNOSES,

    # RISK_FACTOR_OF
    "predispose": ONT.RISK_FACTOR_OF,
    "increase":   ONT.RISK_FACTOR_OF,

    # HAS_SYMPTOM
    "manifest":   ONT.HAS_SYMPTOM,
    "exhibit":    ONT.HAS_SYMPTOM,

    # REVEALS
    "reveal":     ONT.REVEALS,
    "show":       ONT.REVEALS,
    "demonstrate": ONT.REVEALS,

    # MEASURES
    "measure":    ONT.MEASURES,
    "quantify":   ONT.QUANTIFIES,

    # PERFORMED_ON
    "perform":    ONT.PERFORMED_ON,
    "conduct":    ONT.PERFORMED_ON,
    "apply":      ONT.PERFORMED_ON,

    # OCCURS_AT
    "occur":      ONT.OCCURS_AT,
    "happen":     ONT.OCCURS_AT,
    "schedule":   ONT.OCCURS_AT,

    # LEADS_TO
    "culminate":  ONT.LEADS_TO,

    # LOCATED_IN
    "locate":     ONT.LOCATED_IN,
    "lie":        ONT.LOCATED_IN,

    # CONTAINS
    "contain":    ONT.CONTAINS,
    "comprise":   ONT.CONTAINS,

    # DESCRIBES
    "describe":   ONT.DESCRIBES,
    "detail":     ONT.DESCRIBES,
    "explain":    ONT.DESCRIBES,
}

# ---------------------------------------------------------------------------
# Relation prototypes – short sentences describing each relation type.
# BioLinkBERT encodes these once; entity-pair embeddings are compared against
# them to pick the most semantically similar relation label.
# ---------------------------------------------------------------------------
RELATION_PROTOTYPES: dict[str, str] = {
    ONT.CAUSES: (
        "X causes Y. X leads to Y. X results in Y. "
        "X triggers Y. X produces Y. X induces Y."
    ),
    ONT.TREATS: (
        "X treats Y. X is used to treat Y. "
        "X is therapy for Y. X manages Y. X cures Y. "
        "X reduces Y. X controls Y."
    ),
    ONT.ASSOCIATED_WITH: (
        "X is associated with Y. X is related to Y. "
        "X correlates with Y. X is linked to Y. "
        "X co-occurs with Y. X coexists with Y."
    ),
    ONT.AFFECTS: (
        "X affects Y. X involves Y. X damages Y. "
        "X impairs Y. X impacts Y. X disrupts Y. "
        "X spreads to Y. X invades Y."
    ),
    ONT.DIAGNOSES: (
        "X diagnoses Y. X detects Y. X confirms Y. "
        "X reveals Y. X identifies Y. X presents with Y."
    ),
    ONT.INDICATES: (
        "X indicates Y. X is a marker of Y. "
        "X suggests Y. X is consistent with Y. X shows Y. "
        "X reflects Y. X is elevated in Y."
    ),
    ONT.QUANTIFIES: (
        "X measures Y. X is the value of Y. "
        "X quantifies Y. X is the amount of Y. X is the level of Y."
    ),
    ONT.DURATION_OF: (
        "X is the duration of Y. X is how long Y lasted. "
        "Y occurred over X. Y persisted for X."
    ),
    ONT.ADMINISTERED_AS: (
        "X is administered as Y. X is given at dose Y. "
        "X is taken Y. X is dosed Y. X is prescribed at Y."
    ),
    ONT.RISK_FACTOR_OF: (
        "X is a risk factor for Y. X predisposes to Y. "
        "X increases risk of Y. X is associated with higher risk of Y."
    ),
    ONT.LOCATED_IN: (
        "X is located in Y. X is found in Y. "
        "X is present in Y. X is within Y. X is part of Y."
    ),
    ONT.HAS_SYMPTOM: (
        "X has symptom Y. X presents with Y. "
        "X is characterized by Y. X manifests as Y."
    ),
    ONT.HAS_SEVERITY: (
        "X has severity Y. X is Y in severity. "
        "X is graded as Y. X is classified as Y."
    ),
    ONT.GIVEN_FOR: (
        "X is given for Y. X is prescribed for Y. "
        "X is indicated for Y. X is administered for Y."
    ),
    ONT.HAS_DOSAGE: (
        "X has dosage Y. X is given at dose Y. "
        "X is dosed at Y. X is taken at Y."
    ),
    ONT.HAS_FREQUENCY: (
        "X has frequency Y. X is taken Y. "
        "X is given Y times per day. X is administered Y."
    ),
    ONT.HAS_ADMINISTRATION: (
        "X is administered as Y. X is given via Y. "
        "X is taken Y. X is delivered by Y route."
    ),
    ONT.PERFORMED_ON: (
        "X is performed on Y. X is done on Y. "
        "X is applied to Y. X is conducted on Y."
    ),
    ONT.OCCURS_AT: (
        "X occurs at Y. X is scheduled at Y. "
        "X happened at Y. X took place at Y."
    ),
    ONT.HAS_DURATION: (
        "X has duration Y. X lasted for Y. "
        "X continued for Y. X persisted over Y."
    ),
    ONT.RESULTS_IN: (
        "X results in Y. X led to outcome Y. "
        "X produced outcome Y. X culminated in Y."
    ),
    ONT.LEADS_TO: (
        "X leads to Y. X led to Y. "
        "X resulted in Y. X culminated in Y."
    ),
    ONT.PERFORMED_IN: (
        "X is performed in Y. X is done in Y. "
        "X took place in Y. X was conducted at Y."
    ),
    ONT.REVEALS: (
        "X reveals Y. X detected Y. X identifies Y. "
        "X demonstrates Y. X confirms the presence of Y."
    ),
    ONT.MEASURES: (
        "X measures Y. X is the measurement of Y. "
        "X quantifies Y. X is the amount of Y."
    ),
    ONT.IS_VALUE_OF: (
        "X is the value of Y. X is the result of Y. "
        "X corresponds to Y. X was obtained from Y."
    ),
    ONT.CONTAINS: (
        "X contains Y. X includes Y. X comprises Y."
    ),
    ONT.HAS_COLOR: (
        "X has color Y. X appears Y. X is colored Y."
    ),
    ONT.HAS_SHAPE: (
        "X has shape Y. X is Y shaped. X appears Y in shape."
    ),
    ONT.HAS_TEXTURE: (
        "X has texture Y. X feels Y. X is Y in texture."
    ),
    ONT.HAS_MASS: (
        "X has mass Y. X has a mass of Y grams. X weighs Y."
    ),
    ONT.HAS_VOLUME: (
        "X has volume Y. X has a volume of Y. X measures Y in volume."
    ),
    ONT.PREVIOUSLY_OCCURRED: (
        "X previously occurred Y. X had a prior event of Y. "
        "X had past occurrence of Y."
    ),
    ONT.DESCRIBES: (
        "X describes Y. X gives a description of Y. "
        "X details Y. X explains Y."
    ),
    ONT.RELATED_TO: (
        "X is related to Y. X is linked to Y. X pertains to Y."
    ),
    ONT.HISTORY_OF: (
        "X has history of Y. X had Y in the past. "
        "X has prior history of Y. X previously had Y."
    ),
    ONT.HAS_AGE: (
        "X has age Y. X is Y years old. X is aged Y."
    ),
    ONT.HAS_SEX: (
        "X has sex Y. X is Y. X is a Y patient."
    ),
    ONT.HAS_WEIGHT: (
        "X has weight Y. X weighs Y. X has a body weight of Y."
    ),
    ONT.HAS_HEIGHT: (
        "X has height Y. X stands Y tall. X is Y in height."
    ),
    ONT.HAS_OCCUPATION: (
        "X has occupation Y. X works as Y. X is employed as Y."
    ),
    ONT.HAS_PERSONAL_BACKGROUND: (
        "X has personal background Y. X has a social history of Y."
    ),
    ONT.HAS_FAMILY_HISTORY: (
        "X has family history of Y. X has a family member with Y."
    ),
}

# ---------------------------------------------------------------------------
# Valid candidate relations per entity-type pair.
# Limits the BioLinkBERT search space and reduces false positives.
# Where the same pair appears in both v1 and v2, relations are unioned.
# ---------------------------------------------------------------------------
_D   = ENT.DISEASE_DISORDER
_S   = ENT.SIGN_SYMPTOM
_SV  = ENT.SEVERITY
_M   = ENT.MEDICATION
_DO  = ENT.DOSAGE
_FR  = ENT.FREQUENCY
_AD  = ENT.ADMINISTRATION
_TP  = ENT.THERAPEUTIC_PROCEDURE
_DP  = ENT.DIAGNOSTIC_PROCEDURE
_LV  = ENT.LAB_VALUE
_BS  = ENT.BIOLOGICAL_STRUCTURE
_AR  = ENT.AREA
_DA  = ENT.DATE
_TI  = ENT.TIME
_DU  = ENT.DURATION
_OU  = ENT.OUTCOME
_FH  = ENT.FAMILY_HISTORY
_CE  = ENT.CLINICAL_EVENT
_HI  = ENT.HISTORY
_DD  = ENT.DETAILED_DESCRIPTION
_AC  = ENT.ACTIVITY
_SU  = ENT.SUBJECT
_AG  = ENT.AGE
_SE  = ENT.SEX
_WE  = ENT.WEIGHT
_HT  = ENT.HEIGHT
_OC  = ENT.OCCUPATION
_PB  = ENT.PERSONAL_BACKGROUND
_NL  = ENT.NONBIOLOGICAL_LOCATION
_CO  = ENT.COLOR
_SH  = ENT.SHAPE
_TX  = ENT.TEXTURE
_MA  = ENT.MASS
_VO  = ENT.VOLUME
_QN  = ENT.QUANTITATIVE_CONCEPT
_QL  = ENT.QUALITATIVE_CONCEPT

ENTITY_PAIR_RELATIONS: dict[tuple[str, str], list[str]] = {
    # --- Condition ↔ Condition ---
    (_D,  _D):  [ONT.ASSOCIATED_WITH, ONT.CAUSES],
    (_D,  _S):  [ONT.CAUSES, ONT.ASSOCIATED_WITH, ONT.HAS_SYMPTOM],
    (_S,  _D):  [ONT.CAUSES, ONT.ASSOCIATED_WITH, ONT.INDICATES],
    (_S,  _S):  [ONT.ASSOCIATED_WITH],

    # --- Condition detail ---
    (_D,  _SV): [ONT.HAS_SEVERITY],
    (_S,  _SV): [ONT.HAS_SEVERITY],
    # Appearance can attach to a lesion/finding tagged DISEASE_DISORDER/SIGN_SYMPTOM
    # too, not only BIOLOGICAL_STRUCTURE — e.g. "hypoechoic nodule", "irregular mass".
    # Syntax-only (amod/compound), extracted in nlp/re/spacy.py _extract_syntactic.
    (_D,  _CO): [ONT.HAS_COLOR],
    (_D,  _SH): [ONT.HAS_SHAPE],
    (_D,  _TX): [ONT.HAS_TEXTURE],
    (_S,  _CO): [ONT.HAS_COLOR],
    (_S,  _SH): [ONT.HAS_SHAPE],
    (_S,  _TX): [ONT.HAS_TEXTURE],

    # --- Treatment → Condition ---
    (_M,  _D):  [ONT.TREATS, ONT.CAUSES, ONT.GIVEN_FOR],
    (_M,  _S):  [ONT.TREATS, ONT.CAUSES],
    (_TP, _D):  [ONT.TREATS, ONT.CAUSES],
    (_TP, _S):  [ONT.TREATS, ONT.CAUSES],
    (_AC, _S):  [ONT.CAUSES, ONT.ASSOCIATED_WITH],
    (_AC, _D):  [ONT.CAUSES, ONT.ASSOCIATED_WITH],

    # --- Condition → Treatment (passive constructions) ---
    (_D,  _M):  [ONT.TREATS],
    (_D,  _TP): [ONT.TREATS],
    (_S,  _M):  [ONT.TREATS],
    (_S,  _TP): [ONT.TREATS],

    # --- Medication detail ---
    (_M,  _DO): [ONT.ADMINISTERED_AS, ONT.HAS_DOSAGE],
    (_M,  _FR): [ONT.ADMINISTERED_AS, ONT.HAS_FREQUENCY],
    (_M,  _AD): [ONT.ADMINISTERED_AS, ONT.HAS_ADMINISTRATION],
    (_M,  _DU): [ONT.DURATION_OF],

    # --- Therapeutic procedure detail ---
    (_TP, _BS): [ONT.PERFORMED_ON],
    (_TP, _AR): [ONT.PERFORMED_ON],
    (_TP, _DA): [ONT.OCCURS_AT],
    (_TP, _TI): [ONT.OCCURS_AT],
    (_TP, _DU): [ONT.DURATION_OF, ONT.HAS_DURATION],
    (_TP, _OU): [ONT.RESULTS_IN],

    # --- Diagnostics ---
    (_DP, _D):  [ONT.DIAGNOSES, ONT.REVEALS],
    (_DP, _S):  [ONT.DIAGNOSES, ONT.REVEALS],
    (_DP, _LV): [ONT.MEASURES],
    # A diagnostic procedure is performed on a body site just like a therapeutic one
    # ("USG tarczycy", "RTG klatki piersiowej", "MRI of the brain").
    (_DP, _BS): [ONT.PERFORMED_ON],
    (_DP, _AR): [ONT.PERFORMED_ON],
    (_DP, _DA): [ONT.OCCURS_AT],
    (_DP, _TI): [ONT.OCCURS_AT],
    (_DP, _DU): [ONT.HAS_DURATION],
    (_D,  _DP): [ONT.DIAGNOSES],
    (_LV, _D):  [ONT.INDICATES],
    (_LV, _S):  [ONT.INDICATES, ONT.QUANTIFIES],
    (_LV, _DP): [ONT.IS_VALUE_OF],
    (_D,  _LV): [ONT.INDICATES],
    (_S,  _LV): [ONT.QUANTIFIES],

    # --- Anatomy ---
    (_D,  _BS): [ONT.AFFECTS, ONT.LOCATED_IN],
    (_D,  _AR): [ONT.AFFECTS, ONT.LOCATED_IN],
    (_S,  _BS): [ONT.AFFECTS, ONT.LOCATED_IN],
    (_S,  _AR): [ONT.LOCATED_IN],
    (_BS, _AR): [ONT.LOCATED_IN],
    (_BS, _CO): [ONT.HAS_COLOR],
    (_BS, _SH): [ONT.HAS_SHAPE],
    (_BS, _TX): [ONT.HAS_TEXTURE],
    (_AR, _BS): [ONT.CONTAINS],

    # --- Measurements quantifying symptoms/disease ---
    (_WE, _S):  [ONT.QUANTIFIES],
    (_WE, _D):  [ONT.QUANTIFIES, ONT.INDICATES],
    (_MA, _S):  [ONT.QUANTIFIES],
    (_MA, _D):  [ONT.QUANTIFIES, ONT.INDICATES],
    (_VO, _S):  [ONT.QUANTIFIES],
    (_VO, _D):  [ONT.QUANTIFIES, ONT.INDICATES],
    (_QN, _S):  [ONT.QUANTIFIES],
    (_QN, _D):  [ONT.QUANTIFIES, ONT.INDICATES],

    # --- Reverse: anatomy/disease/symptom HAS mass / volume ---
    (_BS, _MA): [ONT.HAS_MASS, ONT.QUANTIFIES],
    (_D,  _MA): [ONT.HAS_MASS],
    (_S,  _MA): [ONT.HAS_MASS],
    (_BS, _VO): [ONT.HAS_VOLUME, ONT.QUANTIFIES],
    (_D,  _VO): [ONT.HAS_VOLUME],
    (_S,  _VO): [ONT.HAS_VOLUME],
    (_AR, _VO): [ONT.HAS_VOLUME],

    # --- Duration / time of symptoms / events ---
    (_DU, _S):  [ONT.DURATION_OF],
    (_DU, _D):  [ONT.DURATION_OF],
    (_DU, _TP): [ONT.DURATION_OF],
    (_DU, _M):  [ONT.DURATION_OF],
    (_DA, _S):  [ONT.DURATION_OF],
    (_DA, _D):  [ONT.DURATION_OF],
    (_DA, _TP): [ONT.DURATION_OF],
    (_TI, _S):  [ONT.DURATION_OF],
    (_TI, _D):  [ONT.DURATION_OF],
    # Reverse: disease/symptom → duration (passive constructions)
    (_D,  _DU): [ONT.DURATION_OF],
    (_S,  _DU): [ONT.DURATION_OF],
    (_D,  _DA): [ONT.DURATION_OF],
    (_S,  _DA): [ONT.DURATION_OF],

    # Note: SEVERITY is an ATTRIBUTE, reached only as the object of HAS_SEVERITY
    # (disease/symptom -> severity, extracted syntactically from amod/compound in
    # nlp/re/spacy.py). It is deliberately NOT a subject of INDICATES — "severe
    # indicates disease" is spurious when severity is just a qualifier.

    # --- Risk & history ---
    (_AG, _D):  [ONT.RISK_FACTOR_OF],
    (_FH, _D):  [ONT.RISK_FACTOR_OF, ONT.INDICATES],
    (_HI, _D):  [ONT.RISK_FACTOR_OF, ONT.ASSOCIATED_WITH],

    # --- Outcome ---
    (_OU, _D):  [ONT.INDICATES],
    (_OU, _TP): [ONT.INDICATES],
    (_OU, _M):  [ONT.INDICATES],

    # --- Qualitative descriptions ---
    (_QL, _S):  [ONT.INDICATES],
    (_QL, _D):  [ONT.INDICATES],

    # --- Clinical event ---
    (_CE, _DA): [ONT.OCCURS_AT],
    (_CE, _TI): [ONT.OCCURS_AT],
    (_CE, _DU): [ONT.HAS_DURATION],
    (_CE, _OU): [ONT.LEADS_TO],
    (_CE, _NL): [ONT.PERFORMED_IN],

    # --- History & narrative ---
    (_HI, _CE): [ONT.PREVIOUSLY_OCCURRED],
    (_DD, _CE): [ONT.DESCRIBES],
    (_AC, _CE): [ONT.RELATED_TO],

    # --- Detailed descriptions modify the described entity ---
    # e.g. "chronic"/"inflammatory" -> lesions, "hypoechoic" -> nodule.
    (_DD, _D):  [ONT.DESCRIBES],
    (_DD, _S):  [ONT.DESCRIBES],
    (_DD, _BS): [ONT.DESCRIBES],
    (_DD, _TP): [ONT.DESCRIBES],
    (_DD, _DP): [ONT.DESCRIBES],

    # --- Subject demographics ---
    (_SU, _AG): [ONT.HAS_AGE],
    (_SU, _SE): [ONT.HAS_SEX],
    (_SU, _WE): [ONT.HAS_WEIGHT],
    (_SU, _HT): [ONT.HAS_HEIGHT],
    (_SU, _OC): [ONT.HAS_OCCUPATION],
    (_SU, _PB): [ONT.HAS_PERSONAL_BACKGROUND],
    (_SU, _FH): [ONT.HAS_FAMILY_HISTORY],
    (_SU, _D):  [ONT.HISTORY_OF],
}

# ---------------------------------------------------------------------------
# All valid triplets, sorted alphabetically by relation label.
# Format: (SUBJECT_TYPE)  --[RELATION]-->  (OBJECT_TYPE)
# ---------------------------------------------------------------------------
#
# ADMINISTERED_AS                                                      [v1]
#   (MEDICATION)              --[ADMINISTERED_AS]-->     (ADMINISTRATION)
#   (MEDICATION)              --[ADMINISTERED_AS]-->     (DOSAGE)
#   (MEDICATION)              --[ADMINISTERED_AS]-->     (FREQUENCY)
#
# AFFECTS                                                              [v1]
#   (DISEASE_DISORDER)        --[AFFECTS]-->             (AREA)
#   (DISEASE_DISORDER)        --[AFFECTS]-->             (BIOLOGICAL_STRUCTURE)
#   (SIGN_SYMPTOM)            --[AFFECTS]-->             (BIOLOGICAL_STRUCTURE)
#
# ASSOCIATED_WITH                                                      [v1]
#   (ACTIVITY)                --[ASSOCIATED_WITH]-->     (DISEASE_DISORDER)
#   (ACTIVITY)                --[ASSOCIATED_WITH]-->     (SIGN_SYMPTOM)
#   (DISEASE_DISORDER)        --[ASSOCIATED_WITH]-->     (DISEASE_DISORDER)
#   (DISEASE_DISORDER)        --[ASSOCIATED_WITH]-->     (SIGN_SYMPTOM)
#   (HISTORY)                 --[ASSOCIATED_WITH]-->     (DISEASE_DISORDER)
#   (SIGN_SYMPTOM)            --[ASSOCIATED_WITH]-->     (DISEASE_DISORDER)
#   (SIGN_SYMPTOM)            --[ASSOCIATED_WITH]-->     (SIGN_SYMPTOM)
#
# CAUSES                                                               [both]
#   (ACTIVITY)                --[CAUSES]-->              (DISEASE_DISORDER)
#   (ACTIVITY)                --[CAUSES]-->              (SIGN_SYMPTOM)
#   (DISEASE_DISORDER)        --[CAUSES]-->              (DISEASE_DISORDER)
#   (DISEASE_DISORDER)        --[CAUSES]-->              (SIGN_SYMPTOM)
#   (MEDICATION)              --[CAUSES]-->              (DISEASE_DISORDER)
#   (MEDICATION)              --[CAUSES]-->              (SIGN_SYMPTOM)
#   (SIGN_SYMPTOM)            --[CAUSES]-->              (DISEASE_DISORDER)
#   (THERAPEUTIC_PROCEDURE)   --[CAUSES]-->              (DISEASE_DISORDER)
#   (THERAPEUTIC_PROCEDURE)   --[CAUSES]-->              (SIGN_SYMPTOM)
#
# CONTAINS                                                             [v2]
#   (AREA)                    --[CONTAINS]-->            (BIOLOGICAL_STRUCTURE)
#
# DESCRIBES                                                            [v2]
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (BIOLOGICAL_STRUCTURE)
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (CLINICAL_EVENT)
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (DIAGNOSTIC_PROCEDURE)
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (DISEASE_DISORDER)
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (SIGN_SYMPTOM)
#   (DETAILED_DESCRIPTION)    --[DESCRIBES]-->           (THERAPEUTIC_PROCEDURE)
#
# DIAGNOSES                                                            [v1]
#   (DIAGNOSTIC_PROCEDURE)    --[DIAGNOSES]-->           (DISEASE_DISORDER)
#   (DIAGNOSTIC_PROCEDURE)    --[DIAGNOSES]-->           (SIGN_SYMPTOM)
#   (DISEASE_DISORDER)        --[DIAGNOSES]-->           (DIAGNOSTIC_PROCEDURE)
#
# DURATION_OF                                                          [v1]
#   (DATE)                    --[DURATION_OF]-->         (DISEASE_DISORDER)
#   (DATE)                    --[DURATION_OF]-->         (SIGN_SYMPTOM)
#   (DATE)                    --[DURATION_OF]-->         (THERAPEUTIC_PROCEDURE)
#   (DISEASE_DISORDER)        --[DURATION_OF]-->         (DATE)
#   (DISEASE_DISORDER)        --[DURATION_OF]-->         (DURATION)
#   (DURATION)                --[DURATION_OF]-->         (DISEASE_DISORDER)
#   (DURATION)                --[DURATION_OF]-->         (MEDICATION)
#   (DURATION)                --[DURATION_OF]-->         (SIGN_SYMPTOM)
#   (DURATION)                --[DURATION_OF]-->         (THERAPEUTIC_PROCEDURE)
#   (MEDICATION)              --[DURATION_OF]-->         (DURATION)
#   (SIGN_SYMPTOM)            --[DURATION_OF]-->         (DATE)
#   (SIGN_SYMPTOM)            --[DURATION_OF]-->         (DURATION)
#   (THERAPEUTIC_PROCEDURE)   --[DURATION_OF]-->         (DURATION)
#   (TIME)                    --[DURATION_OF]-->         (DISEASE_DISORDER)
#   (TIME)                    --[DURATION_OF]-->         (SIGN_SYMPTOM)
#
# GIVEN_FOR                                                            [v2]
#   (MEDICATION)              --[GIVEN_FOR]-->           (DISEASE_DISORDER)
#
# HAS_ADMINISTRATION                                                   [v2]
#   (MEDICATION)              --[HAS_ADMINISTRATION]--> (ADMINISTRATION)
#
# HAS_AGE                                                              [v2]
#   (SUBJECT)                 --[HAS_AGE]-->             (AGE)
#
# HAS_COLOR                                                            [v2]
#   (BIOLOGICAL_STRUCTURE)    --[HAS_COLOR]-->           (COLOR)
#   (DISEASE_DISORDER)        --[HAS_COLOR]-->           (COLOR)
#   (SIGN_SYMPTOM)            --[HAS_COLOR]-->           (COLOR)
#
# HAS_DOSAGE                                                           [v2]
#   (MEDICATION)              --[HAS_DOSAGE]-->          (DOSAGE)
#
# HAS_DURATION                                                         [v2]
#   (CLINICAL_EVENT)          --[HAS_DURATION]-->        (DURATION)
#   (DIAGNOSTIC_PROCEDURE)    --[HAS_DURATION]-->        (DURATION)
#   (THERAPEUTIC_PROCEDURE)   --[HAS_DURATION]-->        (DURATION)
#
# HAS_FAMILY_HISTORY                                                   [v2]
#   (SUBJECT)                 --[HAS_FAMILY_HISTORY]--> (FAMILY_HISTORY)
#
# HAS_FREQUENCY                                                        [v2]
#   (MEDICATION)              --[HAS_FREQUENCY]-->       (FREQUENCY)
#
# HAS_HEIGHT                                                           [v2]
#   (SUBJECT)                 --[HAS_HEIGHT]-->          (HEIGHT)
#
# HAS_MASS                                                             [new]
#   (BIOLOGICAL_STRUCTURE)    --[HAS_MASS]-->            (MASS)
#   (DISEASE_DISORDER)        --[HAS_MASS]-->            (MASS)
#   (SIGN_SYMPTOM)            --[HAS_MASS]-->            (MASS)
#
# HAS_OCCUPATION                                                       [v2]
#   (SUBJECT)                 --[HAS_OCCUPATION]-->      (OCCUPATION)
#
# HAS_PERSONAL_BACKGROUND                                              [v2]
#   (SUBJECT)                 --[HAS_PERSONAL_BACKGROUND]--> (PERSONAL_BACKGROUND)
#
# HAS_SEX                                                              [v2]
#   (SUBJECT)                 --[HAS_SEX]-->             (SEX)
#
# HAS_SEVERITY                                                         [v2]
#   (DISEASE_DISORDER)        --[HAS_SEVERITY]-->        (SEVERITY)
#   (SIGN_SYMPTOM)            --[HAS_SEVERITY]-->        (SEVERITY)
#
# HAS_SHAPE                                                            [v2]
#   (BIOLOGICAL_STRUCTURE)    --[HAS_SHAPE]-->           (SHAPE)
#   (DISEASE_DISORDER)        --[HAS_SHAPE]-->           (SHAPE)
#   (SIGN_SYMPTOM)            --[HAS_SHAPE]-->           (SHAPE)
#
# HAS_SYMPTOM                                                          [v2]
#   (DISEASE_DISORDER)        --[HAS_SYMPTOM]-->         (SIGN_SYMPTOM)
#
# HAS_TEXTURE                                                          [v2]
#   (BIOLOGICAL_STRUCTURE)    --[HAS_TEXTURE]-->         (TEXTURE)
#   (DISEASE_DISORDER)        --[HAS_TEXTURE]-->         (TEXTURE)
#   (SIGN_SYMPTOM)            --[HAS_TEXTURE]-->         (TEXTURE)
#
# HAS_VOLUME                                                           [new]
#   (AREA)                    --[HAS_VOLUME]-->          (VOLUME)
#   (BIOLOGICAL_STRUCTURE)    --[HAS_VOLUME]-->          (VOLUME)
#   (DISEASE_DISORDER)        --[HAS_VOLUME]-->          (VOLUME)
#   (SIGN_SYMPTOM)            --[HAS_VOLUME]-->          (VOLUME)
#
# HAS_WEIGHT                                                           [v2]
#   (SUBJECT)                 --[HAS_WEIGHT]-->          (WEIGHT)
#
# HISTORY_OF                                                           [v2]
#   (SUBJECT)                 --[HISTORY_OF]-->          (DISEASE_DISORDER)
#
# INDICATES                                                            [both]
#   (DISEASE_DISORDER)        --[INDICATES]-->           (LAB_VALUE)
#   (FAMILY_HISTORY)          --[INDICATES]-->           (DISEASE_DISORDER)
#   (LAB_VALUE)               --[INDICATES]-->           (DISEASE_DISORDER)
#   (LAB_VALUE)               --[INDICATES]-->           (SIGN_SYMPTOM)
#   (MASS)                    --[INDICATES]-->           (DISEASE_DISORDER)
#   (OUTCOME)                 --[INDICATES]-->           (DISEASE_DISORDER)
#   (OUTCOME)                 --[INDICATES]-->           (MEDICATION)
#   (OUTCOME)                 --[INDICATES]-->           (THERAPEUTIC_PROCEDURE)
#   (QUALITATIVE_CONCEPT)     --[INDICATES]-->           (DISEASE_DISORDER)
#   (QUALITATIVE_CONCEPT)     --[INDICATES]-->           (SIGN_SYMPTOM)
#   (QUANTITATIVE_CONCEPT)    --[INDICATES]-->           (DISEASE_DISORDER)
#   (SIGN_SYMPTOM)            --[INDICATES]-->           (DISEASE_DISORDER)
#   (WEIGHT)                  --[INDICATES]-->           (DISEASE_DISORDER)
#
# IS_VALUE_OF                                                          [v2]
#   (LAB_VALUE)               --[IS_VALUE_OF]-->         (DIAGNOSTIC_PROCEDURE)
#
# LEADS_TO                                                             [v2]
#   (CLINICAL_EVENT)          --[LEADS_TO]-->            (OUTCOME)
#
# LOCATED_IN                                                           [both]
#   (BIOLOGICAL_STRUCTURE)    --[LOCATED_IN]-->          (AREA)
#   (DISEASE_DISORDER)        --[LOCATED_IN]-->          (AREA)
#   (DISEASE_DISORDER)        --[LOCATED_IN]-->          (BIOLOGICAL_STRUCTURE)
#   (SIGN_SYMPTOM)            --[LOCATED_IN]-->          (AREA)
#   (SIGN_SYMPTOM)            --[LOCATED_IN]-->          (BIOLOGICAL_STRUCTURE)
#
# MEASURES                                                             [v2]
#   (DIAGNOSTIC_PROCEDURE)    --[MEASURES]-->            (LAB_VALUE)
#
# OCCURS_AT                                                            [v2]
#   (CLINICAL_EVENT)          --[OCCURS_AT]-->           (DATE)
#   (CLINICAL_EVENT)          --[OCCURS_AT]-->           (TIME)
#   (DIAGNOSTIC_PROCEDURE)    --[OCCURS_AT]-->           (DATE)
#   (DIAGNOSTIC_PROCEDURE)    --[OCCURS_AT]-->           (TIME)
#   (THERAPEUTIC_PROCEDURE)   --[OCCURS_AT]-->           (DATE)
#   (THERAPEUTIC_PROCEDURE)   --[OCCURS_AT]-->           (TIME)
#
# PERFORMED_IN                                                         [v2]
#   (CLINICAL_EVENT)          --[PERFORMED_IN]-->        (NONBIOLOGICAL_LOCATION)
#
# PERFORMED_ON                                                         [v2]
#   (DIAGNOSTIC_PROCEDURE)    --[PERFORMED_ON]-->        (AREA)
#   (DIAGNOSTIC_PROCEDURE)    --[PERFORMED_ON]-->        (BIOLOGICAL_STRUCTURE)
#   (THERAPEUTIC_PROCEDURE)   --[PERFORMED_ON]-->        (AREA)
#   (THERAPEUTIC_PROCEDURE)   --[PERFORMED_ON]-->        (BIOLOGICAL_STRUCTURE)
#
# PREVIOUSLY_OCCURRED                                                  [v2]
#   (HISTORY)                 --[PREVIOUSLY_OCCURRED]--> (CLINICAL_EVENT)
#
# QUANTIFIES                                                           [v1]
#   (BIOLOGICAL_STRUCTURE)    --[QUANTIFIES]-->          (MASS)
#   (BIOLOGICAL_STRUCTURE)    --[QUANTIFIES]-->          (VOLUME)
#   (LAB_VALUE)               --[QUANTIFIES]-->          (SIGN_SYMPTOM)
#   (MASS)                    --[QUANTIFIES]-->          (DISEASE_DISORDER)
#   (MASS)                    --[QUANTIFIES]-->          (SIGN_SYMPTOM)
#   (QUANTITATIVE_CONCEPT)    --[QUANTIFIES]-->          (DISEASE_DISORDER)
#   (QUANTITATIVE_CONCEPT)    --[QUANTIFIES]-->          (SIGN_SYMPTOM)
#   (SIGN_SYMPTOM)            --[QUANTIFIES]-->          (LAB_VALUE)
#   (VOLUME)                  --[QUANTIFIES]-->          (DISEASE_DISORDER)
#   (VOLUME)                  --[QUANTIFIES]-->          (SIGN_SYMPTOM)
#   (WEIGHT)                  --[QUANTIFIES]-->          (DISEASE_DISORDER)
#   (WEIGHT)                  --[QUANTIFIES]-->          (SIGN_SYMPTOM)
#
# RELATED_TO                                                           [v2]
#   (ACTIVITY)                --[RELATED_TO]-->          (CLINICAL_EVENT)
#
# RESULTS_IN                                                           [v2]
#   (THERAPEUTIC_PROCEDURE)   --[RESULTS_IN]-->          (OUTCOME)
#
# REVEALS                                                              [v2]
#   (DIAGNOSTIC_PROCEDURE)    --[REVEALS]-->             (DISEASE_DISORDER)
#   (DIAGNOSTIC_PROCEDURE)    --[REVEALS]-->             (SIGN_SYMPTOM)
#
# RISK_FACTOR_OF                                                       [v1]
#   (AGE)                     --[RISK_FACTOR_OF]-->      (DISEASE_DISORDER)
#   (FAMILY_HISTORY)          --[RISK_FACTOR_OF]-->      (DISEASE_DISORDER)
#   (HISTORY)                 --[RISK_FACTOR_OF]-->      (DISEASE_DISORDER)
#
# TREATS                                                               [both]
#   (DISEASE_DISORDER)        --[TREATS]-->              (MEDICATION)
#   (DISEASE_DISORDER)        --[TREATS]-->              (THERAPEUTIC_PROCEDURE)
#   (MEDICATION)              --[TREATS]-->              (DISEASE_DISORDER)
#   (MEDICATION)              --[TREATS]-->              (SIGN_SYMPTOM)
#   (SIGN_SYMPTOM)            --[TREATS]-->              (MEDICATION)
#   (SIGN_SYMPTOM)            --[TREATS]-->              (THERAPEUTIC_PROCEDURE)
#   (THERAPEUTIC_PROCEDURE)   --[TREATS]-->              (DISEASE_DISORDER)
#   (THERAPEUTIC_PROCEDURE)   --[TREATS]-->              (SIGN_SYMPTOM)
#
# ---------------------------------------------------------------------------
# Summary: 43 relation labels, 140 valid (subject, relation, object) triplets.
# ---------------------------------------------------------------------------
