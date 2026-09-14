# Clinical text to knowledge graph

A pipeline that turns free-text clinical records into a Neo4j knowledge graph. Polish
source text is translated to English, medical entities and the relations between them
are extracted, every entity is checked for negation/uncertainty/family/historical
context and linked to a UMLS concept, and the result is written out as graph CSVs and
loaded into Neo4j. Entity extraction and relation extraction each run several
independent methods over the same input and merge their output by method priority, so
the methods can be compared against one another instead of being trusted individually.

## Requirements

- Python 3.10 or newer (the code uses PEP 604 `X | Y` annotations)
- A running Neo4j instance, for the `neo4j_import` step
- A Google Gemini API key, for the optional LLM steps and for the holistic variant
- Enough disk for the downloaded models: the scispaCy UMLS knowledge base alone is
  about 1 GB, and torch with the three transformer models adds several more

## Setup

There is no `requirements.txt` under version control. `setup_venv.py` builds the
environment and generates one:

```powershell
python setup_venv.py
.\.venv\Scripts\activate
```

It creates `.venv` and installs everything the pipeline imports: spaCy with scispaCy
and medSpaCy, the `en_core_sci_md` and `en_core_web_sm` models,
torch/transformers/datasets, and the packages that only the optional steps need,
`python-dotenv` for `.env` loading, `google-genai` for the LLM steps, `neo4j` for the
graph import and `graphviz` for the dependency-graph figure. It then pre-downloads the
MarianMT translation model, BioLinkBERT and the scispaCy UMLS knowledge base (about
1 GB), and freezes the result into `requirements.txt`.

The one thing pip cannot supply is the Graphviz binaries that the Python `graphviz`
wrapper renders through. Install those from [graphviz.org](https://graphviz.org/download/)
if you need `testing/dep_tree_figure.py`; nothing else in the pipeline touches them.

Secrets are read from the environment, never from `config.py`. Put them in a `.env`
file in the repository root (git-ignored); `config.py` loads it on import, so every
entry point picks it up:

```
GEMINI_API_KEY=...
NEO4J_PASSWORD=...
```

`NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_DATABASE` can be overridden the same way;
they default to `bolt://localhost:7687`, `neo4j` and `neo4j`.

## Running

```powershell
python orchestrator.py
```

The `STEPS` dictionary at the top of [orchestrator.py](orchestrator.py) switches
stages on and off with booleans. The two kinds of switch behave differently.
`preprocessing`, `visualization` and `neo4j_import` skip their stage outright, print
`[SKIP]` and leave whatever the previous run wrote on disk, so a later stage can be
re-run on its own. The `ner_*` and `re_*` switches instead select which methods run
inside the NER and RE stages, and those stages always execute: turning every method
off writes an empty result rather than keeping the previous one.

| Step             | What it does                                                             |
| ---------------- | ------------------------------------------------------------------------ |
| `preprocessing`  | parses the raw input and translates Polish to English with MarianMT       |
| `ner_neural`     | `blaze999/Medical-NER` token classification                              |
| `ner_gazetteer`  | spaCy `PhraseMatcher` lexicons plus regex extractors                     |
| `ner_llm`        | LLM entity extraction as a peer method, needs an API key                 |
| `ner_context`    | medSpaCy ConText assertion flags: negated, uncertain, family, historical |
| `ner_umls`       | scispaCy UMLS linking, adds CUI, canonical name and linker score         |
| `re_biolink`     | BioLinkBERT zero-shot relation classification over candidate pairs       |
| `re_spacy`       | rule-based extraction from lexical triggers and the dependency parse     |
| `re_llm`         | LLM relation extraction as a peer method, needs an API key               |
| `visualization`  | builds the graph CSVs from the extraction output                         |
| `neo4j_import`   | loads those CSVs into Neo4j                                             |

`preprocessing` skips itself when its output file already exists, so the expensive
translation pass runs once per corpus.

The NER and RE stages also have their own command-line entry points, useful for
re-running one stage with different switches:

```powershell
python -m nlp.ner.tasks --llm --no-gazetteer
python -m nlp.re.tasks --no-biolink
```

To follow a single sentence through the whole pipeline, with the entities, the
assertion flags, the relations and the resulting FHIR layer printed to the terminal:

```powershell
python -m testing.test_sentence "CT revealed a mass in the left lung."
python -m testing.test_sentence --holistic "..."
```

## Configuration

[config.py](config.py) holds the model names, every score cutoff and the data paths.
Two switches change the shape of the run rather than tune it:

**`PIPELINE_MODE`** picks the extraction variant. `"pipeline"` runs the modular path:
the NER methods first, then the RE methods over their output. `"holistic"` replaces
both with a single generative step ([nlp/holistic.py](nlp/holistic.py)) that
classifies entities and relations jointly in one ontology-constrained call, then
reuses the same ConText and UMLS enrichment and writes the same CSVs. In holistic
mode the individual `ner_*`/`re_*` method toggles are ignored; `ner_context` and
`ner_umls` still apply.

**`GRAPH_MODE`** picks the graph model. `"concept"` builds a concept-only graph:
entities become nodes merged by CUI, relations become edges, written to
`visualization/nodes.csv` and `visualization/relationships.csv`. `"fhir"` builds a
FHIR-style reification graph instead: every entity mention becomes a reified fact
resource (`Condition`, `Observation`, `Procedure`, ...) attached to a patient, the
shared CUI stays a `:Concept` node, and provenance rides on the fact as source id and
character offsets. It writes `fhir_patients.csv`, `fhir_concepts.csv`,
`fhir_facts.csv` and `fhir_edges.csv`, and is driven by `FHIR_RESOURCE_MAP` and
`RELATION_FHIR_MAP` in `config.py`.

Output paths and the Neo4j connection live in
[visualization/config.py](visualization/config.py).

The entity ontology (41 labels, with the gazetteer lexicons and regex extractors) is
[nlp/ner/ontology.py](nlp/ner/ontology.py); the relation ontology (43 labels, with the
trigger patterns and the entity-type pairs each relation is licensed for) is
[nlp/re/ontology.py](nlp/re/ontology.py).

## Data

`config.DATA_DIR` points at `data/`, or at `test_data/` when `USE_TEST_DATA = True`.
Both hold the same file names: the corpus input, the structured CSV produced by
preprocessing, and the NER/RE result CSVs. Everything under those directories is
git-ignored except the two inputs below, so the generated CSVs stay out of version
control.

Both committed inputs carry the same name, `structured.csv`, and the same columns,
`id`, `text`, `text_en`, `specjalizacja` and `kategoria`, already translated and in
the shape the NER stage expects. `data/structured.csv` is the full corpus of 3282
records and `test_data/structured.csv` a 46-record subset for quick runs, so flipping
`USE_TEST_DATA` is all it takes to switch between them.

Their `text_en` did not come from the MarianMT `preprocessing` step; it was translated
separately, and it is noticeably more faithful than what MarianMT produces on this
corpus. `preprocessing` skips itself whenever `STRUCTURED_CSV` already exists, which
is what keeps that translation from being overwritten. Delete the file before
enabling that step, and expect a different, weaker translation back.

The NER and RE stages each write two CSVs, a deduplicated one and a `_raw` one that keeps the
pre-dedup union of all methods including the mentions a score threshold rejected. The
method-comparison statistics read the raw files, and a threshold can be re-tuned
against a saved run without extracting again.

## Statistics

`statistics/` holds one script per metric, grouped into `ner/` and `re/`. Each writes
its plot or table next to itself and can be run on its own. To run all of them and
snapshot the output for later comparison between runs:

```powershell
python statistics/run_all.py "run name"
```

Results land in `stat_output/<run name>/`, mirroring the source folder structure, with
a `run_info.txt` listing what passed and how long it took.

`scripts/save_run.py "run name"` does the same for a pipeline run, copying the input,
the NER/RE CSVs and the graph CSVs into `prev_runs/<run name>/`. `.gitignore` keeps
that directory out of version control except for the run names listed there
explicitly, so archiving a run does not commit tens of megabytes by accident.

## Repository layout

| Path             | Contents                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| `orchestrator.py` | the pipeline driver and the `STEPS` switches                            |
| `config.py`      | models, thresholds, data paths, the two mode switches                    |
| `nlp/`           | extraction: `ner/`, `re/`, the holistic variant, the LLM client          |
| `preprocessing/` | input parsing and MarianMT Polish to English translation                 |
| `visualization/` | graph building, both modes, and the Neo4j import                         |
| `statistics/`    | per-metric scripts and the runner that snapshots their output            |
| `testing/`       | single-sentence runs, parse/NER inspection, one dependency-graph figure  |
| `scripts/`       | one-off corpus conversions and run archiving                             |
| `docs/`          | a self-contained Text2Cypher prompt for querying the built graph         |
| `test_data/`     | the committed sample input                                               |
| `prev_runs/`     | archived outputs of earlier full runs                                    |
