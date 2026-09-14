import re
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from config import MARIANMT_MODEL_PL_EN

# Load model once
tokenizer = MarianTokenizer.from_pretrained(MARIANMT_MODEL_PL_EN)
model = MarianMTModel.from_pretrained(MARIANMT_MODEL_PL_EN)


def translate(text: str) -> str:
    """
    Translate Polish text to English using MarianMT
    """
    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    outputs = model.generate(**inputs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def clean_text(value: str) -> str:
    """
    Normalize text:
    - Strip whitespace
    - Remove outer quotes if present
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    value = str(value).strip()

    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()

    return value


def run(raw_text: str, data_format: str = "txt") -> pd.DataFrame:
    """
    Parse records formatted as:

    ID
    TEXT
        -   1. SPECJALIZACJA   KATEGORIA

    Or process CSV data with 'phrase' column
    """
    if data_format == "csv":
        return run_csv(raw_text)
    else:
        return run_txt(raw_text)


def run_txt(raw_text: str) -> pd.DataFrame:
    """
    Parse records formatted as:

    ID
    TEXT
        -   1. SPECJALIZACJA   KATEGORIA
    """

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    records = []
    i = 0

    while i < len(lines):

        record_id = int(lines[i])
        i += 1

        text_lines = []

        while i < len(lines) and not lines[i].startswith("-"):
            text_lines.append(lines[i])
            i += 1

        text_pl = clean_text(" ".join(text_lines))

        if i >= len(lines):
            raise ValueError("Missing metadata line for record")

        metadata_line = lines[i]
        i += 1

        match = re.search(r"\d+\.\s*(.+?)\s{2,}(.+)$", metadata_line)

        if not match:
            raise ValueError(f"Invalid metadata format: {metadata_line}")

        specjalizacja = clean_text(match.group(1))
        kategoria = clean_text(match.group(2))

        text_en = translate(text_pl)

        records.append({
            "id": record_id,
            "text": text_pl,
            "text_en": text_en,
            "specjalizacja": specjalizacja,
            "kategoria": kategoria
        })

    return pd.DataFrame(records)


def run_csv(csv_content: str) -> pd.DataFrame:
    """
    Process CSV data by reading just the 'phrase' column and handling it
    the same way as default data.txt
    """
    # Read the CSV content
    from io import StringIO
    df_csv = pd.read_csv(StringIO(csv_content), sep=';')
    # Each phrase was read by ~16 speakers, so the source has ~16x duplicate rows.
    # Assign the result back (the previous call dropped nothing) so every unique
    # sentence is translated/processed once instead of ~16 times.
    df_csv = df_csv.drop_duplicates(subset=["phrase"]).reset_index(drop=True)
    # Extract the 'phrase' column
    phrases = df_csv['phrase'].tolist()

    # Create records with dummy IDs and default values for specjalizacja and kategoria
    records = []
    for i, phrase in enumerate(phrases):
        text_pl = clean_text(phrase)
        text_en = translate(text_pl)

        records.append({
            "id": i + 1,  # Generate dummy IDs
            "text": text_pl,
            "text_en": text_en,
            "specjalizacja": "Unknown",  # Default value
            "kategoria": "Unknown"       # Default value
        })

    return pd.DataFrame(records)