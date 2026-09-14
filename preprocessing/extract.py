"""Module extracts data from text file or CSV file"""
def run(input_path: str):
    print(f"Extracting data from {input_path}")
    # Example: read file
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    return data
