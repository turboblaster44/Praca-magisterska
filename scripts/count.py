import pandas as pd

# Load CSV
df = pd.read_csv("data/corpus_summary_all.csv", sep=";")

# Count occurrences of each ID
id_counts = df["id"].value_counts()

# Print results
for id_value, count in id_counts.items():
    print(f"ID {id_value}: {count} times")

# Optional: save to CSV
id_counts.reset_index().rename(
    columns={"index": "id", "id": "count"}
).to_csv("id_counts.csv", sep=";", index=False)

print("\nSaved counts to id_counts.csv")