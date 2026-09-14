import pandas as pd

"""Module saves the original text input into a readable csv format"""

def run(df: pd.DataFrame, output_path: str) -> str:
    """
    Saves the transformed DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by transform step
    output_path : str
        Path to output CSV file

    Returns
    -------
    str
        Path to saved file
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"CSV successfully saved to {output_path}")

    return output_path
