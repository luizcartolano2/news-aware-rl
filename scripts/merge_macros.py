""" This script merges all macroeconomic indicator CSV files into a single DataFrame.
It reads each CSV file, renames the 'value' column to the filename (without extension),
and merges them on the 'date' column. The final DataFrame is sorted by date and saved to a consolidated CSV file.

"""
import pandas as pd
import os
from constants import DATA_PATH

# Define the directory containing macroeconomic indicator CSV files and the output file path
MACRO_DIR = f"{DATA_PATH}/macro"
OUTPUT_FILE = f"{DATA_PATH}/consolidated/macro_indicators.csv"

if __name__ == "__main__":
    # Check if the macro directory exists
    if not os.path.exists(MACRO_DIR):
        raise FileNotFoundError(f"Macro directory {MACRO_DIR} does not exist.")

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    # Read all CSV files in the macro directory
    all_macro = []

    # Loop through each file in the macro_dir
    for filename in os.listdir(MACRO_DIR):
        # Check if the file is a CSV
        if filename.endswith(".csv"):
            # Read the CSV file and parse dates
            macro_df = pd.read_csv(os.path.join(MACRO_DIR, filename), parse_dates=["date"])
            # Rename the 'value' column to the filename (without extension)
            var_name = filename.replace(".csv", "")
            macro_df = macro_df.rename(columns={"value": var_name})
            # Select only the 'date' and the variable column
            all_macro.append(macro_df[["date", var_name]])

    # Merge on "date"
    merged = all_macro[0]
    for macro_df in all_macro[1:]:
        merged = pd.merge(merged, macro_df, on="date", how="outer")

    # Sort by date and drop rows with NaN values
    merged = merged.sort_values('date').fillna(method='ffill')
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved merged macro to {OUTPUT_FILE} with shape {merged.shape}")
