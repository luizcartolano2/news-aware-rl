"""This script merges all asset price CSV files in the specified directory into a single CSV file.
It reads each CSV file, renames the 'Price' column to the filename (without extension),
and merges them on the 'Date' column. The final DataFrame is sorted by date and saved to a consolidated CSV file.

"""
import pandas as pd
import os
from constants import DATA_PATH

# Define paths
PRICE_DIR = f"{DATA_PATH}/prices"
OUTPUT_FILE = f"{DATA_PATH}/consolidated/asset_prices.csv"

if __name__ == "__main__":
    # Check if the price directory exists
    if not os.path.exists(PRICE_DIR):
        raise FileNotFoundError(f"Price directory {PRICE_DIR} does not exist.")

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    # Read all CSV files in the price directory
    all_prices = []

    # Loop through each file in the PRICE_DIR
    for filename in os.listdir(PRICE_DIR):
        # Check if the file is a CSV
        if filename.endswith(".csv"):
            # Read the CSV file and parse dates
            asset_df = pd.read_csv(os.path.join(PRICE_DIR, filename), parse_dates=["Date"])
            # Rename the 'Price' column to the asset name (filename without extension)
            asset_name = filename.replace(".csv", "")
            asset_df = asset_df.rename(columns={"Price": asset_name})
            # Select only the 'Date' and the renamed asset price column
            all_prices.append(asset_df[["Date", asset_name]])

    # Merge on "date"
    merged = all_prices[0]
    for asset_df in all_prices[1:]:
        merged = pd.merge(merged, asset_df, on="Date", how="outer")

    merged = merged.sort_values("Date").dropna()
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved merged prices to {OUTPUT_FILE} with shape {merged.shape}")
