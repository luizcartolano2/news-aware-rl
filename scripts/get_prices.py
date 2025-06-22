""" Script to fetch and save financial index prices.
This script retrieves historical price data for the IBOVESPA index, BRL/USD exchange rate, and IMAB11 ETF,
and saves them as CSV files in the specified data directory.

"""
import os

from constants import DATA_PATH
from getters.prices import get_index


if __name__ == "__main__":
    # fetch historical prices for the IBOVESPA index, BRL/USD exchange rate, and IMAB11 ETF
    ibov = get_index("^BVSP", start="2000-01-01", end="2025-06-01")
    brl = get_index("BRL=X", start="2000-01-01", end="2025-06-01")
    imab = get_index("IMAB11.SA", start="2000-01-01", end="2025-06-01")

    # ensure the data directory exists
    os.makedirs(f"{DATA_PATH}/prices", exist_ok=True)

    # save the fetched data to CSV files
    ibov.to_csv(f"{DATA_PATH}/prices/ibov.csv")
    brl.to_csv(f"{DATA_PATH}/prices/brl_usd.csv")
    imab.to_csv(f"{DATA_PATH}/prices/imab11.csv")
