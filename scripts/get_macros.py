""" Script to download macroeconomic data from the BCB (Banco Central do Brasil) API.

"""
import os
from constants import DATA_PATH
from getters.macros import download_bcb_series

if __name__ == "__main__":
    # download macroeconomic data from BCB
    selic = download_bcb_series(432, start_date="01/01/2000", end_date="31/12/2004")
    ipca = download_bcb_series(433, start_date="01/01/2000", end_date="01/06/2025")
    gdp = download_bcb_series(7326, start_date="01/01/2000", end_date="01/06/2025")

    # ensure the data directory exists
    os.makedirs(f"{DATA_PATH}/macro", exist_ok=True)

    # save the data to CSV files
    selic.to_csv(f"{DATA_PATH}/macro/selic3.csv")
    ipca.to_csv(f"{DATA_PATH}/macro/ipca.csv")
    gdp.to_csv(f"{DATA_PATH}/macro/gdp.csv")
