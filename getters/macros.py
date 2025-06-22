"""Download time series from the Central Bank of Brazil (BCB) SGS API.
This module provides a function to download time series data from the BCB SGS API.
It allows you to specify the series code, start date, and end date.
The data is returned as a pandas DataFrame with date and value columns.

Example usage:
```python
from getters.macros import download_bcb_series
df = download_bcb_series(series_code=4189, start_date='01/01/2020', end_date='31/12/2020')
df.head()
```
"""
import pandas as pd
from datetime import datetime


def download_bcb_series(series_code, start_date='01/01/2000', end_date=None):
    """
    Download a time series from the Central Bank of Brazil (BCB) SGS API.

    :param series_code: Integer. The BCB series code (e.g., 4189 for SELIC).
    :param start_date: String. Start date in dd/mm/yyyy format.
    :param end_date: String. End date in dd/mm/yyyy format. Defaults to today.
    :return: DataFrame with date and value.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%d/%m/%Y')
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=csv&dataInicial={start_date}&dataFinal={end_date}'
    print(url)
    df = pd.read_csv(url, sep=';')
    df['date'] = pd.to_datetime(df['data'], dayfirst=True)
    df['value'] = pd.to_numeric(df['valor'].str.replace(',', '.'), errors='coerce')
    return df[['date', 'value']]
