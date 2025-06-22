""" This module provides a function to fetch historical price data for a given index using the yfinance library.

"""
import yfinance as yf


def get_index(index, start="2020-01-01", end="2025-01-01"):
    """
    Fetch historical price data for a given index. Allows specifying a date range.
    :param index: a string representing the index ticker (e.g., '^GSPC' for S&P 500)
    :param start: a string representing the start date in 'YYYY-MM-DD' format
    :param end: a string representing the end date in 'YYYY-MM-DD' format
    :return: a pandas DataFrame containing the historical price data for the index
    """
    return yf.download(index, start=start, end=end, auto_adjust=True, progress=False)
