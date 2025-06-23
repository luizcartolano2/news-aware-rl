""" This script extracts test metrics from text files in a specified directory,
classifies them by dataset and metric types, and saves the results in CSV format.

"""
import os
import re
import pandas as pd

from constants import TEXT_PATH


def extract_test_metrics_from_file(filepath):
    """
    Extracts test metrics from a given text file.
    :param filepath: absolute path to the text file
    :return: a dictionary of metrics or None if no metrics found
    """
    with open(filepath, 'r') as f:
        content = f.read()

    test_match = re.search(r"=== TEST METRICS ===(.*?)={60}", content, re.DOTALL)
    if not test_match:
        return None

    test_block = test_match.group(1)
    metrics = {}

    perf_match = re.search(r"PERFORMANCE METRICS:(.*?)(?:FINAL ALLOCATIONS:|$)", test_block, re.DOTALL)
    if perf_match:
        for line in perf_match.group(1).strip().splitlines():
            key, val = line.strip().split(":")
            key = key.strip().lower().replace(" ", "_")
            val = float(val.strip('% \t'))
            metrics[key] = val / 100 if "%" in line else val

    alloc_match = re.search(r"FINAL ALLOCATIONS:(.*)", test_block, re.DOTALL)
    if alloc_match:
        for line in alloc_match.group(1).strip().splitlines():
            asset, pct = line.strip().split(":")
            key = f"allocation_{asset.strip().lower()}"
            val = float(pct.strip('% \t'))
            metrics[key] = val

    return metrics


def collect_all_test_metrics(directory):
    """
    Collects test metrics from all text files in the specified directory.
    :param directory: absolute path to the directory containing text files
    :return: a pandas DataFrame with filenames as columns and metrics as rows
    """
    results = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(directory, filename)
        metrics = extract_test_metrics_from_file(filepath)
        if metrics:
            results[filename] = metrics
        else:
            print(f"Warning: No test metrics found in {filename}")

    return pd.DataFrame(results)


def classify_files(filenames):
    """
    Classifies filenames into dataset types and metric types based on their naming conventions.
    :param filenames: a list of filenames to classify
    :return: a tuple of two dictionaries:
    """
    dataset_types = {
        "prices": [],
        "prices+macros": [],
        "all": []
    }
    metric_types = {
        "log_return": [],
        "sharpe": [],
        "sortino": [],
        "volatility_penalty": []
    }

    for fname in filenames:
        if "prices+macros" in fname:
            dataset_types["prices+macros"].append(fname)
        elif "prices" in fname:
            dataset_types["prices"].append(fname)
        elif "all" in fname:
            dataset_types["all"].append(fname)

        for m in metric_types:
            if m in fname:
                metric_types[m].append(fname)

    return dataset_types, metric_types


def save_grouped_dataframes(df, dataset_types, metric_types):
    """
    Saves the DataFrame grouped by dataset and metric types into separate CSV files.
    :param df: a pandas DataFrame containing the test metrics
    :param dataset_types: a dictionary mapping dataset types to lists of filenames
    :param metric_types: a dictionary mapping metric types to lists of filenames
    :return: a void function that saves files to disk
    """
    os.makedirs(TEXT_PATH, exist_ok=True)

    # Save by dataset type
    for group, files in dataset_types.items():
        if files:
            df_subset = df[files]
            df_subset.to_csv(os.path.join(TEXT_PATH, f"{group}_datasets.csv"))

    # Save by metric type
    for group, files in metric_types.items():
        if files:
            df_subset = df[files]
            df_subset.to_csv(os.path.join(TEXT_PATH, f"{group}_metrics.csv"))


if __name__ == "__main__":
    df = collect_all_test_metrics(TEXT_PATH)
    df.to_csv(f"{TEXT_PATH}/test_metrics_summary.csv")
    print("Saved full table to test_metrics_summary.csv")

    dataset_types, metric_types = classify_files(df.columns)
    save_grouped_dataframes(df, dataset_types, metric_types)

    print("Saved grouped tables to 'grouped_metrics/'")
