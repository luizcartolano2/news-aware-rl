import os

import pandas as pd

from constants import DATA_PATH

NEWS_DIR = f"{DATA_PATH}/news"
OUTPUT_FILE = f"{DATA_PATH}/consolidated/daily_sentiments.csv"

if __name__ == "__main__":
    # Check if the news directory exists
    if not os.path.exists(NEWS_DIR):
        raise FileNotFoundError(f"News directory {NEWS_DIR} does not exist.")

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    # Read the parsed news CSV file
    df = pd.read_csv(f"{NEWS_DIR}/parsed_news_with_sentiment.csv", parse_dates=["timestamp"])
    # Create a score column based on sentiment
    df["score"] = df["sentiment"].map({
        "POSITIVO": 1,
        "NEUTRO": 0,
        "NEGATIVO": -1
    })

    # Average per day
    daily = df.groupby(df["timestamp"].dt.date)["score"].mean().reset_index()
    daily.columns = ["date", "sentiment_score"]

    # Convert date to datetime
    daily["date"] = pd.to_datetime(daily["date"])

    # Sort by date
    daily = daily.sort_values("date")
    daily.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved daily sentiment with shape {daily.shape}")
