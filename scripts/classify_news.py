""" Script to classify news headlines using a sentiment classifier.
This script reads a CSV file containing news headlines, classifies each headline's sentiment using
a pre-trained Llama model, and saves the results back to a new CSV file.

"""
import pandas as pd
from tqdm import tqdm
from constants import DATA_PATH
from llm.llama_sentiment_classifier import LlamaSentimentClassifier

if __name__ == "__main__":
    # Load the news data
    news_data = pd.read_csv(f"{DATA_PATH}/news/parsed_news.csv", parse_dates=["timestamp"])
    news_data["sentiment"] = None

    # Initialize the sentiment classifier
    classifier = LlamaSentimentClassifier()

    # Process each headline and classify sentiment
    for idx, row in tqdm(news_data.iterrows(), total=len(news_data)):
        sentiment = classifier.classify(row["headline"])
        news_data.at[idx, "sentiment"] = sentiment

        # Save progress every 100 rows to avoid losing work
        if idx % 100 == 0:
            news_data.to_csv(f"{DATA_PATH}/news/parsed_news_with_sentiment_temp.csv", index=False)

    # Final save after processing all rows
    news_data.to_csv(f"{DATA_PATH}/news/parsed_news_with_sentiment.csv", index=False)
