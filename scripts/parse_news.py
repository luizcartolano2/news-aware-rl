""" Script to parse news headlines from text files and save them in a CSV format.

"""
import csv
import os
from datetime import datetime

from constants import DATA_PATH


def parse_line(line_to_parse: str):
    """
    Parses a line from the news text file to extract the headline and timestamp.
    The expected format of the line is:
    "Headline text | DD/MM/YYYY | Other info | HHhMM"
    :param line_to_parse: a string representing a line from the news text file
    :return: a tuple containing the ISO formatted timestamp and the headline text,
            or None if the line cannot be parsed successfully.
    """
    try:
        headline_txt, date_short, _, time = [x.strip() for x in line_to_parse.split("|")]
        dt = datetime.strptime(f"{date_short} {time}", "%d/%m/%Y %Hh%M")
        return dt.isoformat(), headline_txt
    except Exception as e:
        return None


if __name__ == "__main__":
    # Define the path to the news folder
    news_folder = f"{DATA_PATH}/news"

    # check if the news folder exists
    os.makedirs(news_folder, exist_ok=True)

    # Process each text file in the news folder
    parsed_rows = []

    # Iterate through all files in the news folder
    for file_name in os.listdir(news_folder):
        # Process only text files
        if file_name.endswith(".txt"):
            file_path = os.path.join(news_folder, file_name)
            # Open the file and parse each line
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    # Parse the line and append to the list if successful
                    parsed = parse_line(line)
                    if parsed:
                        parsed_rows.append(parsed)

    # Write the parsed data to a CSV file
    with open(f'{news_folder}/parsed_news.csv', 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "headline"])
        for timestamp, headline in parsed_rows:
            writer.writerow([timestamp, headline])
