import csv
from pathlib import Path

from similarity import (
    compare_rankings,
    display_cosine_ranking,
)

from squashing import (
    test_squashing_functions,
    test_overflow
)

CSV_PATH = Path(__file__).with_name("songs.csv")

SONG_COLUMN = "title"

FEATURES = [
    "tempo_bpm",
    "duration_sec",
    "energy",
    "danceability"
]

DEFAULT_SONG = [124, 210, 0.78, 0.82] # given song parameters

def load_songs():

    song_names = []
    songs = []

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        # Validate CSV headers
        required_columns = [SONG_COLUMN] + FEATURES

        missing_columns = [
            column for column in required_columns
            if column not in (reader.fieldnames or [])
        ]

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {missing_columns}"
            )

        for row_number, row in enumerate(reader, start=2):

            try:
                name = row[SONG_COLUMN].strip()

                if not name:
                    raise ValueError("Song title is missing")

                vector = [
                    float(row[feature])
                    for feature in FEATURES
                ]

                song_names.append(name)
                songs.append(vector)

            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid data in CSV row {row_number}: {error}"
                ) from error

    if not songs:
        raise ValueError("Dataset is empty!")

    return song_names, songs

def run_similarity_experiments():

    song_names, songs = load_songs()
    target = DEFAULT_SONG
    print("\nInput Song Vector:", target)

    rankings = compare_rankings(
        song_names,
        songs,
        target
    )

    # Raw cosine similarity
    display_cosine_ranking(
        "\nRAW COSINE SIMILARITY",
        rankings["raw_cosine"]
    )

    # Min-max normalized cosine similarity
    display_cosine_ranking(
        "\nMIN-MAX NORMALIZED COSINE SIMILARITY",
        rankings["normalized_cosine"]
    )

    print("\nTop recommendations")

    print(
        "Raw cosine:",
        rankings["raw_cosine"][0]["name"]
    )

    print(
        "Normalized cosine:",
        rankings["normalized_cosine"][0]["name"]
    )

def main():

    print("Level 1: Similarity and Squashing")

    # Similarity experiments
    run_similarity_experiments()

    # Squashing function experiments
    test_overflow()
    test_squashing_functions()

if __name__ == "__main__":
    main()
