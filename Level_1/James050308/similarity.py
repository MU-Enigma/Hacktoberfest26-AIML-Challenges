
import csv
import math
from pathlib import Path

# Features used to compare songs
FEATURES = ["tempo_bpm", "duration_sec", "energy", "danceability"]

# Your favourite song's features
MY_SONG = [124, 210, 0.78, 0.82]

# Prediction made before running the calculations
MY_PREDICTION = "Rooftop Monsoon"


def dot_product(a, b):
    """Multiply corresponding elements and add the results."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    magnitude_a = math.sqrt(sum(x ** 2 for x in a))
    magnitude_b = math.sqrt(sum(x ** 2 for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot_product(a, b) / (magnitude_a * magnitude_b)


def min_max_normalize(value, minimum, maximum):
    """Scale a value to the range 0 to 1."""
    if maximum == minimum:
        return 0.0

    return (value - minimum) / (maximum - minimum)


def load_songs():
    """Read the song dataset from the repository."""
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "songs.csv"
    )

    songs = []

    with open(dataset_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            features = [float(row[column]) for column in FEATURES]
            songs.append((row["title"], features))

    return songs


def rank_songs(songs, target, normalize=False):
    """Calculate similarity scores and rank songs."""
    if normalize:
        # Find minimum and maximum for each feature
        minimums = [
            min(song[1][i] for song in songs)
            for i in range(len(FEATURES))
        ]

        maximums = [
            max(song[1][i] for song in songs)
            for i in range(len(FEATURES))
        ]

        # Normalize the favourite song using dataset limits
        target = [
            min_max_normalize(target[i], minimums[i], maximums[i])
            for i in range(len(FEATURES))
        ]

        # Normalize every song using the SAME limits
        processed_songs = []

        for name, features in songs:
            normalized_features = [
                min_max_normalize(features[i], minimums[i], maximums[i])
                for i in range(len(FEATURES))
            ]

            processed_songs.append((name, normalized_features))

        songs = processed_songs

    results = []

    for name, features in songs:
        score = cosine_similarity(target, features)
        results.append((name, score))

    return sorted(results, key=lambda item: item[1], reverse=True)


def main():
    songs = load_songs()

    # Similarity without normalization
    raw_results = rank_songs(songs, MY_SONG)

    # Similarity after min-max normalization
    normalized_results = rank_songs(songs, MY_SONG, normalize=True)

    print("===== SONG SIMILARITY RESULTS =====")
    print("My prediction:", MY_PREDICTION)

    print("\n--- Top 3 WITHOUT Normalization ---")
    for name, score in raw_results[:3]:
        print(f"{name}: {score:.6f}")

    print("\n--- Top 3 WITH Normalization ---")
    for name, score in normalized_results[:3]:
        print(f"{name}: {score:.6f}")

    raw_top3 = [name for name, score in raw_results[:3]]
    normalized_top3 = [name for name, score in normalized_results[:3]]

    print("\n--- Comparison ---")
    print("Top-three rankings agree:", raw_top3 == normalized_top3)
    print("Prediction appears in raw top three:", MY_PREDICTION in raw_top3)
    print("Prediction appears in normalized top three:",
          MY_PREDICTION in normalized_top3)


if __name__ == "__main__":
    main()
