"""
similarity.py -- Level 1, Hacktoberfest "Similarity and Squashing".

Story: we just loved a song. Which of the 10 songs in datasets/songs.csv
should we recommend next? We answer that with cosine similarity, once on
the raw numbers and once after min-max normalization, and compare the two
rankings.

Pure Python only. `math` is allowed, NumPy is not (per the challenge rules).
"""

import csv
import math
import os

# The song we just loved. Given directly by the challenge spec (not in the CSV).
LOVED_SONG = {
    "tempo_bpm": 124.0,
    "duration_sec": 210.0,
    "energy": 0.78,
    "danceability": 0.82,
}

# The order we always read/compare features in. Keeping one fixed order means
# every vector we build (loved song, each CSV row, normalized or not) lines
# up feature-for-feature.
FEATURES = ["tempo_bpm", "duration_sec", "energy", "danceability"]


def load_songs(csv_path):
    """Read songs.csv into a list of dicts: {"title": str, feature: float, ...}.

    Using csv.DictReader (the csv module, as required) so each row comes back
    keyed by its column header -- no guessing column order.
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = {"title": row["title"]}
            for feature in FEATURES:
                song[feature] = float(row[feature])
            songs.append(song)
    return songs


def to_vector(song):
    """Pull a song dict's numeric features out, in FEATURES order, as a list."""
    return [song[feature] for feature in FEATURES]


def dot_product(a, b):
    """Sum of element-wise products of two equal-length vectors (lists of floats)."""
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def magnitude(v):
    """Euclidean length of a vector: sqrt of the dot product of v with itself."""
    return math.sqrt(dot_product(v, v))


def cosine_similarity(a, b):
    """Cosine of the angle between vectors a and b.

    Cosine similarity cares only about *direction*, not length: it is
    dot(a, b) divided by the product of the two magnitudes. The result is
    always between -1 (opposite direction) and 1 (same direction); 1 means
    "the two songs point the same way in feature space".

    Returns 0.0 if either vector has zero length, so a degenerate row never
    raises a ZeroDivisionError.
    """
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)


def min_max_columns(songs):
    """Work out the (min, max) of each feature across all songs in the list.

    This is computed ONLY from the 10 songs in the file -- the loved song
    does not get a vote in setting the scale, it just gets scaled by the
    result (see min_max_scale_song below). That mirrors how you'd normalize
    a real dataset: fit the scaler on your known data, then apply it to a
    new point.
    """
    bounds = {}
    for feature in FEATURES:
        values = [song[feature] for song in songs]
        bounds[feature] = (min(values), max(values))
    return bounds


def min_max_scale_value(value, min_value, max_value):
    """Scale one number into [0, 1] given the column's min and max.

    If min == max (every song has the same value for that feature), there is
    nothing to scale by, so we return 0.5 -- "right in the middle" -- instead
    of dividing by zero.
    """
    if max_value == min_value:
        return 0.5
    return (value - min_value) / (max_value - min_value)


def min_max_scale_song(song, bounds):
    """Apply min_max_scale_value to every feature of one song dict."""
    scaled = {"title": song.get("title", "")}
    for feature in FEATURES:
        min_value, max_value = bounds[feature]
        scaled[feature] = min_max_scale_value(song[feature], min_value, max_value)
    return scaled


def rank_by_similarity(loved_vector, songs):
    """Return songs sorted by cosine similarity to loved_vector, highest first.

    Each entry in the result is (title, similarity_score).
    """
    scored = []
    for song in songs:
        score = cosine_similarity(loved_vector, to_vector(song))
        scored.append((song["title"], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def print_ranking(label, ranking, top_n=3):
    """Pretty-print the top_n entries of a ranking list."""
    print(f"\n{label}")
    for i, (title, score) in enumerate(ranking[:top_n], start=1):
        print(f"  {i}. {title:<22} cosine similarity = {score:.4f}")


def main():
    # Resolve the CSV path relative to THIS file, not the current working
    # directory, so `python similarity.py` works no matter where you run it
    # from.
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "..", "..", "datasets", "songs.csv")

    songs = load_songs(csv_path)

    # --- Ranking 1: raw features, no normalization ---------------------
    loved_raw_vector = to_vector(LOVED_SONG)
    raw_ranking = rank_by_similarity(loved_raw_vector, songs)

    # --- Ranking 2: min-max normalized features -------------------------
    # Bounds come from the 10 songs only; the loved song is scaled with
    # those same bounds, it does not change them.
    bounds = min_max_columns(songs)
    normalized_songs = [min_max_scale_song(song, bounds) for song in songs]
    loved_normalized = min_max_scale_song(LOVED_SONG, bounds)
    loved_normalized_vector = to_vector(loved_normalized)
    normalized_ranking = rank_by_similarity(loved_normalized_vector, normalized_songs)

    print("Loved song:", LOVED_SONG)
    print("Column bounds (min, max) from the 10 songs:")
    for feature, (lo, hi) in bounds.items():
        print(f"  {feature}: min={lo}, max={hi}")

    print_ranking("Top 3 by RAW cosine similarity (no normalization):", raw_ranking)
    print_ranking("Top 3 by NORMALIZED cosine similarity (min-max):", normalized_ranking)

    raw_top3 = {title for title, _ in raw_ranking[:3]}
    norm_top3 = {title for title, _ in normalized_ranking[:3]}
    agree = raw_top3 == norm_top3
    same_winner = raw_ranking[0][0] == normalized_ranking[0][0]

    print("\nDo the two rankings agree?")
    print(f"  Same #1 pick: {same_winner} "
          f"({raw_ranking[0][0]!r} vs {normalized_ranking[0][0]!r})")
    print(f"  Same top-3 set: {agree}")


if __name__ == "__main__":
    main()
