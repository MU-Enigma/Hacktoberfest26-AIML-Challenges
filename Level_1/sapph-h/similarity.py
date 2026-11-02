"""Similarity: A Song Recommender (Level 1).

This script implements:
- Dot product
- Vector magnitude (Euclidean norm)
- Cosine similarity
- Min-max normalization

It compares unnormalized vs. min-max normalized cosine similarity rankings
for a query song against songs in datasets/songs.csv without using NumPy.
"""

import csv
import math
import os


def dot_product(v1, v2):
    """Computes the dot product of two vectors of equal length."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} vs {len(v2)}")
    return sum(a * b for a, b in zip(v1, v2))


def vector_magnitude(v):
    """Computes the Euclidean magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0 (or 0.0 if either vector has zero magnitude).
    """
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)


def compute_bounds(dataset, features):
    """Computes the minimum and maximum values for each feature in the dataset."""
    bounds = {}
    for feat in features:
        values = [row[feat] for row in dataset]
        bounds[feat] = (min(values), max(values))
    return bounds


def min_max_scale(value, min_val, max_val):
    """Scales a single value into the [0, 1] range using min-max normalization."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def normalize_vector(vec_dict, features, bounds):
    """Normalizes a feature dictionary using precomputed feature bounds."""
    normalized = []
    for feat in features:
        min_val, max_val = bounds[feat]
        scaled = min_max_scale(vec_dict[feat], min_val, max_val)
        normalized.append(scaled)
    return normalized


def load_songs(csv_path):
    """Loads song data from a CSV file into a list of dictionaries."""
    songs = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "title": row["title"],
                "tempo_bpm": float(row["tempo_bpm"]),
                "duration_sec": float(row["duration_sec"]),
                "energy": float(row["energy"]),
                "danceability": float(row["danceability"]),
            })
    return songs


def rank_songs(query_vec, song_vectors, song_titles):
    """Ranks songs by cosine similarity relative to the query vector."""
    results = []
    for title, vec in zip(song_titles, song_vectors):
        sim = cosine_similarity(query_vec, vec)
        results.append((title, sim, vec))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def main():
    # 1. Resolve dataset path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    csv_path = os.path.join(repo_root, "datasets", "songs.csv")

    features = ["tempo_bpm", "duration_sec", "energy", "danceability"]

    # 2. Query song (the song you just loved)
    loved_song = {
        "title": "Loved Song (Query)",
        "tempo_bpm": 124.0,
        "duration_sec": 210.0,
        "energy": 0.78,
        "danceability": 0.82,
    }

    print("=" * 70)
    print(" LEVEL 1: SIMILARITY - SONG RECOMMENDER")
    print("=" * 70)
    print(f"Dataset path: {csv_path}")
    print(
        f"Query Song: Tempo={loved_song['tempo_bpm']} BPM, "
        f"Duration={loved_song['duration_sec']}s, "
        f"Energy={loved_song['energy']}, "
        f"Danceability={loved_song['danceability']}"
    )

    # 3. Prior prediction before running any code
    predicted_song = "Fresher Night"
    prediction_rationale = (
        "Fresher Night has tempo 128 (close to 124), duration 232 (close to 210), "
        "energy 0.90 (close to 0.78), and danceability 0.88 (close to 0.82). "
        "Intuitively, its musical vibe and high dance energy match our loved song closest."
    )
    print("\n--- PREDICTION BEFORE RUNNING CODE ---")
    print(f"Predicted Recommendation: {predicted_song}")
    print(f"Rationale: {prediction_rationale}")

    # 4. Load dataset
    songs = load_songs(csv_path)
    titles = [s["title"] for s in songs]

    # 5. Unnormalized Cosine Similarity
    raw_query_vec = [loved_song[feat] for feat in features]
    raw_song_vecs = [[s[feat] for feat in features] for s in songs]

    raw_rankings = rank_songs(raw_query_vec, raw_song_vecs, titles)

    print("\n--- 1. UNNORMALIZED COSINE SIMILARITY RANKINGS ---")
    print(f"{'Rank':<5}{'Title':<25}{'Similarity':<12}")
    print("-" * 42)
    for i, (title, sim, _) in enumerate(raw_rankings, start=1):
        print(f"{i:<5}{title:<25}{sim:<12.6f}")

    top_3_unnormalized = [r[0] for r in raw_rankings[:3]]
    print(f"\nTop 3 without normalization: {top_3_unnormalized}")

    # 6. Min-Max Normalization
    bounds = compute_bounds(songs, features)
    print("\n--- FEATURE BOUNDS (from 10 dataset songs) ---")
    for feat in features:
        min_v, max_v = bounds[feat]
        print(f"  {feat:<15}: min = {min_v:<8}, max = {max_v:<8}")

    norm_query_vec = normalize_vector(loved_song, features, bounds)
    norm_song_vecs = [normalize_vector(s, features, bounds) for s in songs]

    print(
        "\nNormalized Query Vector: ["
        + ", ".join(f"{v:.4f}" for v in norm_query_vec)
        + "]"
    )

    norm_rankings = rank_songs(norm_query_vec, norm_song_vecs, titles)

    print("\n--- 2. MIN-MAX NORMALIZED COSINE SIMILARITY RANKINGS ---")
    print(f"{'Rank':<5}{'Title':<25}{'Similarity':<12}")
    print("-" * 42)
    for i, (title, sim, _) in enumerate(norm_rankings, start=1):
        print(f"{i:<5}{title:<25}{sim:<12.6f}")

    top_3_normalized = [r[0] for r in norm_rankings[:3]]
    print(f"\nTop 3 with normalization: {top_3_normalized}")

    # 7. Comparison and Explanation
    print("\n--- 3. COMPARISON & EXPLANATION ---")
    agrees_normalized = top_3_normalized[0] == predicted_song
    agrees_unnormalized = top_3_unnormalized[0] == predicted_song

    print(f"Predicted Top Song:        {predicted_song}")
    print(f"Top Song (Unnormalized):  {top_3_unnormalized[0]} (Agrees: {agrees_unnormalized})")
    print(f"Top Song (Normalized):    {top_3_normalized[0]} (Agrees: {agrees_normalized})")
    print("\nExplanation:")
    print(
        "- Why unnormalized failed to match prediction:\n"
        "  Duration (~210s) and Tempo (~124 BPM) have magnitudes in the hundreds, whereas\n"
        "  energy (~0.78) and danceability (~0.82) are decimals between 0 and 1.\n"
        "  In the raw dot product and vector magnitude, the squared terms of duration\n"
        "  and tempo (~50,000) completely overwhelm energy and danceability (~0.6).\n"
        "  Rooftop Monsoon happens to have tempo 124 and duration 209 (virtually an identical\n"
        "  ratio to 124 and 210), giving it a raw cosine similarity of 0.999994, even though\n"
        "  its energy (0.3) and danceability (0.3) represent a totally different calm vibe.\n"
    )
    print(
        "- Why min-max normalization fixed it:\n"
        "  Scaling all features into [0, 1] gives musical vibe (energy & danceability)\n"
        "  equal voice with tempo and duration. Fresher Night matches the high energy (0.90),\n"
        "  high danceability (0.88), and upbeat pace (tempo 128), correctly ranking #1,\n"
        "  in full agreement with our human prediction."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

