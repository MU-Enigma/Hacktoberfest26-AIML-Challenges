"""Level 1, Task 1: a tiny song recommender built from scratch.

Pure Python only (math is allowed, NumPy is not).

Run from the repo root:
    python Level_1/<your-github-username>/similarity.py
or point it at any CSV:
    python similarity.py path/to/songs.csv
"""

import csv
import math
import os
import sys

# The song you just loved: tempo (BPM), duration (s), energy, danceability
MY_SONG = [124, 210, 0.78, 0.82]

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "songs.csv"
)


# ---------- the three building blocks ----------

def dot_product(a, b):
    """Multiply matching entries and add them up."""
    if len(a) != len(b):
        raise ValueError(f"Vectors have different lengths: {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def magnitude(v):
    """Length of a vector: sqrt(v . v)."""
    return math.sqrt(dot_product(v, v))


def cosine_similarity(a, b):
    """cos(angle) between two vectors = (a . b) / (|a| * |b|).

    Returns 0.0 if either vector is all zeros (the angle is undefined there).
    """
    mag_a, mag_b = magnitude(a), magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)


def column_min_max(rows):
    """Min and max of every column across a list of rows."""
    columns = list(zip(*rows))
    return [min(c) for c in columns], [max(c) for c in columns]


def min_max_scale(row, mins, maxs):
    """Scale each value to (value - min) / (max - min).

    If a column never changes (max == min) it carries no information, so it
    becomes 0. Values outside the file's range can land below 0 or above 1,
    which is fine and expected for a song that isn't in the file.
    """
    scaled = []
    for value, lo, hi in zip(row, mins, maxs):
        scaled.append(0.0 if hi == lo else (value - lo) / (hi - lo))
    return scaled


def min_max_normalize(rows):
    """Min-max normalize a whole table, column by column."""
    mins, maxs = column_min_max(rows)
    return [min_max_scale(r, mins, maxs) for r in rows]


# ---------- loading data ----------

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_songs(path):
    """Return (names, numeric_column_names, rows_of_floats).

    Numeric columns are detected automatically, so this works on any table
    with 2+ numeric columns. The first non-numeric column (if any) is used
    as the name; otherwise songs are named by row number.
    """
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError(f"{path} has no data rows")

    headers = list(records[0].keys())
    numeric = [h for h in headers if all(_is_number(r[h]) for r in records)]
    text = [h for h in headers if h not in numeric]

    names = [r[text[0]] for r in records] if text else [f"row {i + 1}" for i in range(len(records))]
    rows = [[float(r[h]) for h in numeric] for r in records]
    return names, numeric, rows


# ---------- ranking ----------

def rank(query, rows, names):
    """All songs sorted by cosine similarity to the query, best first."""
    scored = [(cosine_similarity(query, row), name) for row, name in zip(rows, names)]
    return sorted(scored, reverse=True)


def print_ranking(title, ranking):
    print(f"\n{title}")
    for i, (score, name) in enumerate(ranking, start=1):
        print(f"  {i:2}. {name:<25} {score:.6f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    names, columns, rows = load_songs(path)
    print(f"Loaded {len(rows)} songs with columns: {', '.join(columns)}")

    if len(MY_SONG) != len(columns):
        sys.exit(f"MY_SONG has {len(MY_SONG)} values but the file has "
                 f"{len(columns)} numeric columns. Edit MY_SONG to match.")

    raw = rank(MY_SONG, rows, names)
    print_ranking("Ranking WITHOUT normalization:", raw)

    # Min/max come from the 10 songs in the file; my song uses the same numbers.
    mins, maxs = column_min_max(rows)
    norm_rows = [min_max_scale(r, mins, maxs) for r in rows]
    norm_query = min_max_scale(MY_SONG, mins, maxs)
    print(f"\nMy song after scaling: {[round(v, 3) for v in norm_query]}")
    norm = rank(norm_query, norm_rows, names)
    print_ranking("Ranking WITH min-max normalization:", norm)

    top_raw = [n for _, n in raw[:3]]
    top_norm = [n for _, n in norm[:3]]
    print("\nTop 3 without normalization:", top_raw)
    print("Top 3 with normalization:   ", top_norm)
    print("Agree?", "yes" if top_raw == top_norm else "no")


if __name__ == "__main__":
    main()
