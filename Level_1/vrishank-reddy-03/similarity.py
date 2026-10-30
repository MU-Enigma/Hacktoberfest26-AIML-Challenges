import csv
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def column_ranges(rows):
    cols = list(zip(*rows))
    return [min(c) for c in cols], [max(c) for c in cols]


def min_max(row, lows, highs):
    return [0.0 if h == l else (x - l) / (h - l)
            for x, l, h in zip(row, lows, highs)]


def load_table(path):
    """Returns (names, rows). Non-numeric cells become the song name."""
    names, rows = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, line in enumerate(reader):
            nums, label = [], []
            for cell in line:
                try:
                    nums.append(float(cell))
                except ValueError:
                    label.append(cell)
            names.append(" ".join(label) or f"row{i}")
            rows.append(nums)
    return names, rows


def rank(query, names, rows):
    scored = [(cosine(query, r), n) for n, r in zip(names, rows)]
    return sorted(scored, reverse=True)


def self_test():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert abs(cosine([3, 4], [4, 3]) - 0.96) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([0, 0], [1, 1]) == 0.0
    assert min_max([5], [0], [10]) == [0.5]
    assert min_max([5], [3], [3]) == [0.0]


if __name__ == "__main__":
    self_test()
    names, rows = load_table("../../datasets/songs.csv")
    loved = [124, 210, 0.78, 0.82]

    print("RAW cosine similarity")
    for score, name in rank(loved, names, rows):
        print(f"  {score:.4f}  {name}")

    # Use the dataset's min and max for BOTH the songs and your loved song,
    # so everything sits on one shared scale.
    lows, highs = column_ranges(rows)
    scaled_rows = [min_max(r, lows, highs) for r in rows]
    scaled_loved = min_max(loved, lows, highs)

    print("\nMIN-MAX NORMALIZED cosine similarity")
    for score, name in rank(scaled_loved, names, scaled_rows):
        print(f"  {score:.4f}  {name}")
