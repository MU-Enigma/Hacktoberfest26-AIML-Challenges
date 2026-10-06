import math
import csv
from typing import List

def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(v1, v2):
    dot = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    return dot/(mag1*mag2)

def get_min_max(data):
    # Transpose data to easily get min/max of each column
    cols = list(zip(*data))
    return [min(c) for c in cols], [max(c) for c in cols]

def min_max_normalize(vector, mins, maxs):
    return [(v - c_min) / (c_max - c_min) if c_max != c_min else 0.0 
            for v, c_min, c_max in zip(vector, mins, maxs)]

songs = []

with open("../../datasets/songs.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        features = [float(row['tempo_bpm']), float(row['duration_sec']), 
                    float(row['energy']), float(row['danceability'])]
        songs.append((row['title'], features))

target = [124, 120, 0.78, 0.82]

unnormalized = []
normalized = []

# FIXED: Added [target] to the pool so the global minimum for duration becomes 120
all_features = [s[1] for s in songs] + [target]
mins, maxs = get_min_max(all_features)

normalize_tgt = min_max_normalize(target, mins, maxs)

for title, features in songs:
    # 1. Unnormalized Calculation
    score_unnorm = cosine_similarity(target, features)
    unnormalized.append((score_unnorm, title))
    
    # 2. Normalized Calculation
    norm_features = min_max_normalize(features, mins, maxs)
    score_norm = cosine_similarity(normalize_tgt, norm_features)
    normalized.append((score_norm, title))

# Sort from highest similarity to lowest
unnormalized.sort(reverse=True)
normalized.sort(reverse=True)

print("=============== Unnormalized ===================")
print("Top 3:", unnormalized[:3])
print("=============== Normalized ===================")
print("Top 3:", normalized[:3])


# Squashing
def sigmoid(x: float) -> float:

    # Very large negative values can break the function as it'll overflow and will cause `OverflowError`.
    # The fix is to multiply by e^x, which would make the formula as (e^x) / (1 + e^x)
    if x < 0:
        return math.exp(x) / (1 + math.exp(x))
    return 1/(1+math.exp(-x))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0, x)
