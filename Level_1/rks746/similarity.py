import math

def dot_product(vector1, vector2):

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    return sum(a * b for a, b in zip(vector1, vector2))

def cosine_similarity(vector1, vector2):

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    dot = dot_product(vector1, vector2)

    norm1 = math.hypot(*vector1)
    norm2 = math.hypot(*vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)

def minmax_normalize(values):
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return [0.0] * len(values)

    return [
        (x - min_val) / (max_val - min_val)
        for x in values
    ]

# normalize each feature column independently.
def normalize_dataset(songs, target):

    num_features = len(target)

    normalized_songs = [
        [0.0] * num_features for _ in songs
    ]

    normalized_target = []

    for j in range(num_features):

        column = [song[j] for song in songs]

        min_val = min(column)
        max_val = max(column)

        if max_val == min_val:
            normalized_column = [0.0] * len(column)
            target_value = 0.0

        else:
            normalized_column = [
                (x - min_val) / (max_val - min_val)
                for x in column
            ]

            target_value = (
                (target[j] - min_val) / (max_val - min_val)
            )

        for i in range(len(songs)):
            normalized_songs[i][j] = normalized_column[i]

        normalized_target.append(target_value)

    return normalized_songs, normalized_target

def rank_by_cosine(song_names, songs, target):

    rankings = []

    for name, song in zip(song_names, songs):

        similarity = cosine_similarity(song, target)

        rankings.append({
            "name": name,
            "similarity": similarity,
        })

    return sorted(
        rankings,
        key=lambda x: x["similarity"],
        reverse=True
    )

def compare_rankings(song_names, songs, target):

    # Raw cosine similarity
    raw_ranking = rank_by_cosine(
        song_names, songs, target
    )

    # min-max normalized cosine similarity
    normalized_songs, normalized_target = normalize_dataset(
        songs, target
    )

    normalized_ranking = rank_by_cosine(
        song_names, normalized_songs, normalized_target
    )

    return {
        "raw_cosine": raw_ranking,
        "normalized_cosine": normalized_ranking
    }

def display_cosine_ranking(title, rankings):

    print(title)

    for rank, result in enumerate(rankings, 1):

        print(
            f"{rank:2}. {result['name']:25} "
            f"Cosine: {result['similarity']:.6f} "
        )
