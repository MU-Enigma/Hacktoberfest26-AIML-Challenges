#hi Paarth
import pandas as pd

def dot_prod(a, b):
    if len(a) != len(b):
        return 0

    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def cosine_similarity(a, b):
    num = dot_prod(a, b)
    moda = 0
    modb = 0

    for i in range(len(a)):
        moda += a[i] * a[i]
        modb += b[i] * b[i]

    moda = moda**0.5
    modb = modb**0.5
    denom = moda * modb

    return num / denom


def min_max_normalizaSON(a):
    maximum=max(a)
    minimum=min(a)
    denom = maximum - minimum
    a_norm = []

    for i in range(len(a)):
        if denom == 0:
            a_norm.append(0.0)
        else:
            num = (a[i] - minimum) / denom
            a_norm.append(num)

    return a_norm

songs = pd.read_csv("songs.csv")
song_column = "title"
features = ["tempo_bpm", "duration_sec", "energy", "danceability"]


my_song = [
    124,
    210,
    0.78,
    0.82
]


prediction = [
    "Canteen Bass",
    "Rooftop Monsoon"
]

if len(my_song) != len(features):
    raise ValueError("Enter one value for every feature")

song_features = songs[features].apply(pd.to_numeric, errors="raise")

if song_features.isna().any().any():
    raise ValueError("The selected features contain missing values")



songs["cosine_without_normalization"] = song_features.apply(
    lambda row: cosine_similarity(my_song, row.tolist()),
    axis=1,
)


normalized_features = song_features.copy()
my_song_normalized = []

for i in range(len(features)):
    feature = features[i]
    minimum = song_features[feature].min()
    maximum = song_features[feature].max()

    normalized_features[feature] = min_max_normalizaSON(
        song_features[feature].tolist(), minimum, maximum
    )

    my_song_normalized.append(
        min_max_normalizaSON([my_song[i]], minimum, maximum)[0]
    )


songs["cosine_with_normalization"] = normalized_features.apply(
    lambda row: cosine_similarity(my_song_normalized, row.tolist()),
    axis=1,
)


ranking_without = songs[
    [song_column, "cosine_without_normalization"]
].sort_values(
    "cosine_without_normalization", ascending=False, na_position="last"
).reset_index(drop=True)

ranking_with = songs[
    [song_column, "cosine_with_normalization"]
].sort_values(
    "cosine_with_normalization", ascending=False, na_position="last"
).reset_index(drop=True)

ranking_without.index += 1
ranking_with.index += 1

print("Ranking without min-max normalization:")
print(ranking_without.to_string())

print("\nRanking with min-max normalization:")
print(ranking_with.to_string())


if prediction:
    n = len(prediction)

    for label, ranking in [
        ("Without normalization", ranking_without),
        ("With normalization", ranking_with),
    ]:
        actual = ranking[song_column].head(n).tolist()
        print(f"\n{label}:")
        print("Prediction:", prediction)
        print("Computed ranking:", actual)
        print("Matches prediction:", actual == prediction)

        if actual != prediction:
            if label == "Without normalization":
                print(
                    "The ranking uses the angles between the selected feature "
                    "Features with larger numerical scales can dominate "
                    " which may differ from the importance you "
                    "gave each feature in your prediction."
                )
            else:
                print(
                    "Min-max normalization shifts and rescales each feature, "
                    "changing vector directions and cosine similarities. This "
                    "ranking reflects the normalized numeric features, which "
                    "may not capture the preferences behind your prediction."
                )
#bye Paarth