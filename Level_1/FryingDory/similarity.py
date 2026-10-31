
import csv
import math
import os

features = ["tempo_bpm", "duration_sec", "energy", "danceability"]


my_song = [124, 210, 0.78, 0.82]

predicted_song = "Canteen Bass"

print("My prediction:", predicted_song)



def dot_product(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def cosine_similarity(a, b):
    magnitude_a = math.sqrt(dot_product(a, a))
    magnitude_b = math.sqrt(dot_product(b, b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product(a, b) / (magnitude_a * magnitude_b)



def normalize(value, minimum, maximum):
    if maximum == minimum:
        return 0

    return (value - minimum) / (maximum - minimum)


if os.path.exists("datasets/songs.csv"):
    file_path = "datasets/songs.csv"
else:
    file_path = "songs.csv"

songs = []

with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        songs.append({
            "title": row["title"],
            "values": [float(row[f]) for f in features]
        })

minimums = []
maximums = []

for i in range(len(features)):
    values = [song["values"][i] for song in songs]
    minimums.append(min(values))
    maximums.append(max(values))

normalized_my_song = [
    normalize(my_song[i], minimums[i], maximums[i])
    for i in range(len(features))
]

original_results = []
normalized_results = []

for song in songs:
    values = song["values"]

    normalized_values = [
        normalize(values[i], minimums[i], maximums[i])
        for i in range(len(features))
    ]

    score1 = cosine_similarity(my_song, values)
    score2 = cosine_similarity(normalized_my_song, normalized_values)

    original_results.append((song["title"], score1))
    normalized_results.append((song["title"], score2))
original_results.sort(key=lambda x: x[1], reverse=True)
normalized_results.sort(key=lambda x: x[1], reverse=True)

print("\nTop 3 WITHOUT normalization:")

for title, score in original_results[:3]:
    print(title, "-", round(score, 4))


print("\nTop 3 WITH normalization:")

for title, score in normalized_results[:3]:
    print(title, "-", round(score, 4))



print("\nPrediction comparison:")

print("My prediction:", predicted_song)
print("Top song without normalization:", original_results[0][0])
print("Top song with normalization:", normalized_results[0][0])

if predicted_song == original_results[0][0]:
    print("Prediction matches the unnormalized ranking.")
else:
    print("Prediction does not match the unnormalized ranking.")

if predicted_song == normalized_results[0][0]:
    print("Prediction matches the normalized ranking.")
else:
    print("Prediction does not match the normalized ranking.")


top1 = [song[0] for song in original_results[:3]]
top2 = [song[0] for song in normalized_results[:3]]

print("\nDo the top 3 rankings agree?")

if top1 == top2:
    print("Yes, both rankings are the same.")
else:
    print("No, the rankings are different.")
