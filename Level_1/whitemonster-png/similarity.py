import csv
import math
import os

features = ["tempo_bpm", "duration_sec", "energy", "danceability"]

my_song = [124, 210, 0.78, 0.82]
predicted_song = "Hostel Corridor"

print("My predicted song:", predicted_song)

def dot_product(a, b):
    result = 0

    for x, y in zip(a, b):
        result += x * y

    return result

def cosine_similarity(a, b):
    dot = dot_product(a, b)

    length_a = math.sqrt(dot_product(a, a))
    length_b = math.sqrt(dot_product(b, b))

    if length_a == 0 or length_b == 0:
        return 0

    return dot / (length_a * length_b)

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
        song_values = [float(row[feature]) for feature in features]

        songs.append({
            "title": row["title"],
            "values": song_values
        })

minimums = []
maximums = []

for i in range(len(features)):
    feature_values = [song["values"][i] for song in songs]

    minimums.append(min(feature_values))
    maximums.append(max(feature_values))

normalized_my_song = []

for i in range(len(features)):
    value = normalize(my_song[i], minimums[i], maximums[i])
    normalized_my_song.append(value)

results_original = []
results_normalized = []

for song in songs:
    values = song["values"]
    scaled_values = []

    for i in range(len(features)):
        scaled = normalize(values[i], minimums[i], maximums[i])
        scaled_values.append(scaled)

    score_original = cosine_similarity(my_song, values)
    score_normalized = cosine_similarity(
        normalized_my_song, scaled_values
    )

    results_original.append((song["title"], score_original))
    results_normalized.append((song["title"], score_normalized))


results_original.sort(key=lambda song: song[1], reverse=True)
results_normalized.sort(key=lambda song: song[1], reverse=True)

print("\nTop 3 without normalization:")

for title, score in results_original[:3]:
    print(title, "-", round(score, 4))

print("\nTop 3 with normalization:")

for title, score in results_normalized[:3]:
    print(title, "-", round(score, 4))

print("\nPrediction check:")
print("My prediction:", predicted_song)
print("Best match without normalization:", results_original[0][0])
print("Best match with normalization:", results_normalized[0][0])

if predicted_song == results_original[0][0]:
    print("My prediction matches the result without normalization.")
else:
    print("My prediction does not match the result without normalization.")


if predicted_song == results_normalized[0][0]:
    print("My prediction matches the result with normalization.")
else:
    print("My prediction does not match the result with normalization.")

top_original = [song[0] for song in results_original[:3]]
top_normalized = [song[0] for song in results_normalized[:3]]

if top_original == top_normalized:
    print("\nThe top 3 rankings agree.")
else:
    print("\nThe top 3 rankings do not agree.")