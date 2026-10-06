import csv 
import math 
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "datasets" / "songs.csv"

def load_songs(path):
    titles, rows = [] , [] 
    with open(path,newline = '') as f: # i nmean the csv was pretty normal idk why i wrote newline but yeah 
        reader = csv.reader(f) # split at commas 
        header = next(reader)
        feature_names = header[1:] # removing title as it is not a numerical feature value so well it cannot be a feature 
        for record in reader: # going over reader now that the first row is taken 
            if not record : # checking blank lines 
                continue; 
            titles.append(record[0]) # appends the title into titles list declared 
            values = [] # now we try to store the values of these songs the feature numbers 
            for v in record[1:]:
                values.append(float(v))
            rows.append(values) # rows is a lists of lists that holds the numbers 
    return titles, feature_names, rows 
# now the cosine similarity
# well is there a function for dot product in math ? turns our there isnt one

def dot_product(a,b): # well takes 2 lists 🙏🏻
    if len(a) != len(b):
        raise ValueError("get proper vectors man")
    total = 0.0 
    for i in range(len(a)):
        total = total + a[i] * b[i];

    return total

def cosine_similarity(a,b):
    mag_sq_a = 0; 
    for i in a:
        mag_sq_a += i*i 
    mag_a = math.sqrt(mag_sq_a);
    mag_sq_b = 0; 
    for _ in b:
        mag_sq_b += _*_
    mag_b = math.sqrt(mag_sq_b); # its math.sqrt for fucks sake
    if mag_a == 0 or mag_b == 0:
        raise ValueError("One of the vecs is 0")
    value = dot_product(a,b) / (mag_a * mag_b)
    return value
# so lets normalise 
def min_max_fit(rows):
    mins = []
    maxs = []

    for i in range(len(rows[0])):
        column = []

        for row in rows:
            column.append(row[i])

        mins.append(min(column))
        maxs.append(max(column))

    return mins, maxs

def min_max_transform(row, mins, maxs):
    if len(row) != len(mins): # well we need to make sure we have mins for everythign ig
        raise ValueError("gimme proper things ")

    out = []

    for i in range(len(row)):
        if maxs[i] == mins[i]:
            out.append(0.0)
        else:
            value = (row[i] - mins[i]) / (maxs[i] - mins[i])
            out.append(value)

    return out
# prints the songs after sim scores 
def print_ranking(label, titles, scores):

    # song + score = pair lmao 
    pairs = []
    for i in range(len(titles)):
        pairs.append([titles[i], scores[i]])
    pairs.sort(key=lambda x: x[1], reverse=True)
    print(label) # printing the ehad of the rangkingasfsdfa
    rank = 1
    for pair in pairs:
        print(rank, pair[0], pair[1])
        rank = rank + 1
if __name__ == "__main__":
    titles, names, rows = load_songs(DATA_PATH) 
    my_song = {
        "tempo_bpm": 124,
        "duration_sec": 210,
        "energy": 0.78,
        "danceability": 0.82
    }
    query = []
    for n in names:
        query.append(my_song[n])

    # sweet raw similarity scores below 
    raw = []
    for r in rows:
        raw.append(cosine_similarity(query, r))
    mins, maxs = min_max_fit(rows)
    rows_n = []
    for r in rows:
        rows_n.append(min_max_transform(r, mins, maxs))
    query_n = min_max_transform(query, mins, maxs)

    normalized = []
    for r in rows_n:
        normalized.append(cosine_similarity(query_n, r))
#phewww 
    print_ranking("Raw cosine similarity", titles, raw)
    print_ranking("Min-max normalized cosine similarity", titles, normalized)
