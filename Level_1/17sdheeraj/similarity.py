import math
import csv
import os

def dotproduct(veca, vecb):
    tot = 0.0
    for i in range(len(veca)):
        tot += veca[i] * vecb[i]
    return tot

def magnitude(vec):
    return math.sqrt(dotproduct(vec, vec))

def cossim(veca, vecb):
    maga = magnitude(veca)
    magb = magnitude(vecb)
    if maga == 0.0 or magb == 0.0:
        return 0.0
    return dotproduct(veca, vecb) / (maga * magb)

def minmaxnorm(rows):
    numcols = len(rows[0])
    mins = []
    maxs = []
    for colidx in range(numcols):
        colvals = [row[colidx] for row in rows]
        mins.append(min(colvals))
        maxs.append(max(colvals))
    normrows = []
    for row in rows:
        normrow = []
        for colidx in range(numcols):
            colmin = mins[colidx]
            colmax = maxs[colidx]
            if colmax == colmin:
                normrow.append(0.0)
            else:
                normrow.append((row[colidx] - colmin) / (colmax - colmin))
        normrows.append(normrow)
    return normrows, mins, maxs

def scalepoint(point, mins, maxs):
    normpoint = []
    for colidx in range(len(point)):
        colmin = mins[colidx]
        colmax = maxs[colidx]
        if colmax == colmin:
            normpoint.append(0.0)
        else:
            normpoint.append((point[colidx] - colmin) / (colmax - colmin))
    return normpoint

def loadsongs(filepath):
    titles = []
    rows = []
    with open(filepath, "r") as filehandle:
        reader = csv.reader(filehandle)
        header = next(reader)
        for row in reader:
            titles.append(row[0])
            rows.append([float(val) for val in row[1:]])
    return titles, rows

def ranksongs(lovedvec, titles, rows):
    scores = []
    for i in range(len(rows)):
        sim = cossim(lovedvec, rows[i])
        scores.append((sim, titles[i]))
    scores.sort(reverse=True)
    return scores

def run():
    paths = [
        "datasets/songs.csv",
        "songs.csv",
        "../datasets/songs.csv",
        "../../Hacktoberfest26-AIML-Challenges/datasets/songs.csv",
        "/home/sdheeraj/Documents/GitHub/Hacktoberfest26-AIML-Challenges/datasets/songs.csv"
    ]
    filepath = ""
    for p in paths:
        if os.path.exists(p):
            filepath = p
            break
    if not filepath:
        print("could not find songs dataset")
        return

    titles, rows = loadsongs(filepath)
    targetsong = [124.0, 210.0, 0.78, 0.82]

    rawscores = ranksongs(targetsong, titles, rows)
    print("=== raw cosine similarity ===")
    for score, name in rawscores:
        print(f"{name}: {score:.5f}")

    normrows, mins, maxs = minmaxnorm(rows)
    normtarget = scalepoint(targetsong, mins, maxs)

    normscores = ranksongs(normtarget, titles, normrows)
    print("\n=== min max normalized cosine similarity ===")
    for score, name in normscores:
        print(f"{name}: {score:.5f}")

    print("\n=== top 3 comparison ===")
    print("top 3 raw:")
    for rankidx in range(3):
        print(f"{rankidx + 1}. {rawscores[rankidx][1]} ({rawscores[rankidx][0]:.5f})")

    print("\ntop 3 normalized:")
    for rankidx in range(3):
        print(f"{rankidx + 1}. {normscores[rankidx][1]} ({normscores[rankidx][0]:.5f})")

run()
