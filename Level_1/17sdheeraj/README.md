# level 1: similarity and squashing 

mah solution for lvl 1 of hacktoberfest aiml. only using the standard library (`math`, `csv`, `os`)

## filesss

- `similarity.py`: dot product, vector magnitude, cosine similarity, and min-max normalization. loads `songs.csv` and ranks songs against my loved song.
- `squashing.py`: sigmoid, relu, and tanh functions, including the demonstration and fix for the sigmoid overflow bug on extreme negative numbers.
- `NOTES.md`: short conceptual explanation written for a friend explaining probabilities, scores like 47, and when to pick sigmoid vs tanh.
- `datasets/songs.csv`: the dataset of 10 songs with tempo, duration, energy, and danceability.

## how to run

from this directory:

```bash
python3 similarity.py
python3 squashing.py
```

## song recommender

### my target song
the song i just loved:
- tempo: 124 bpm
- duration: 210 sec
- energy: 0.78
- danceability: 0.82

### my prediction (made before running any code)
**predicted song:** Fresher Night (runner up: Canteen Bass)

**reasoning:** looking at the numbers in `songs.csv`, the song i loved is clearly an upbeat, high-energy dance track. Fresher Night has tempo 128, energy 0.90, and danceability 0.88, while Canteen Bass has tempo 132, energy 0.82, and danceability 0.85. Both closely capture the same high-energy party vibe.