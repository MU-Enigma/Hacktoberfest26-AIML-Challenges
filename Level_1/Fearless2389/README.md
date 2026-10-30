# Level 1: Similarity and Squashing — Fearless2389

## Prediction (made before running code)

_Drafted with an AI assistant before any code was run; see the AI disclosure in the PR._

I looked only at `datasets/songs.csv` and the loved song (tempo 124, duration 210,
energy 0.78, danceability 0.82) before writing or running any similarity code.

**My pick: `Fresher Night`** (tempo 128, duration 232, energy 0.90, danceability 0.88).

Reasoning: when I think about what makes two songs feel alike, I weigh energy and
danceability the most — they describe the *vibe* of the song, not just its length or
speed. `Fresher Night` is the closest match on both of those (0.90 vs 0.78 energy,
0.88 vs 0.82 danceability), and its tempo/duration are also in a similar ballpark
(128 vs 124, 232 vs 210). `Canteen Bass` (0.82 energy, 0.85 danceability) is a close
second by the same logic.

<!-- Results section below is filled in AFTER running the code. The prediction above
     is never edited once results are known. -->

## How to run

From inside this folder (or from anywhere -- the CSV path is resolved relative to
the script file, not the current directory):

```bash
python similarity.py
python squashing.py
python -m pytest -q
```

No extra setup needed: the only import outside the standard library is `pytest`,
used for the tests, and `squashing.py`/`similarity.py` only use `csv`, `math` and
`os` from the standard library (NumPy is intentionally not used).

## Results

**Loved song:** tempo 124, duration 210, energy 0.78, danceability 0.82.

**Top 3 -- raw cosine similarity (no normalization):**

| # | Song | cosine similarity |
|---|------|---|
| 1 | Rooftop Monsoon | 1.0000 |
| 2 | Fresher Night | 0.9996 |
| 3 | Canteen Bass | 0.9992 |

**Top 3 -- normalized cosine similarity (min-max, bounds from the 10 songs):**

| # | Song | cosine similarity |
|---|------|---|
| 1 | Fresher Night | 0.9952 |
| 2 | Canteen Bass | 0.9838 |
| 3 | Hostel Corridor | 0.9754 |

**Do the two rankings agree?** No. The #1 pick changes (Rooftop Monsoon raw vs.
Fresher Night normalized), and the top-3 sets only partially overlap (Canteen Bass
is in both, but Rooftop Monsoon drops out and Hostel Corridor appears after
normalizing).

### Why they disagree

Checked against the actual numbers, not just assumed: `tempo_bpm` (78-170) and
`duration_sec` (176-300) are on a much bigger scale than `energy` and
`danceability` (both 0-1). In the raw dot product, a term like `124 * 209`
(tempo) is on the order of 10,000-30,000, while a term like `0.78 * 0.3`
(energy) is under 1. The big-scale features completely dominate the raw
cosine similarity, so it effectively ranks songs by how closely their
tempo/duration *ratio* matches the loved song's, almost ignoring energy and
danceability.

That's exactly what happens here: `Rooftop Monsoon` (tempo 124, duration 209)
has an almost perfect tempo/duration match to the loved song (124, 210) --
but its energy and danceability (0.3, 0.3) are nowhere close to the loved
song's (0.78, 0.82). Raw cosine still ranks it #1 purely on the tempo/duration
match.

Once every feature is min-max scaled into [0, 1], no single feature can
dominate by sheer magnitude anymore, and the ranking shifts to songs that are
actually close on *all four* features. `Fresher Night` (0.90 energy, 0.88
danceability, tempo 128, duration 232) becomes the top pick -- genuinely close
on every dimension, not just the two with the biggest raw numbers.

### Prediction vs. actual

My prediction (made before running any code, see above) was **Fresher Night**,
because I reasoned about energy/danceability as the "vibe" of a song. That
matches the **normalized** ranking's #1 pick exactly, but not the raw ranking's
#1 pick (Rooftop Monsoon). The takeaway: reasoning about the "vibe" of a song
was implicitly already normalizing -- treating energy and danceability as just as important
as tempo and duration -- which is exactly what min-max scaling does
mathematically, and exactly what raw cosine similarity fails to do.

## Files in this folder

- `similarity.py` -- dot product, cosine similarity, min-max normalization, and the
  song ranking/printing logic.
- `squashing.py` -- sigmoid (naive and numerically stable), ReLU, tanh, plus a demo
  of the naive sigmoid overflowing and the stable version handling the same input.
- `NOTES.md` -- the under-200-word explanation for a non-maths friend.
- `test_level1.py` -- pytest coverage for all of the above.
- `README.md` -- this file.
