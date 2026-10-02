# Level 1: Similarity and Squashing

Put your work in **`Level_1/<your-github-username>/`**. Full task description is in the [main README](../README.md#level-1-similarity-and-squashing-oct-7-workshop).

## What to submit

- Your code (any file names you like), pure Python: the `math` library is allowed, **NumPy is not**
  - dot product, cosine similarity, min-max normalization
  - sigmoid, ReLU, tanh
- Your results: the song you predicted before running anything, the top 3 songs with and without normalization, and one line on whether they agree
- `NOTES.md` (under 200 words, written for a friend who hasn't taken maths)

## Data

`datasets/songs.csv`: 10 made-up songs. The song you just loved: tempo 124, duration 210, energy 0.78, danceability 0.82.

When you min-max normalize, work out the min and max from the 10 songs in the file, and scale your song using those same numbers.

## Example layout

```
Level_1/
└── your-github-username/
    ├── similarity.py
    ├── squashing.py
    └── NOTES.md
```

Fill in the PR template when you open your pull request, and see [CONTRIBUTING.md](../CONTRIBUTING.md) for the Git steps.