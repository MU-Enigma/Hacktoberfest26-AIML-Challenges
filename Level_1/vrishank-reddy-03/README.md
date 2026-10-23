# Level 1: Similarity and Squashing

Pure Python (only the `math` and `csv` libraries, no NumPy).

## How to run

```
cd Level_1/vrishank-reddy-03
python3 similarity.py
python3 squashing.py
```

`similarity.py` runs a few self-tests, then prints both rankings. `squashing.py` prints a small table and demonstrates the sigmoid bug. Full outputs are in `output_similarity.txt` and `output_squashing.txt`.

## Example run (top 5 of each ranking)

```
RAW cosine similarity
  1.0000  Rooftop Monsoon
  0.9996  Fresher Night
  0.9992  Canteen Bass
  0.9952  Hostel Corridor
  0.9934  Midnight Metro

MIN-MAX NORMALIZED cosine similarity
  0.9952  Fresher Night
  0.9838  Canteen Bass
  0.9754  Hostel Corridor
  0.9158  Bus Pass Anthem
  0.8856  Sprint to Class
```

## My prediction vs the results

My prediction (in `predictions.txt`, committed before I ran any code): **Fresher Night**, runner-up **Canteen Bass**.

- **Raw ranking:** Fresher Night came 2nd (0.9996), behind Rooftop Monsoon (1.0000). Canteen Bass was 3rd.
- **Normalized ranking:** Fresher Night came 1st (0.9952) and Canteen Bass 2nd (0.9838). Both of my picks match.

**Did my reasoning match?** Only partly. I wrote that I weighted tempo and duration most. By that rule I should have picked Rooftop Monsoon, which is almost identical to the loved song on both (124 BPM and 209 s, against 124 BPM and 210 s). So my pick turned out right in the normalized ranking, but the reason I gave for it does not hold up. I would not have picked Fresher Night by following my own stated logic.

**Why the raw ranking disagrees.** Tempo and duration are in the hundreds, while energy and danceability are below 1. In the dot product the large numbers dominate, so energy and danceability barely register. Rooftop Monsoon is far from the loved song on energy (0.30 vs 0.78) and danceability (0.30 vs 0.82), yet the raw score ignores that and gives it a perfect 1.0000. The raw scores are also squashed together: nine of the ten sit between 0.97 and 1.00.

After min-max normalization every column is scaled to 0 to 1, so all four count equally. Rooftop Monsoon then falls from 1st to 9th (0.6424).

**Which I would trust.** The normalized ranking. The raw one mostly measures the tempo to duration ratio, which is an accident of the units, not a sign that songs sound alike.

Note: the loved song is scaled with the dataset's own min and max, so it lives on the same scale as the songs.

## Sigmoid bug

- **Input that broke it:** `-1000`
- **What happened:** `sigmoid_naive(-1000)` raised `math range error`.
- **Why:** it computes `exp(1000)`, which is bigger than the largest number a float can hold, so Python raises an overflow error.
- **Fix:** for negative `x`, use `e / (1 + e)` with `e = exp(x)`. This only takes the exponent of a number that is zero or negative, so it cannot overflow. The two formulas are equal, they differ only in which is safe for each sign.
- **Result:** the fixed version returns `0.0` at -1000. That value is exactly zero because the true answer is too small for a float. This is harmless for sigmoid, but a later log-loss would need clipping because `log(0)` is undefined.

## Data

Used `datasets/songs.csv` as provided.