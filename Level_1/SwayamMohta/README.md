# Level 1: Similarity and Squashing

**Author:** Swayam Mohta (`@SwayamMohta`)  
**Level:** Level 1 (Oct 7 Workshop)

Pure Python implementation of core mathematical building blocks for machine learning models using only the Python standard library (`math`, `csv`). **NumPy is not used.**

---

## 1. Quick Start & How to Run

From the repository root or inside this folder:

```bash
# 1. Run similarity recommender (cosine similarity with and without normalization)
python Level_1/SwayamMohta/similarity.py

# 2. Run squashing demonstration (sigmoid, ReLU, tanh, overflow edge case)
python Level_1/SwayamMohta/squashing.py

# 3. Run unit tests
python -m unittest Level_1/SwayamMohta/test_level1.py
```

---

## 2. Song Recommender Results

### Input Song (Loved Song)
- **Tempo:** 124 BPM
- **Duration:** 210 sec
- **Energy:** 0.78
- **Danceability:** 0.82

### Prediction Before Running Any Code
- **Predicted Next Song:** `Fresher Night` or `Canteen Bass`
- **Reasoning:** Intuitively, an upbeat, danceable song (energy 0.78, danceability 0.82) should match another energetic dance track with a similar tempo (~124–132 BPM).

### Results Table

| Rank | Unnormalized (Raw Features) | Cosine Sim | Normalized (Min-Max Scaled to [0, 1]) | Cosine Sim |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **Rooftop Monsoon** | 0.999994 | **Fresher Night** | 0.995150 |
| **2** | **Fresher Night** | 0.999573 | **Canteen Bass** | 0.983837 |
| **3** | **Canteen Bass** | 0.999164 | **Hostel Corridor** | 0.975356 |

- **Top 3 Unnormalized:** `['Rooftop Monsoon', 'Fresher Night', 'Canteen Bass']`
- **Top 3 Normalized:** `['Fresher Night', 'Canteen Bass', 'Hostel Corridor']`
- **Do they agree?** Normalized rankings agree with the intuitive human prediction; unnormalized rankings do **not**.

### Why They Disagree (Scale Dominance)
In unnormalized space, `duration_sec` ($\approx 200$) and `tempo_bpm` ($\approx 120$) have squared values in the tens of thousands ($\approx 40,000$ and $15,000$), while `energy` and `danceability` ($\le 1.0$) contribute less than $0.001\%$ to the vector norm. Unnormalized cosine similarity is almost entirely measuring duration and tempo, recommending *Rooftop Monsoon* purely because 124 BPM / 209s almost identically matches 124 BPM / 210s, despite its low energy (0.3) and danceability (0.3). 

When min-max normalized to $[0, 1]$, each dimension has equal scale, allowing energy and danceability to contribute proportionally, correctly placing *Fresher Night* at #1.

---

## 3. Squashing Functions & Overflow Fix

- **Implemented:**
  - `relu(x)`: $\max(0, x)$
  - `tanh(x)`: Hyperbolic tangent, mapping $\mathbb{R} \to (-1, 1)$
  - `sigmoid(x)`: Numerically stable logistic function, mapping $\mathbb{R} \to (0, 1)$
- **The Failure:** Naive $\sigma(x) = \frac{1}{1 + e^{-x}}$ crashes with `OverflowError: math range error` on inputs like $x = -1000$ because $e^{1000}$ exceeds the maximum floating point number ($\approx 1.79 \times 10^{308}$).
- **The Fix:** For $x < 0$, rewrite algebraically as $\frac{e^x}{1 + e^x}$. When $x = -1000$, $e^{-1000}$ safely underflows to `0.0`, yielding `0.0 / 1.0 = 0.0` without any error.

---

## 4. Notes

See [`NOTES.md`](NOTES.md) for conceptual answers under 200 words explaining raw linear model scores vs. probabilities and when to select sigmoid vs. tanh.
