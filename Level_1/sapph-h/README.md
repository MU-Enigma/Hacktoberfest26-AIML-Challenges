# Level 1: Similarity and Squashing

**Author:** sapph-h  
**Track:** Pure Python (standard library `math` only, **no NumPy**)

---

## Folder Structure

```
Level_1/sapph-h/
├── similarity.py     # Pure-Python dot product, cosine similarity, min-max normalization, song recommender
├── squashing.py      # Numerically stable sigmoid, ReLU, tanh, and overflow demonstration
├── NOTES.md          # Conceptual notes under 200 words for a non-math friend
├── test_level1.py    # Unit tests for all similarity and squashing functions
└── README.md         # Documentation, how to run, and full results summary
```

---

## How to Run

From the repository root or from inside `Level_1/sapph-h/`:

```bash
# 1. Run the song similarity recommender
python3 Level_1/sapph-h/similarity.py

# 2. Run the squashing functions demonstration (break & fix)
python3 Level_1/sapph-h/squashing.py

# 3. Run the unit test suite
python3 Level_1/sapph-h/test_level1.py
```

---

## Results: Song Recommender

### Target Song (Loved Track)
- **Tempo:** 124 BPM
- **Duration:** 210 seconds
- **Energy:** 0.78
- **Danceability:** 0.82

### Prediction Before Running Code
- **Predicted recommendation:** `Fresher Night`
- **Reasoning:** We loved an upbeat, high-energy, danceable song. Looking at `datasets/songs.csv`, `Fresher Night` (tempo 128, duration 232, energy 0.90, danceability 0.88) intuitively matches the high energy and danceability characteristics closest to our loved song.

### Rankings

| Metric | Top 1 | Top 2 | Top 3 |
|---|---|---|---|
| **Without Normalization** | Rooftop Monsoon (0.999994) | Fresher Night (0.999573) | Canteen Bass (0.999164) |
| **With Min-Max Normalization** | Fresher Night (0.995150) | Canteen Bass (0.983837) | Hostel Corridor (0.975356) |

### Agreement with Prediction
- **Agreement:** Normalized cosine similarity agrees with the prediction (`Fresher Night` is #1), whereas unnormalized cosine similarity disagrees (`Rooftop Monsoon` is #1).
- **Why they disagree:**
  - In raw form, `duration_sec` (~200) and `tempo_bpm` (~120) have magnitudes hundreds of times larger than `energy` (0.0–1.0) and `danceability` (0.0–1.0).
  - Dot products and magnitudes are dominated by the large numbers ($\approx 50,000$ vs $\approx 0.6$).
  - `Rooftop Monsoon` has tempo 124 and duration 209 (virtually an identical 124:210 ratio), so raw cosine similarity ranks it #1 despite it having completely opposite low energy (0.3) and low danceability (0.3).
  - Min-max scaling places all features onto an equal $[0, 1]$ scale, allowing the musical vibe (energy and danceability) to carry proportional weight.

---

## Squashing: Break & Fix

- **Breaking Input:** $x = -1000$ in naive $\sigma(x) = \frac{1}{1 + e^{-x}}$.
- **Error:** `OverflowError: math range error`.
- **Why it broke:** When $x = -1000$, $-x = 1000$. Python attempts to compute $e^{1000}$. Standard 64-bit IEEE 754 floating point numbers cap at $\approx 1.79 \times 10^{308}$ ($e^{709.78}$). $e^{1000}$ exceeds this limit and crashes with overflow.
- **The Fix:** Use the mathematically equivalent piecewise form:
  $$\sigma(x) = \begin{cases} \frac{1}{1 + e^{-x}}, & x \ge 0 \\ \frac{e^x}{1 + e^x}, & x < 0 \end{cases}$$
  For $x < 0$, $e^x$ is between $0$ and $1$. For $x = -1000$, $e^{-1000}$ safely underflows to $0.0$, producing $\frac{0}{1 + 0} = 0.0$ stably without error.

