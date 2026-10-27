# Level 2 Error Analysis: Predicting Mess Lunch Quality

**Author:** Swayam Mohta (`@SwayamMohta`)  
**Task:** Error analysis on 500 synthetic mess lunches (`datasets/binary_classification.csv`)

---

## 1. Model Baseline Summary

- **Architecture:** Logistic Regression trained via Batch Gradient Descent from scratch.
- **Split:** Deterministic 80/20 train/test split (400 train, 100 test, `random_state = 42`).
- **Standardization:** Zero-mean, unit-variance computed strictly on the training set.
- **Convergence Criterion:** $\Delta \text{Loss} = |\mathcal{L}_t - \mathcal{L}_{t-1}| < 10^{-6}$.
- **Performance:**
  - **Test Accuracy:** 73.00%
  - **Test Log Loss (BCE):** 0.5473
  - **Learned Weights:**
    - `queue_length`: `+1.9620` (strongest positive predictor)
    - `plate_waste_fraction`: `-1.3014` (strongest negative predictor)
    - `menu_board_kcal`: `+0.3317` (slight positive correlation)
    - `handwriting_neatness`: `+0.1113` (near zero effect)
    - `bias`: `-0.7468`

Out of 100 test samples, the model misclassifies **27 examples**.

---

## 2. Three Misclassified Examples

Below are three representative cases the model got wrong:

| Test Index | True Label | Predicted Label | Predicted Prob ($\hat{P}$) | `queue_length` | `plate_waste_fraction` | `menu_board_kcal` | `handwriting_neatness` | Failure Category |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Row 488** | **1** (Good) | **0** (Bad) | `0.475` | 38 | 0.322 | 1564 | 5.9 | Boundary Case |
| **Row 161** | **1** (Good) | **0** (Bad) | `0.362` | 33 | 0.306 | 1705 | 4.2 | Feature Conflict |
| **Row 401** | **0** (Bad) | **1** (Good) | `0.909` | 54 | 0.248 | 1740 | 3.5 | Inherent Noise / Anomaly |

### Detailed Breakdown

1. **Row 488 (The Boundary Case):**
   - **Observation:** The model predicted $\hat{P} = 0.475$, falling just below the $0.50$ decision threshold.
   - **Reasoning:** Queue length (38) sits almost exactly at the population mean (36.9). Even though plate waste is low (0.322, indicating students liked the food), the moderate queue length combined with the negative intercept ($-0.7468$) kept the probability slightly below the cutoff.

2. **Row 161 (Feature Conflict):**
   - **Observation:** True label is 1 (good lunch), but predicted as 0 ($\hat{P} = 0.362$).
   - **Reasoning:** Plate waste is very low (0.306), which is a clear hallmark of delicious food. However, the queue happened to be below average (33 students), possibly due to off-peak arrival or bad weather. Because `queue_length` has the highest absolute weight (`+1.9620`), the shorter queue penalized the score too harshly.

3. **Row 401 (Misleading Signals / Label Noise):**
   - **Observation:** High-confidence false positive ($\hat{P} = 0.909$), but true label is 0.
   - **Reasoning:** The queue was long (54 students) and waste was very low (0.248). By all standard metrics, this looks like a great lunch. Yet diners rated it poorly. This represents real-world noise: perhaps a popular dessert created an initial long line, but the main curry was substandard, or the food ran out. A linear model cannot capture unmeasured qualitative factors.

---

## 3. Hypothesis & Experimental Test

### Hypothesis
The feature `handwriting_neatness` shows virtually zero statistical correlation with meal quality ($r \approx +0.008$) and carries an insignificant model weight (`+0.1113`). In a dataset of 500 rows, including irrelevant noise features increases model variance and degrades test log loss.
**Hypothesis Statement:** Dropping `handwriting_neatness` will reduce model complexity, mitigate noise fitting, and improve test loss without harming accuracy.

### The Experiment
We retrained the identical model architecture (seed 42, $\alpha = 0.1$, $\Delta \text{Loss} < 10^{-6}$) on the exact same train/test split, removing `handwriting_neatness`:

| Model Configuration | Features Used | Test Accuracy | Test Log Loss | Training Time |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (4 features)** | `queue`, `waste`, `kcal`, `neatness` | **73.00%** | **0.5473** | 29.6 ms |
| **Experiment (Drop `neatness`)** | `queue`, `waste`, `kcal` | **73.00%** | **0.5447** | 24.1 ms |
| **Minimalist (Top 2 only)** | `queue`, `waste` | **73.00%** | **0.5502** | 18.5 ms |

### Outcome & Conclusion
- **Outcome:** Test log loss improved from `0.5473` down to `0.5447`, and test accuracy remained unchanged at `73.00%`.
- **Conclusion:** The hypothesis is **supported**. Removing `handwriting_neatness` eliminated noise, sharpened the model's predicted probability calibration, and sped up convergence time by ~18% with zero penalty to classification accuracy.
