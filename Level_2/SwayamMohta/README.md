# Level 2: Learning by Gradient Descent

**Author:** Swayam Mohta (`@SwayamMohta`)  
**Level:** Level 2 (Oct 7 Workshop)

An end-to-end implementation of **Logistic Regression with Batch Gradient Descent** built from scratch using NumPy and Pandas (no scikit-learn models or optimizers used). Predicts whether mess lunch is good (`good_lunch = 1`) on the synthetic 500-sample dining dataset (`datasets/binary_classification.csv`).

---

## 1. Quick Start & Execution

From the repository root or inside this directory:

```bash
# Run logistic regression, compare learning rates, and output error analysis
python Level_2/SwayamMohta/logistic_regression.py
```

Generated deliverables:
- `loss_curves.png`: Training loss curves comparing learning rates ($\alpha \in \{0.01, 0.1, 1.0\}$).
- `analysis.md`: Detailed error analysis of 3 misclassified test cases, hypothesis, and experimental test.
- `teaching_notebook.ipynb`: Educational Jupyter notebook teaching the model step-by-step from scratch.

---

## 2. Experimental Setup

- **Dataset:** `datasets/binary_classification.csv` (500 lunches, 4 continuous features).
- **Train/Test Split:** 80% train (400 samples), 20% test (100 samples) with fixed seed `seed = 42`.
- **Feature Scaling:** Standardization ($z = \frac{x - \mu}{\sigma}$) computed strictly on the training set to prevent data leakage.
- **Convergence Criterion:** Change in training binary cross-entropy loss between consecutive epochs $< 10^{-6}$ ($|\mathcal{L}_t - \mathcal{L}_{t-1}| < 10^{-6}$).

---

## 3. Learning Rate Comparison & Predictions

### Predictions Made Before Running
1. **$\alpha = 0.01$ (Low):** Slow, steady convergence; will likely take 2,000+ iterations without reaching the true minimum before max iterations.
2. **$\alpha = 0.1$ (Moderate):** Balanced and stable exponential descent; will reach convergence in a few hundred epochs.
3. **$\alpha = 1.0$ (High):** Rapid initial drop in the first 10–20 epochs; converges very quickly without divergence due to feature standardization.

### Actual Results

| Learning Rate ($\alpha$) | Convergence Iterations | Training Time (ms) | Final Train Loss | Test Accuracy | Test BCE Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | 2000 (Hit max) | 83.37 ms | 0.375038 | **73.00%** | 0.516883 |
| **0.10** | 669 | 29.64 ms | 0.366810 | **73.00%** | 0.547301 |
| **1.00** | 94 | 3.99 ms | 0.366699 | **73.00%** | 0.551221 |

- **Loss Curves:** Saved to [`loss_curves.png`](loss_curves.png).
- **Prediction Evaluation:** All predictions held true. With standardized features, $\alpha = 1.0$ converged in just 94 iterations without instability.

---

## 4. Learned Weights & Model Interpretation

| Feature | Weight ($w$) | Interpretation |
| :--- | :---: | :--- |
| `queue_length` | **+1.9620** | Strongest positive indicator (longer queues signal popular food). |
| `plate_waste_fraction` | **-1.3014** | Strongest negative indicator (high waste means food was discarded). |
| `menu_board_kcal` | **+0.3317** | Slight positive effect. |
| `handwriting_neatness` | **+0.1113** | Minimal to no predictive power (noise). |
| `bias` (intercept) | **-0.7468** | Negative baseline prior. |

---

## 5. Error Analysis Summary

See [`analysis.md`](analysis.md) for full breakdown:
- **Case 1 (Row 488 - Boundary):** Predicted $P = 0.475$ vs true label 1 due to neutral queue length sitting near average.
- **Case 2 (Row 161 - Feature Conflict):** Predicted $P = 0.362$ vs true label 1 because an unusually short queue overrode low plate waste.
- **Case 3 (Row 401 - Label Noise):** Predicted $P = 0.909$ vs true label 0 despite long queue and low waste, reflecting unobserved qualitative dining factors.
- **Hypothesis & Test:** Dropping `handwriting_neatness` reduced test log loss from `0.5473` to `0.5447` and improved convergence time by 18% with zero loss in test accuracy (73.00%).
