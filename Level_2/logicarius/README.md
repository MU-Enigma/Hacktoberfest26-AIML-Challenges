# Level 2: logistic regression with gradient descent (logicarius)

Predicts `good_lunch` from four numbers per lunch. Pure Python, no scikit-learn.

## Files

- `predictions.md`: my predictions, written and committed before the code was run
- `logistic_regression.py`: the model, with the gradient and update step written by hand
- `loss_curves.png`: loss curves for three learning rates
- `analysis.md`: error analysis (3 wrong lunches, hypotheses, tests)
- `teaching_notebook.ipynb`: step-by-step explanation for a first-time reader

## How to run

From the repo root:

```
pip install matplotlib
python Level_2/logicarius/logistic_regression.py
```

- Takes about 30 seconds
- Prints epochs, convergence and test accuracy for learning rates 0.001, 0.1 and 10
- Saves `loss_curves.png` next to the script

The notebook runs on its own: open it in VS Code or Jupyter and choose Run All. It downloads the dataset if the repo file is not found.

## Results

| Learning rate | Converged | Epochs | Test accuracy |
|---|---|---|---|
| 0.001 | no | 20000 (cap) | 66% |
| 0.1 | yes | 7500 | 78% |
| 10 | no | 20000 (cap) | 75% |

Stop rule: the loss improves by less than 1e-6, or 20000 epochs.
Split: 400 train / 100 test, fixed seed 42.
