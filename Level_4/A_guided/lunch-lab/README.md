# lunch-lab

A tiny machine learning pipeline that answers one question: **will the mess lunch be good today?**

It loads a CSV of logged lunches, scales the features, trains a model (logistic regression or k-nearest-neighbours, both written in NumPy), evaluates it, saves it, and predicts for new lunches. Everything runs on a laptop CPU in a couple of seconds.

The data is synthetic. It is made up for teaching, not taken from a real mess.

## This project is broken on purpose

This is the Level 4A project of Enigma's Hacktober. Some pieces are missing or wrong, and each one is an issue tagged `level4-guided`. Pick one, comment to claim it, wait to be assigned, then fix it. The rules are in the [main README](../../../README.md).

Because pieces are missing, `train`, `evaluate` and `predict` will not fully work until the issues that fix them are merged. The tests that are here should pass from the start.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirement.txt
```

## Check that your setup works

Run this from the root of the project (this `lunch-lab` folder):

```bash
python -m pytests
```

## Trying the CLI (once the pieces exist)

```bash
python -m lunchlab.cli train --data lunches.csv --out model.json
python -m lunchlab.cli evaluate --model-file model.json --data lunches.csv
python -m lunchlab.cli predict --model-file model.json --input samples/new_lunches.csv
```

## Project layout

```
lunchlab/
├── data.py             load_csv, train_test_split
├── preprocessing.py    MinMaxScaler, StandardScaler
├── models/
│   ├── knn.py          KNNClassifier
│   └── logistic.py     sigmoid, binary_cross_entropy, LogisticRegression
├── metrics.py          accuracy, precision, recall, f1_score, confusion_matrix
├── persistence.py      save_bundle, load_bundle (model + scaler as JSON)
└── cli.py              train, evaluate, predict
data/lunches.csv        500 logged lunches
samples/                a few new lunches to predict
tests/                  pytest tests
```

## The data

`data/lunches.csv` has one row per lunch:

| Column | Meaning |
|--------|---------|
| `queue_length` | people in the queue at 12:30 |
| `plate_waste_fraction` | fraction of plates returned with food left (0 to 1) |
| `menu_board_kcal` | calories the menu board claims |
| `handwriting_neatness` | how neat the menu board handwriting is (1 to 10) |
| `good_lunch` | 1 if the lunch was good, 0 if not (the label) |

## Working on an issue

1. Read the issue and the docstring of the function it points to. The docstring says exactly what the function must do.
2. Reproduce the problem, or write a test that shows it.
3. Fix it, and add or extend tests in `tests/` that cover your fix.
4. Keep the change small: one issue, one pull request.
5. In your PR description say the root cause, your fix, and which AI tools you used (or none).

## Code conventions

- Type hints where they help, and Google-style docstrings.
- Tests use `pytest`, in `tests/`, in files named `test_*.py`.
- Set seeds where randomness is involved, so results can be reproduced.
