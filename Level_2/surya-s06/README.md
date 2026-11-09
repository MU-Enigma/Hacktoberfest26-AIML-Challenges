
# Level 2: Learning by Gradient Descent

## Objective
Implement logistic regression from scratch using gradient descent to predict whether a mess lunch is good (1) or not good (0).

## Dataset
Used the provided binary_classification.csv dataset containing 500 synthetic lunch records.

Input features:
- queue_length
- plate_waste_fraction
- menu_board_kcal
- handwriting_neatness

Target: good_lunch

## Implementation
- Implemented sigmoid activation.
- Implemented binary cross-entropy loss.
- Calculated gradients manually.
- Updated weights and bias using gradient descent.
- Used an 80/20 train-test split with random seed 42.
- Standardized features using training-set statistics.
- Used a fixed convergence criterion: loss change below 1e-9 for 20 consecutive epochs.

## Learning Rate Experiments
Compared learning rates 0.001, 0.05 and 0.5.

Predictions before running:
- 0.001: expected to converge slowly.
- 0.05: expected to provide a stable balance.
- 0.5: expected to converge faster, but potentially make unstable updates.

Actual results are recorded in the terminal output.

## Files
- logistic_regression.py: model implementation and experiments.
- analysis.md: error analysis and hypothesis test.
- teaching_notebook.ipynb: beginner-friendly explanation.
- outputs/loss_curves.png: comparison of loss curves.
- outputs/misclassified_examples.csv: incorrect predictions.

## Run
```bash
python logistic_regression.py
```

## AI Assistance
ChatGPT assisted with code drafting, explanations and debugging. The implementation was reviewed and tested by the contributor.
