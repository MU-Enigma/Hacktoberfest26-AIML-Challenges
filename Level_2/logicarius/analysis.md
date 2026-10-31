# Error analysis

## Setup

- Model: logistic regression, learning rate 0.1
- Stops when the loss improves by less than 1e-6 (converged at 7500 epochs)
- Split: 400 train / 100 test, fixed seed 42
- Features: min-max normalized, with min and max taken from train only
- Test accuracy: **78%** (22 of 100 wrong)
- Train accuracy: **83%**

## What a typical lunch looks like

Average of each feature over all good lunches and over all bad lunches (500 lunches in total).

| Feature | Good lunches | Bad lunches |
|---|---|---|
| queue_length | 48 | 29 |
| plate_waste_fraction | 0.35 | 0.55 |
| menu_board_kcal | 2092 | 2016 |
| handwriting_neatness | 5.2 | 5.1 |

## Three lunches the model got wrong

`p` is the model's probability that the lunch is good.

| # | queue_length | plate_waste_fraction | menu_board_kcal | handwriting_neatness | actual | p |
|---|---|---|---|---|---|---|
| 1 | 24 | 0.774 | 2009 | 3.1 | good | 0.02 |
| 2 | 49 | 0.175 | 1751 | 6.5 | not good | 0.88 |
| 3 | 56 | 0.377 | 2203 | 7.0 | not good | 0.86 |

## Hypotheses and tests

Baseline test accuracy: 78%.
Each test changes one small thing, retrains at lr 0.1 on the same split, and compares.

### Lunch 1 (actually good, model says bad)

- Hypothesis: short queue (24) and high waste (0.77) both point to "bad"
- Also thought: kcal might explain it, since a filling meal feels good
- Test: drop `menu_board_kcal`
- Result: 78% to 75%, so kcal helps a little
- But this lunch has 2009 kcal, below the good-lunch average (2092)
- Verdict: kcal does not explain it, probably a hard-to-call lunch

### Lunch 2 (actually not good, model says good)

- Hypothesis: long queue (49) and low waste (0.18) both say "good"
- Kcal is very low (1751), so maybe too few calories ruins a lunch
- Test: add a squared kcal term
- Result: 78% to 78%, no help

### Lunch 3 (actually not good, model says good)

- Hypothesis: a queue of 56 is very long, maybe too crowded
- Test: add a squared queue term
- Result: 78% to 79% (one lunch), too small to trust

## Other ideas checked

### Big numbers (no normalization)

| Scaling | Test accuracy |
|---|---|
| Raw numbers, lr 0.1 | 57% |
| Raw numbers, lr 0.0001 | 61% |
| Min-max (my model) | 78% |
| Z-score | 79% |

- Normalizing is needed
- It is not the cause of the remaining errors: z-score lands at the same level
- Trap: the raw run at lr 0.1 "converged" after 6 epochs, but the loss was just stuck

### Lowering a feature's weight

Halved one learned weight by hand and kept the rest.

| Feature | Learned weight | Test accuracy with weight halved |
|---|---|---|
| queue_length | +7.9 | 62% |
| plate_waste_fraction | -5.9 | 68% |
| menu_board_kcal | +1.6 | 78% |
| handwriting_neatness | +0.2 | 78% |

- Lowering queue or waste hurts a lot, so the model needs them
- Lowering kcal or neatness changes nothing, they are already small
- The learned weights are already the best balance

## Conclusion

- These are not rare outliers: train accuracy is 83%, so about 1 lunch in 5 is wrong even on training data
- Good and bad lunches overlap on these four features, so some lunches are hard to call (the dataset notes say so too)
- `handwriting_neatness` carries almost no information: dropping it leaves accuracy at 78%

## What I got wrong in my predictions

- lr 0.001: slow and not converged, as I guessed. Test accuracy was 66%
- lr 0.1: guessed about 90%, got 78%
- lr 10: guessed about 50%, got 75%
- `queue_length`: most important, as I guessed, but the direction is opposite
  - Longer queue means a more likely good lunch (48 vs 29)
  - Probably popularity, not mood
- `plate_waste_fraction`: second most important, more waste means a worse lunch
