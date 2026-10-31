# Level 2 predictions (written BEFORE running any code)

**Problem:** predict `good_lunch` (1 = good, 0 = not good) from four logged numbers per lunch:
`queue_length`, `plate_waste_fraction`, `menu_board_kcal`, `handwriting_neatness`.
500 lunches. Model: logistic regression, trained with gradient descent.

## What I understand

- Logistic regression is a supervised classification model. It outputs a probability `p` that the lunch is good.
- The loss measures how wrong the model is. For logistic regression it is cross-entropy (log loss), not mean squared error.
- Formula, averaged over all n lunches (y = true label, p = predicted probability of a good lunch):
  `loss = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )`
- Loss is high when the model is **confident and wrong**. It is low when the model is **confident and right**.
- Gradient = derivative (the slope). Descent = moving towards lower loss.
- Learning rate (alpha) = how big a step to take on each update.
- Picture a hill: a small rate walks down slowly. A huge rate jumps to the other side of the valley, overshoots, and bounces back and forth.

## Before training

- Normalise every feature to the range 0 to 1, so no feature dominates because its numbers are bigger.
- Then every feature gets an equal say, and the model can learn which ones matter.

## Learning rate predictions

### 0.001 (tiny steps)
- Loss curve: drops very slowly.
- Convergence: I don't expect it to converge within my epoch limit.
- Test accuracy: if it does converge, high (about 98%).

### 0.1 (medium steps)
- Loss curve: reaches the minimum in a comfortable number of steps.
- Convergence: I expect it to converge.
- Test accuracy: about 90%.

### 10 (huge steps)
- Loss curve: jumps around and overshoots the minimum.
- Convergence: it might not converge.
- Test accuracy: low, about 50%.

## Which features matter

- Most important: `queue_length` (a very long queue spoils the mood) and `plate_waste_fraction` (more waste means a worse lunch).
- Least important: `menu_board_kcal` and `handwriting_neatness`.
