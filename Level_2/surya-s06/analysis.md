# Analysis: Logistic Regression with Gradient Descent

## 1. Learning Rate Experiment

I experimented with three learning rates: 0.001, 0.05 and 0.5.

### Expected behaviour

- **0.001:** I expected slow convergence because the parameter updates would be small.
- **0.05:** I expected a more balanced learning process, with reasonably fast convergence.
- **0.5:** I expected faster convergence, although a very high learning rate could potentially overshoot the minimum or cause unstable loss.

### Actual results

| Learning Rate | Test Accuracy | Epochs | Converged | Final Loss |
|---|---:|---:|---|---:|
| 0.001 | 82% | 10000 | No | 0.423337 |
| 0.05 | 82% | 2334 | Yes | 0.401479 |
| 0.5 | 82% | 294 | Yes | 0.401478 |

The learning rate of 0.001 did not satisfy the convergence criterion within the maximum of 10,000 epochs.

The learning rates 0.05 and 0.5 converged successfully. The learning rate of 0.5 converged much faster and achieved almost the same final loss as 0.05.

All three models achieved 82% test accuracy.

### Observation

The experiment demonstrates that increasing the learning rate can significantly reduce training time. However, an excessively high learning rate can make training unstable. In this experiment, 0.5 worked well.

## 2. Error Analysis

I examined three incorrectly classified test examples using the model trained with a learning rate of 0.05.

| Queue Length | Plate Waste Fraction | Menu Board Kcal | Handwriting Neatness | Actual | Predicted |
|---:|---:|---:|---:|---:|---:|
| 28 | 0.609 | 1985 | 5.6 | 1 | 0 |
| 39 | 0.422 | 1884 | 3.4 | 1 | 0 |
| 58 | 0.491 | 1620 | 3.1 | 0 | 1 |

The first two examples are false negatives: the lunches were actually good, but the model predicted otherwise.

The third example is a false positive: the lunch was not good, but the model predicted that it was.

### Hypothesis

One possible reason for these errors is that the relationship between the input features and lunch quality may not be perfectly linear.

Different combinations of queue length, food waste, calories and handwriting neatness may produce similar feature values even when the actual lunch quality differs.

Since logistic regression learns a linear decision boundary, it may struggle with overlapping or more complex patterns.

## 3. Testing the Hypothesis

I tested whether handwriting neatness was contributing useful information by removing that feature and retraining the model.

| Model | Test Accuracy |
|---|---:|
| Original model (four features) | 82% |
| Reduced model (three features) | 82% |

**Change in accuracy: 0.00 percentage points**

### Conclusion

Removing handwriting neatness did not improve or reduce test accuracy on this particular train-test split.

This suggests that removing this feature alone was not enough to improve the model. The misclassifications may be caused by more complex relationships or overlapping feature values.

Further experiments with feature interactions or a different classification model could be explored in the future.