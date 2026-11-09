# Notes

## Song Similarity

Cosine similarity measures how closely two sets of song features match. A score closer to 1 means the songs are more similar.

Normalization puts different features, like tempo and energy, on the same 0–1 scale. Without it, features with larger numbers can influence the result more.

My prediction before calculating was **Rooftop Monsoon**. It ranked first without normalization, but Fresher Night ranked first after normalization. The top-three rankings did not agree.

## Squashing Functions

Squashing functions change numbers into useful ranges.

- **Sigmoid:** Converts a number into a value between 0 and 1. Sigmoid(47) is almost 1, but a raw score of 47 is not automatically a probability.
- **ReLU:** Changes negative numbers to 0 and keeps positive numbers unchanged.
- **Tanh:** Converts a number into a value between -1 and 1.

Sigmoid is useful for binary classification because its output is between 0 and 1. Tanh is useful when values need to be centred around zero and can be positive or negative.

The naive sigmoid formula overflows for very large negative inputs such as -1000. I used an equivalent stable formula for negative inputs to avoid this problem.
