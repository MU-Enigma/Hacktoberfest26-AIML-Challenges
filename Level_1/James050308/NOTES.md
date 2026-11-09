# Notes

## Song Similarity

Cosine similarity measures how closely two sets of song features match. A score closer to 1 means the songs are more similar.

Normalization puts different features, like tempo and energy, on the same 0–1 scale. Without it, features with larger numbers can influence the result more.

My prediction before calculating was **Rooftop Monsoon**.

Without normalization, Rooftop Monsoon ranked first. After normalization, Fresher Night ranked first. The top-three rankings did not agree.

## Squashing Functions

Squashing functions change numbers into useful ranges.

- **Sigmoid:** Converts a number into a value between 0 and 1. Sigmoid(47) is almost 1.
- **ReLU:** Changes negative numbers to 0 and keeps positive numbers unchanged.
- **Tanh:** Converts a number into a value between -1 and 1.

These functions are commonly used in machine learning to process values and control their ranges.
