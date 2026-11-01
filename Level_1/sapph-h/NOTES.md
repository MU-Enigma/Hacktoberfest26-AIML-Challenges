# Level 1 Notes

### 1. The Score of 47 and Sigmoid

A probability must be between 0 and 1 (meaning 0% to 100%). A score of 47 is meaningless as a probability—you cannot have a 4,700% chance of enjoying a track!

Sigmoid acts like an elastic band. It squashes any score, no matter how huge or negative, into a smooth curve strictly between 0 and 1. Huge positive scores get squashed close to 1 (virtually certain you will like it), big negative scores close to 0 (virtually certain you will not), and a score of 0 becomes 0.5 (an exact 50/50 toss-up).

### 2. Sigmoid vs. Tanh

Sigmoid squashes values into (0, 1), whereas tanh squashes them into (-1, 1).

- **Prefer Sigmoid** for probabilities or binary decisions (e.g., "Will you like this song? Yes or No"), since probabilities can never be negative.
- **Prefer Tanh** when data naturally has opposites and a neutral middle ground (e.g., sentiment where -1 is "hate", 0 is "neutral", and +1 is "love"). Tanh is centered at zero, which also keeps intermediate signals balanced.

