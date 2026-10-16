A linear model might give a song a score of 47, but that doesn't make sense as a probability because probabilities have to be between 0 and 1. Sigmoid fixes this by turning the score into a number between 0 and 1, which can be treated as a probability.

Sigmoid and tanh are similar because they both squash numbers into a smaller range. The difference is that sigmoid gives values between 0 and 1, while tanh gives values between -1 and 1.

I would use sigmoid when I want to represent the chance of something happening, like whether I will enjoy a song. I would use tanh when negative and positive values are useful, because it can show both sides of a result.
