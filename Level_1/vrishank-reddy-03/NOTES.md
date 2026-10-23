# Notes

A linear model just adds up numbers, so it can output anything: 47, -300, 0.2. A probability has to sit between 0 and 1, where 0 means no chance and 1 means certain. A score of 47 does not mean 47 percent, it means nothing in that language. Sigmoid fixes this by squashing any number into the 0 to 1 range. A big positive score like 47 becomes almost exactly 1 (very likely you will like it). A big negative score becomes almost 0. A score of 0 becomes 0.5, a coin flip.

Sigmoid or tanh? Both are S-shaped, but sigmoid runs from 0 to 1 and tanh runs from -1 to 1. I would use sigmoid at the very end when the answer is a probability, because probabilities cannot be negative. I would prefer tanh for values passed between steps inside a model, because its output is centred on zero, so a song can push things up or down instead of only up. There is a catch: both go flat at extremes, so very large inputs stop teaching the model anything.
