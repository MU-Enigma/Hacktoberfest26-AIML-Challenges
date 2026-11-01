# notes

## why a score of 47 makes no sense
probabilities can only live between 0 and 1 (meaning 0% to 100%). 0 means no way, and 1 means guaranteed yes. if a model gives 47, that would mean a 4700% chance, which is impossible nonsense.

sigmoid fixes this by acting like a clamp that squashes any number into the 0 to 1 range. huge negative numbers get pressed near 0, 0 turns into 0.5 (a pure coin flip), and big positive numbers like 47 get squashed right up against 1.0 (you will definitely love it).

## sigmoid vs tanh
they look similar, but their ranges differ.

sigmoid outputs between 0 and 1, so pick it whenever you need a probability or a yes-no answer (like "will i enjoy this track?").

tanh outputs between -1 and 1, with 0 right in the center. pick tanh when data has positive and negative directions (like rating mood from -1 very sad to +1 super energetic, where 0 is neutral). having zero in the middle keeps calculations balanced.