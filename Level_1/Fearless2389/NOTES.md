# Notes for a friend with no maths

**Score of 47 — why's that a problem?**

A probability has to be between 0 and 1. A linear model's raw score can be
anything -- 47, -300, a million -- so 47 isn't a probability yet, it's just
the model's raw "amount of liking." Sigmoid is the fix: it squashes any
number into (0, 1) while keeping the order, so a bigger raw score still
means a bigger probability. Huge positive scores land near 1, huge negative
scores land near 0, and 0 lands at 0.5. Sigmoid turns "the model's opinion"
into "a number you can call a chance."

**Sigmoid vs tanh**

Both are S-shaped squashing curves. Sigmoid squashes into (0, 1); tanh
squashes into (-1, 1), centered on 0. Pick sigmoid when the output should
read as a probability or an on/off gate -- "chance you like this song."
Pick tanh when negative and positive inputs are genuinely opposite, not
just "less likely" vs "more likely," and when you want outputs centered on
zero, which tends to help a network train faster since the average output
isn't dragging everything one way.
