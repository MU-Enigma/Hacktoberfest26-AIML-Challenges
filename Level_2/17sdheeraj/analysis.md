# error analysis

model test accuracy is 78% (misses 22 out of 100 test lunches). train acc is 83%.
some of these lunches are just weird edge cases that don't make obvious sense.

## 3 wrong examples

| test # | queue | waste | kcal | neatness | actual | model p | guess | why it failed |
|---|---|---|---|---|---|---|---|---|
| 5 | 50 | 0.517 | 2163 | 1.9 | 0 (bad) | 0.57 | 1 (good) | sitting right on the boundary. queue of 50 pushed it up, waste of 0.517 pulled it down, barely tripped over 0.5 |
| 12 | 42 | 0.645 | 1603 | 5.2 | 1 (good) | 0.18 | 0 (bad) | huge plate waste (0.65) and low calories made the model think it was garbage, but people liked it anyway (maybe bones/peels left on plate) |
| 33 | 56 | 0.377 | 2203 | 7.0 | 0 (bad) | 0.86 | 1 (good) | long queue and low waste tricked model into 86% confidence, but food was still bad (maybe counter was just moving super slow) |

## hypothesis & test
- **hypothesis**: handwriting neatness on the menu board has basically 0 weight (+0.19) and average is ~5.1 on both good and bad days. it's just random noise that doesn't actually help prediction.
- **test**: dropped `handwriting_neatness` completely and retrained model on the exact same 400/100 split.
- **result**: test accuracy stayed exactly 78.0% (weights: queue = +8.02, waste = -5.90, kcal = +1.61, bias = -2.25).
- **verdict**: hypothesis confirmed. fancy handwriting on the board doesn't make bad mess food taste any better lol.
