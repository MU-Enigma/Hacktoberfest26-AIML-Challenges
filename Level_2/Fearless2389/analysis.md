# Error analysis (Level 2, Fearless2389)

Model: logistic regression, lr 0.5, seed-42 80/20 split, features standardized with train statistics.
Test accuracy 0.750, so 25 of the 100 test lunches are wrong. All numbers below are printed by
`python logistic_regression.py` (section "Error analysis").

## Which 3 mistakes, and why these 3

I did not hand-pick. I took the three test mistakes where the model was **most confident** (predicted
probability furthest from 0.5). These are the most interesting ones, because "it was a coin flip near the
boundary" cannot explain a 6% or 91% prediction.

"Pushes logit by" is weight x standardized value: how much that feature moved the score `w.x + b`
before the sigmoid. Positive pushes toward "good", negative toward "not good". The bias adds -0.675 to every row.

### Example 1: CSV line 63, a good lunch the model called bad (P(good) = 0.062)

| feature | value | pushes logit by |
|---|---|---|
| queue_length | 35 | -0.230 |
| plate_waste_fraction | 0.75 | **-1.613** |
| menu_board_kcal | 2319 | +0.335 |
| handwriting_neatness | 1.2 | -0.542 |

Three-quarters of the food was left on the plates, which the model has learned strongly means "bad".

### Example 2: CSV line 378, a good lunch the model called bad (P(good) = 0.066)

| feature | value | pushes logit by |
|---|---|---|
| queue_length | 16 | **-2.357** |
| plate_waste_fraction | 0.343 | +0.665 |
| menu_board_kcal | 2033 | -0.022 |
| handwriting_neatness | 3.3 | -0.256 |

Only 16 people queued, shorter than about 88% of all logged lunches. A short queue is the model's strongest "bad" signal.

### Example 3: CSV line 42, a bad lunch the model called good (P(good) = 0.914)

| feature | value | pushes logit by |
|---|---|---|
| queue_length | 56 | **+2.122** |
| plate_waste_fraction | 0.377 | +0.475 |
| menu_board_kcal | 2203 | +0.190 |
| handwriting_neatness | 7.0 | +0.250 |

Long queue and little waste: everything about this lunch looks good, yet it was logged as bad.

## What I noticed

In all three, the **handwriting_neatness** term pushes in the wrong direction (-0.54, -0.26, +0.25).
That made me suspicious of that column, so I checked it against the data instead of just assuming:

| | correlation with good_lunch |
|---|---|
| handwriting_neatness, whole file (500 rows) | +0.008 |
| handwriting_neatness, train split (400 rows) | +0.076 |
| handwriting_neatness, test split (100 rows) | **-0.236** |
| queue_length, train / test | +0.581 / +0.508 |
| plate_waste_fraction, train / test | -0.452 / -0.414 |

Over the whole file, how neatly the menu board is written has essentially nothing to do with whether lunch
was good (+0.008; good and bad lunches have mean neatness 5.17 and 5.13). That also makes sense in the story:
the handwriting on the board does not change the food. But the model gave it weight +0.26, more than half of
`menu_board_kcal`'s weight, because in the 400 training rows there happens to be a weak positive link (+0.076).
In the test rows the chance link goes the other way (-0.236), so that weight actively hurts there.

## Hypothesis

**handwriting_neatness is a misleading (pure noise) feature. The model learned a weight for it from a chance
pattern in the training split, and that weight is part of why it gets these examples wrong. Dropping the column
should make these three mistakes less confident and improve test loss and accuracy.**

What I expected before testing: neatness only moves these three logits by 0.25-0.54, while their total logits
are about -2.7, -2.6 and +2.4, so I expected them to become *less wrong* but **not** to flip to correct.

## The test: drop one feature, change nothing else

Same seed-42 split, same lr 0.5, same convergence criterion, same zero starting weights; only
`handwriting_neatness` is removed (the scaler is refit on the remaining three train columns).

| | 4 features | 3 features (no neatness) |
|---|---|---|
| updates to converge | 151 | 147 |
| test accuracy | 0.750 | **0.760** |
| test loss (BCE) | 0.5062 | **0.4786** |
| CSV line 63, true 1, P(good) | 0.062 | 0.101 |
| CSV line 378, true 1, P(good) | 0.066 | 0.087 |
| CSV line 42, true 0, P(good) | 0.914 | 0.889 |

## Did it help?

**Partly.**

- On this split, yes: test loss fell by about 5% and one more test lunch was classified correctly. All three
  examples moved toward the right answer, as expected.
- But none of the three flipped. They are still badly wrong (10%, 9%, 89%). So neatness was **not the main
  reason** these three fail. The main reason is that the two strong, genuinely useful features (queue length
  and plate waste) point firmly the wrong way for these lunches. A good lunch where 75% of the food is wasted,
  or a bad lunch with a 56-person queue, is the kind of "genuinely hard to call" lunch that `datasets/README.md`
  warns about. A straight-line (linear) boundary on these columns cannot get them right without breaking many
  easier ones. They might even be logging mistakes in the label; I cannot tell from the data alone.
- How sure am I that dropping neatness helps in general? Not very. The script repeats the 4- vs 3-feature
  comparison on 20 other splits (seeds 0-19): mean test accuracy is 0.814 in **both** cases. Dropping neatness
  lowered the test loss on 13 of 20 splits and changed accuracy on only 6 of them (better on 4). That fits
  "neatness is noise": removing it costs nothing and usually helps the loss a little, but with 400 training rows
  the model already keeps the noisy weight fairly small.

**Conclusion:** the "misleading feature" half of the hypothesis is supported (the column carries no real signal
and removing it is slightly better), but the "this is why these three are wrong" half is not. These three errors
come from lunches whose strongest features contradict their label, which is a limit of the data and of a linear
model, not of the training procedure.
