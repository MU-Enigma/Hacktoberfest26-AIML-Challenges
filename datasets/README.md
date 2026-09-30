# Datasets

All data here is **synthetic**: made up for teaching, not taken from a real mess. The story is *"will the mess lunch be good today?"*

## `past_lunches.csv` (Level 1)

8 past lunches, one per row.

| Column | Meaning |
|--------|---------|
| `dish` | name of the dish (a label only, not a number to compute with) |
| `queue_length` | number of people in the queue at 12:30 |
| `plate_waste_fraction` | fraction of plates returned with food left (0 to 1) |
| `menu_board_kcal` | calories the menu board claims for the day |
| `handwriting_neatness` | how neat the handwriting on the menu board is (1 to 10) |

## `binary_classification.csv` (Level 2)

500 logged lunches, one per row. Same four numeric columns as above, plus the label:

| Column | Meaning |
|--------|---------|
| `good_lunch` | 1 if the lunch was good, 0 if not |

Some lunches are genuinely hard to call, so expect your model to get a few wrong. That is the point of the error-analysis task.