# Datasets

All data here is **synthetic**: made up for teaching. Level 1 is a song recommender, Levels 2 and 4A use the *"will the mess lunch be good today?"* story.

## `songs.csv` (Level 1)

10 made-up songs, one per row. The titles and numbers are invented, not taken from any music service.

| Column | Meaning |
|--------|---------|
| `title` | name of the song (a label only, not a number to compute with) |
| `tempo_bpm` | beats per minute |
| `duration_sec` | length in seconds |
| `energy` | how intense the song feels (0 to 1) |
| `danceability` | how easy it is to dance to (0 to 1) |

## `binary_classification.csv` (Level 2)

500 logged lunches, one per row. Four numeric columns (`queue_length`, `plate_waste_fraction`, `menu_board_kcal`, `handwriting_neatness`) plus the label:

| Column | Meaning |
|--------|---------|
| `good_lunch` | 1 if the lunch was good, 0 if not |

Some lunches are genuinely hard to call, so expect your model to get a few wrong. That is the point of the error-analysis task.