# Level 1 Notes

## My prediction
I predicted **Canteen Bass**: its energy (0.82) and danceability (0.85) almost match mine. It came 2nd both times; Fresher Night is closer overall.

## Results
| Rank | Without normalization | With min-max |
|------|-----------------------|--------------|
| 1 | Rooftop Monsoon | Fresher Night |
| 2 | Fresher Night | Canteen Bass |
| 3 | Canteen Bass | Hostel Corridor |

**Agree? No.** Without scaling, duration (~200) and tempo (~120) dwarf energy and danceability (~0.8), so cosine mostly compares length and speed. Rooftop Monsoon matches those but has 0.3 energy: a calm song. After scaling, every feature counts equally and Fresher Night wins.

## Why a score of 47 is a problem
A probability must be between 0 and 1; "47" isn't a chance. Sigmoid squashes any number into 0 to 1: big scores land near 1, negative ones near 0.

## Sigmoid vs tanh
Same S-shape, different range. Sigmoid (0 to 1) suits yes/no chances. Tanh (-1 to 1) suits "like vs dislike, and how much".

## Breaking sigmoid
`1/(1+e^-x)` at x = -710 needs e^710, too big for a float: overflow. Fix: for negative x use `e^x/(1+e^x)`.
