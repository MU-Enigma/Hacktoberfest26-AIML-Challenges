# Level 3, Option B: byte-level BPE tokenizer

`tokenizer.py` contains `BPETokenizer` with `train(text, num_merges)`, `encode(text) -> list[int]` and `decode(ids) -> str`. Pure Python, no dependencies (tests need `pytest`).

## How to run

From this folder:

```bash
python -m pytest -v        # run the tests
python tokenizer.py        # small demo (prints the example run below)
```

Use it from Python:

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train("abababab", 2)   # learns (97, 98) -> 256, then (256, 256) -> 257
tok.encode("abab")         # [257]
tok.decode([257])          # 'abab'
```

## How it works

1. **Bytes first.** Text is turned into UTF-8 bytes, so ids 0 to 255 are the starting tokens and any string can be represented.
2. **Train.** Count every adjacent pair of ids. Merge the most frequent pair into a new id (256, then 257, ...). On a tie, the smallest `(first_id, second_id)` tuple wins: `min(counts, key=lambda p: (-counts[p], p))`. Repeat `num_merges` times.
3. **Encode.** Start from the bytes and apply the learned merges in the order they were learned (the earliest-learned pair present is merged first).
4. **Decode.** Look up the bytes for each id, join them, decode as UTF-8.

## Rules I chose where the spec is silent

| Situation | What this code does |
|-----------|---------------------|
| Pre-splitting text into words | Not done. Pairs are counted over the whole byte sequence, so merges can cross spaces (e.g. `"e "`). This keeps the ids exactly what the spec describes. |
| Overlapping pairs, e.g. `"aaa"` | Counted at every position (`(97, 97)` = 2), but merged left to right without reusing a byte: `"aaa"` becomes `[256, 97]`. |
| A pair that appears only once | Still merged if it is the best pair left. |
| Text shorter than 2 bytes, or nothing left to merge | Training stops early, no crash. |
| `train` called twice | The second call starts from scratch. |
| `encode` before `train` | Returns the raw UTF-8 bytes. |
| `decode` of an id list that cuts a character in half | Shows `U+FFFD` instead of crashing (`errors="replace"`). Output of `encode` never does this. |
| `decode` of an unknown id | Raises `ValueError`. |

## Example run

```
$ python tokenizer.py
text bytes : 91
tokens     : 31
first merges:
  (224, 164) -> 256  b'\xe0\xa4'
  (97, 116) -> 257  b'at'
  (101, 32) -> 258  b'e '
  (104, 258) -> 259  b'he '
  (116, 259) -> 260  b'the '
ids        : [260, 265, 115, 262, 111, 110, 264, 109, 274, 264, 265, 257, 258, 260, 114, 274, 275, 271, 272, 266, 269, 135, 275, 271, 272, 266, 269, 135, 32, 273, 273]
round trip : True
unseen     : [260, 98, 257] -> the bat
```

The first merge is `(224, 164)`: in UTF-8 every Devanagari character from U+0900 to U+093F (which includes all the consonants) starts with those two bytes, so the Hindi part of the sample makes them the most common pair. 20 merges shrink 91 bytes to 31 tokens.

## Tests

`test_tokenizer.py` (29 tests) covers round trips for empty, ASCII, Hindi, Telugu, Chinese, emoji and mixed text; unseen text; merge ids starting at 256 and being consecutive; a hand-computed example that checks the exact merges and the tie-break (the working is in the test's comments); overlapping pairs; merge order during encode; `num_merges=0`; training on `""` and `"x"`; retraining; determinism; and decoding a broken or unknown id.
