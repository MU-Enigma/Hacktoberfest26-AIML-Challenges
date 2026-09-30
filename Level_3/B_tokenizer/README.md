# Level 3, Option B: Byte-pair-encoding (BPE) tokenizer

Put your work in **`Level_3/B_tokenizer/<your-github-username>/`**.

## The spec

The piece that turns text into the numbers a language model reads.

- File name: **`tokenizer.py`**, containing a class **`BPETokenizer`** with three methods:
  - `train(text, num_merges)`
  - `encode(text) -> list[int]`
  - `decode(ids) -> str`
- **Byte-level:** the base tokens are the 256 byte values (ids 0 to 255). Each new merged token gets the next id (256, 257, ...) in the order the merges are made
- **Tie-breaking (required, so results are comparable):** always merge the most frequent adjacent pair. If there is a tie, pick the pair that is smallest as a `(first_id, second_id)` tuple
- `decode(encode(x)) == x` must hold for any input, including empty strings and non-English text

## What to submit

```
Level_3/
└── B_tokenizer/
    └── your-github-username/
        ├── tokenizer.py
        ├── test_tokenizer.py    your own tests
        └── README.md            how to run + one example run
```

## In your PR description

One real design decision and the alternative you rejected. A prompt to get you started: what did you do about text that contains characters outside plain ASCII, and why?