# Level 3, Option A: N-gram text generator

Put your work in **`Level_3/A_ngram/<your-github-username>/`**.

## The spec

A command-line tool that learns which words follow which, then generates new text.

- File name: **`generate.py`**
- Interface: `python generate.py --n 2 --seed 42 --length 50 --input text.txt`
- `--n` sets the context size, `--length` sets how many words to generate (it prints exactly that many), and `--seed` must make the output fully reproducible
- **Backoff:** if the current context has never been seen, fall back to a shorter context
- Must not crash on a corpus shorter than `n` words

## What to submit

```
Level_3/
└── A_ngram/
    └── your-github-username/
        ├── generate.py
        ├── test_generate.py     your own tests
        └── README.md            how to run + one example run
```

## In your PR description

One real design decision and the alternative you rejected. A prompt to get you started: what happens with a word that never appears mid-sentence, and what did you do about it?