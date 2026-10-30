# Level 3, Option A: N-gram text generator (Fearless2389)

A command-line tool that learns which words follow which in a text file, then
generates new text from those counts. Pure Python standard library, no installs.

## How to run

```
python generate.py --n 2 --seed 42 --length 50 --input sample.txt
```

| Flag | Meaning |
|------|---------|
| `--n` | Context size: how many previous words are looked at when picking the next one. Must be at least 1. Default 2. |
| `--seed` | Random seed. The same seed, file, `--n` and `--length` always print the same words. If left out, each run differs. |
| `--length` | Exactly how many words to print (0 or more). Default 50. |
| `--input` | Path to a UTF-8 text file (required). |

The output is one line of space-separated words, so `len(output.split()) == --length`.

**About `--n`:** the spec says `--n` sets the *context size*, so I took that
literally. `--n 1` looks at the last 1 word (a classic "bigram" model) and
`--n 2` looks at the last 2 words (a "trigram" model).

## Example run

`sample.txt` is a short original paragraph made for testing (172 words, 80 different words).

```
$ python generate.py --n 2 --seed 42 --length 50 --input sample.txt
boat waits by the mill, and when the river carries the leaves of autumn. In the morning and they bring him tea in the evening. When the river remembers everything that floats on it. The children of the town. The town is small and the river carries the leaves of
```

That is exactly 50 words, and running it again prints the same line.

## How it works

1. **Tokenize:** split the text on whitespace. Punctuation stays attached
   (`autumn.` is one word) and capitals are kept, so the output keeps the
   sentence feel of the source.
2. **Count:** for every word in the text, record it as a follower of the 0, 1,
   ..., n words right before it. The 0-word context `()` is just plain word
   frequency (the unigram counts).
3. **Generate:** to pick each next word, take the last n generated words. If
   that context was seen in the text, sample a follower weighted by its count.
   If not, **back off**: drop the oldest word and try again, all the way down
   to `()` (plain word frequencies), which always exists for a non-empty file.
   The very first word has no history, so it is drawn from plain word frequencies.

Reproducibility: all randomness comes from one local `random.Random(seed)`,
and candidates are stored in dicts (which keep insertion order), never sets,
so the order of choices never changes between runs.

## Edge cases

| Input | Behaviour |
|-------|-----------|
| Corpus shorter than `n` words (even 1 word) | Works. Longer contexts were never seen, so it backs off to shorter ones. |
| Last word of the file (nothing ever follows it) | Backs off to plain word frequencies and continues. |
| `--length 0` | Prints an empty line (zero words), exit code 0. |
| Empty or whitespace-only file | Nothing can be learned, so it prints `error: ... the corpus has no words to learn from` to stderr and exits with code 1 (no traceback). With `--length 0` it still prints zero words. |
| Missing or non-UTF-8 file, `--n 0`, negative `--length`, non-numbers | argparse error message, exit code 2. |
| Non-English text (e.g. `café 日本語`) | Read as UTF-8 (a Windows Notepad BOM is ignored) and printed as UTF-8. |

## Tests

From this folder:

```
python -m pytest -q
```

27 tests cover: exact word count, same seed gives the same output, different
seeds give different output, backoff (to a shorter context and down to
unigrams at the dead-end last word), corpora shorter than `n`, the empty
corpus, every generated word being from the corpus, `n = 1`, `--length 0`,
non-English text, and bad arguments. Some tests call the functions directly,
others run `generate.py` as a real command.
