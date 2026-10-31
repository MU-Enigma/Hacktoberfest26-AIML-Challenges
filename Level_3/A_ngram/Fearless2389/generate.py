"""N-gram text generator with backoff.

Usage:
    python generate.py --n 2 --seed 42 --length 50 --input text.txt

How it works, in three steps:
1. Split the input text into words (on whitespace).
2. For every position in the text, count which word came next after the
   previous 1, 2, ..., n words (the "context"). Also count plain word
   frequencies, which is the same thing with an empty context.
3. Generate words one at a time: look at the last n generated words, use the
   longest ending of them that appeared in the text, and pick the next word
   at random, weighted by how often it followed that context.
   If even a 1-word context was never seen, fall back to plain word
   frequencies (backoff).

--n is the context size: the number of previous words looked at.
So --n 1 looks at one previous word (a "bigram" model),
and --n 2 looks at two previous words (a "trigram" model).
"""

import argparse
import random
import sys


def tokenize(text):
    """Split text into words on any whitespace.

    Punctuation stays attached ("end." is one word) and case is kept,
    so generated text keeps the punctuation and capitals of the source.
    """
    return text.split()


def build_model(words, n):
    """Count which word follows each context of 0 to n words.

    Returns {"n": n, "counts": counts}, where counts maps a context
    (a tuple of words) to a dict {next_word: how_many_times}.
    The empty tuple () is the 0-word context, so counts[()] holds the
    plain frequency of every word in the corpus.

    Example with words = ["a", "b", "a"] and n = 1:
        counts[()]     == {"a": 2, "b": 1}
        counts[("a",)] == {"b": 1}
        counts[("b",)] == {"a": 1}
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    counts = {}
    for i, word in enumerate(words):
        # Try every context length k from 0 up to n that fits before word i.
        for k in range(0, min(n, i) + 1):
            context = tuple(words[i - k:i])  # the k words just before word i
            followers = counts.setdefault(context, {})
            followers[word] = followers.get(word, 0) + 1
    return {"n": n, "counts": counts}


def find_context(model, history):
    """Return the longest ending of `history` that was seen in the corpus.

    "Seen" means some word followed it in the text. We start with the last
    n words and drop the oldest word until we find a known context.
    The empty context () is always known for a non-empty corpus, so this
    is where backoff ends up in the worst case (plain word frequencies).
    """
    counts = model["counts"]
    longest = min(model["n"], len(history))
    for k in range(longest, 0, -1):
        context = tuple(history[-k:])  # the last k words
        if context in counts:
            return context
    return ()


def generate(model, length, seed):
    """Generate exactly `length` words as a list.

    The same model, length and seed always give the same list, because all
    randomness comes from one random.Random(seed), and dicts keep insertion
    order, so the candidate words are always listed in the same order.
    """
    if length == 0:
        return []
    if () not in model["counts"]:
        raise ValueError("the corpus has no words to learn from")

    rng = random.Random(seed)  # local generator: does not touch global random
    output = []
    for _ in range(length):
        followers = model["counts"][find_context(model, output)]
        candidates = list(followers.keys())
        weights = list(followers.values())
        # Pick one word, more frequent followers are more likely.
        output.append(rng.choices(candidates, weights=weights)[0])
    return output


def int_at_least(minimum):
    """Make an argparse type that accepts whole numbers >= minimum."""
    def parse(value):
        number = int(value)  # argparse turns a ValueError into a clean error
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {number}")
        return number
    return parse


def main(argv=None):
    """Command-line entry point: read the file, build the model, print words."""
    parser = argparse.ArgumentParser(description="Generate text with an n-gram model.")
    parser.add_argument("--n", type=int_at_least(1), default=2,
                        help="context size: how many previous words to look at (default 2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; the same seed gives the same output")
    parser.add_argument("--length", type=int_at_least(0), default=50,
                        help="exact number of words to print (default 50)")
    parser.add_argument("--input", required=True, help="path to a UTF-8 text file")
    args = parser.parse_args(argv)

    try:
        # utf-8-sig reads normal UTF-8 and also drops the invisible marker
        # (BOM) that Windows Notepad sometimes puts at the start of a file.
        with open(args.input, encoding="utf-8-sig") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as error:
        parser.error(f"could not read {args.input}: {error}")  # exits with code 2

    model = build_model(tokenize(text), args.n)
    try:
        words = generate(model, args.length, args.seed)
    except ValueError as error:
        sys.exit(f"error: {args.input}: {error}")  # message to stderr, exit code 1

    # Windows may not print non-English characters by default; force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    print(" ".join(words))


if __name__ == "__main__":
    main()
