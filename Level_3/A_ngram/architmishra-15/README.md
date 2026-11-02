# N-gram Text Generator

A command-line tool that learns word transitions from an input text file and generates new text using an $N$-gram model with automatic context backoff.

## How to Run

Ensure you have Python 3 installed. 

Create a sample text file to use as input:
```bash
echo "the quick brown fox jumps over the lazy dog and the quick brown cat jumps over the lazy mouse" > text.txt

```

Run the generator from the command line:

```bash
python generate.py --n 3 --seed 42 --length 15 --input text.txt

```

Run the test suite:

```bash
python -m unittest test_generate.py

```

## Example Run

```bash
$ python generate.py --n 3 --seed 123 --length 15 --input text.txt
lazy mouse dog and the quick brown fox jumps over the lazy dog and the

```

**Design Decision: The "$k=0$" Universal Backoff State**

> **What I did:** To handle the specific edge case of a word that never appears mid-sentence (meaning it only appears at the very end of the corpus), I structured the model's context dictionary to include a `0`-length context (unigram backoff). When the generator hits a dead end—like the final word of the text—it will iteratively back off to shorter contexts until it reaches length 0. The $k=0$ context contains a flat list of every word in the training data, so it seamlessly selects a random word from the corpus to begin a new sequence. 
> 
> **The alternative I rejected:** I considered artificially modifying the corpus to be "circular" by appending the first word of the text to the end of the file during training, effectively ensuring no dead ends. I rejected this because it introduces false semantic transitions (e.g., tying the conclusion of a document directly back to its title) which corrupts the integrity of the n-gram probabilities. I also rejected introducing explicit `<EOF>` termination tokens, as the strict assignment spec requires exactly `--length` words to be printed, meaning early termination was not permitted.

```
