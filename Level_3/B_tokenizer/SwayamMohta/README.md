# Level 3, Option B: Byte-Pair-Encoding (BPE) Tokenizer

A byte-level Byte-Pair-Encoding (BPE) tokenizer implemented in pure Python for **Hacktoberfest 2026 Level 3 Option B**.

## Overview

The tokenizer converts raw text into numerical token IDs and reconstructs text losslessly.

Key properties:
- **Byte-level base vocabulary:** Base tokens are the 256 byte values (`0` to `255`), guaranteeing that any UTF-8 text (English, non-English scripts, emojis, accents) can be represented without out-of-vocabulary (`<UNK>`) tokens.
- **Deterministic sequential IDs:** Each merged token is assigned the next sequential integer ID starting at `256` (`256`, `257`, `258`, ...).
- **Strict tie-breaking:** When multiple adjacent token pairs share the same highest frequency, the pair that is smallest as a `(first_id, second_id)` tuple is chosen.
- **Lossless roundtrip:** `decode(encode(x)) == x` holds for all inputs, including empty strings, arbitrary non-English text, and emojis.

## Project Structure

```
Level_3/B_tokenizer/SwayamMohta/
├── tokenizer.py        # BPETokenizer implementation with train, encode, and decode
├── test_tokenizer.py   # Comprehensive unit tests
└── README.md           # Documentation, run instructions, and design decisions
```

## How to Run

### 1. Run the Example Demonstration

Run `tokenizer.py` directly to see BPE training, encoding, decoding, and roundtrip verification:

```bash
python tokenizer.py
```

### 2. Run the Unit Tests

Using Python's built-in `unittest`:

```bash
python -m unittest test_tokenizer.py -v
```

Or using `pytest`:

```bash
pytest test_tokenizer.py -v
```

## Example Run

Output from running `python tokenizer.py`:

```
Original: BPE handles हिन्दी and 🚀 emojis seamlessly!
Encoded IDs (39 tokens): [66, 80, 69, 32, 104, 97, 110, 100, 282, 261, 185, 261, 191, 261, 168, 280, 141, 261, 166, 280, 128, 283, 240, 159, 154, 128, 32, 101, 277, 106, 263, 267, 101, 97, 109, 264, 115, 276, 33]
Decoded:  BPE handles हिन्दी and 🚀 emojis seamlessly!
Roundtrip match: True
```

## Example Python Usage

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()

# Train merges
corpus = "low lower newest widest lowest"
tokenizer.train(corpus, num_merges=10)

# Encode
text = "lowest"
ids = tokenizer.encode(text)
print("Encoded IDs:", ids)

# Decode
reconstructed = tokenizer.decode(ids)
print("Decoded text:", reconstructed)
assert reconstructed == text
```

## Design Decision (for PR Description)

- **Question:** What did you do about text that contains characters outside plain ASCII, and why?
- **Decision:** Operated directly at the **UTF-8 byte level**, using the 256 individual byte values (`0` to `255`) as the immutable base vocabulary. Non-ASCII characters (e.g. Hindi, Japanese, accents, emojis) naturally decompose into their standard UTF-8 multi-byte sequences. Merges compress frequent byte pairs into higher-level subword tokens without needing custom character tables.
- **Alternative rejected:** Character-level / Unicode code-point tokenization with an `<UNK>` (unknown) token for out-of-vocabulary characters.
- **Why rejected:** Unicode encompasses over 140,000 code points. A character-level vocabulary either explodes in size or must discard rare characters as `<UNK>`, violating lossless reconstruction (`decode(encode(x)) == x`). Byte-level BPE keeps the base vocabulary fixed and compact at exactly 256 entries while guaranteeing 100% lossless encoding and decoding for any text across every language and script.
