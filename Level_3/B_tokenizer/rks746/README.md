# Byte-Level BPE Tokenizer

## Overview

This project implements a Byte-Pair Encoding (BPE) tokenizer from scratch using Python.

The tokenizer learns frequently occurring adjacent byte sequences from a training corpus and merges them into larger tokens. Each token is assigned a unique integer ID.

The implementation supports UTF-8 text, including non-English languages, emojis, and other Unicode characters.

## How BPE Works

The tokenizer begins with 256 base tokens, representing all possible byte values (0–255).

It then performs the following steps:

1. Convert the training text into UTF-8 bytes.
2. Represent each byte using its corresponding token ID.
3. Count the frequency of every adjacent token pair.
4. Select the most frequent pair.
5. If multiple pairs have the same frequency, select the lexicographically smallest `(first_id, second_id)` pair.
6. Merge the selected pair into a new token.
7. Assign the new token the next available ID, starting from 256.
8. Repeat until the specified number of merges has been completed or no adjacent pairs remain.

### Example

Consider the training corpus:

```text
abababab
```

Initially, the byte sequence is:

```text
[97, 98, 97, 98, 97, 98, 97, 98]
```

The pair `(97, 98)` occurs four times.

The tokenizer merges this pair:

```text
(97, 98) -> 256
```

The resulting sequence becomes:

```text
[256, 256, 256, 256]
```

The new token represents the byte sequence `b'ab'`.

Further merges can combine these larger tokens into even larger sequences.

## Implementation Details

### 1. Vocabulary

The vocabulary is stored as a dictionary:

```python
vocab[token_id] = token_bytes
```

Initially, it contains 256 entries.

For example:

```python
vocab[65] = b'A'
vocab[97] = b'a'
```

Every new merged token is assigned the next available ID.

### 2. Pair Frequency Counting

The `get_pair_counts()` helper function uses `Counter` to count adjacent token pairs.

For example:

```python
tokens = [10, 20, 10, 20, 10]

# Pair frequencies:
(10, 20): 2
(20, 10): 2
```

### 3. Pair Selection and Tie-Breaking

The implementation selects the pair with the highest frequency.

When frequencies are equal, it selects the smallest pair according to lexicographic ordering of the two token IDs.

This ensures deterministic training results.

### 4. Merging

The `merge_pair()` helper function replaces non-overlapping occurrences of the selected pair from left to right.

Each merge creates a new vocabulary entry whose byte representation is the concatenation of the two component tokens.

### 5. Encoding

The `encode()` method:

1. Converts the input string into UTF-8 bytes.
2. Converts the bytes into their corresponding base token IDs.
3. Replays the learned merge operations in their original training order.
4. Returns the resulting sequence of token IDs.

The merge operations are not relearned during encoding.

### 6. Decoding

The `decode()` method:

1. Looks up the byte representation of each token ID.
2. Concatenates all the byte sequences.
3. Decodes the reconstructed byte sequence using UTF-8.

Concatenating bytes before UTF-8 decoding is important because an individual token may contain only part of a multibyte Unicode character.

## Handling Non-ASCII Text

### Design Decision

The implementation uses byte-level tokenization instead of character-level tokenization.

Python strings are encoded using UTF-8 before tokenization.

For example:

```python
text = "🚀"

print(list(text.encode("utf-8")))
# [240, 159, 154, 128]
```

A single emoji is represented by four bytes.

### Alternative Considered

Character-level tokenization was an alternative.

However, it would require handling a potentially large and variable character vocabulary. It would also not satisfy the assignment's requirement of exactly 256 initial byte tokens.

Byte-level tokenization provides a fixed initial vocabulary and supports arbitrary valid Unicode strings without requiring a separate vocabulary entry for every character.

### Checking Round-Trip Correctness

```python
text = "Hello नमस्ते 🚀"

tokenizer.train(text, num_merges=20)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

assert decoded == text
```

## Running Tests

Run the following command from the project directory:

```bash
python -m unittest -v test_tokenizer.py
```

## Correctness Guarantee

For every valid Python Unicode string `x`, the tokenizer should satisfy:

```python
decode(encode(x)) == x
```

This is achieved by preserving the original UTF-8 byte sequence during encoding and reconstructing it before decoding.

## Limitations

- This is a basic BPE implementation intended for learning and experimentation.
- The entire training corpus is processed as one sequence.
- It does not implement pre-tokenization, special tokens, or normalization.
- It does not implement the performance optimizations used by production tokenizers.
- The learned token IDs are identifiers, not semantic embeddings.

## Conclusion

This project demonstrates the core BPE algorithm: starting from a fixed byte vocabulary, learning frequent adjacent pairs, merging them into larger tokens, and using the learned merge sequence to tokenize previously unseen text.

The result is a reversible byte-level tokenizer that supports Unicode text and follows deterministic merge-selection rules.
