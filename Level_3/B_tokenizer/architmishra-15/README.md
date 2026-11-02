# Byte-Pair Encoding (BPE) Tokenizer

A from-scratch, byte-level BPE tokenizer implementation in Python. 

## How to run the tests

Ensure you have Python 3 installed. You can run the test suite using the standard `unittest` module:

```bash
python -m unittest test_tokenizer.py

```

## Example Run

```python
from tokenizer import BPETokenizer

# 1. Initialize
tokenizer = BPETokenizer()
text = "tok tok tokenizer!"

# 2. Train (Learn 3 merges)
tokenizer.train(text, num_merges=3)

# 3. Encode
encoded = tokenizer.encode(text)
print("Encoded IDs:", encoded)

# 4. Decode
decoded = tokenizer.decode(encoded)
print("Decoded Text:", decoded)

assert decoded == text
```


### PR Description Snippet

Here is a block you can paste directly into your Pull Request description regarding your design decisions:

**Design Decision: Byte-Level Encoding vs. Character-Level Encoding**
> "For this BPE implementation, I decided to build a **Byte-Level Tokenizer** by immediately encoding all input text into UTF-8 bytes before processing, rather than treating raw Unicode characters as the base tokens. 
> 
> **The alternative I rejected:** Character-level BPE, which involves finding every unique Unicode character in the training text and adding it to the base vocabulary. 
> 
> **Why I rejected it:** A character-level approach creates severe Out-Of-Vocabulary (OOV) risks during inference. If the `encode` method encounters a rare emoji or foreign character that wasn't present in the training set, it fails. Alternatively, initializing the vocabulary with *all* possible Unicode characters would result in a massively bloated base vocabulary of over 140,000 tokens. By operating at the byte level, the base vocabulary is strictly capped at exactly 256 tokens (representing all possible byte values 0-255). This guarantees that *any* string—regardless of the language, alphabet, or presence of emojis—can be fully tokenized and perfectly reconstructed without ever needing an `<UNK>` (unknown) token."

```
