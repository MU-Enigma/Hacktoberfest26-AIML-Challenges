"""Comprehensive unit tests for BPETokenizer.

Tests cover:
- Roundtrip property (decode(encode(x)) == x) for various inputs (empty, ASCII, non-English, emojis)
- Sequential token ID assignment (256, 257, ...)
- Strict deterministic tie-breaking by smallest (first_id, second_id) tuple
- Training edge cases (empty text, single character, zero merges, excessive merges)
- Handling of unseen text during inference
- Compression efficiency on repetitive text
"""

import sys
from pathlib import Path
import unittest

# Ensure local folder is on sys.path regardless of working directory
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

from tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer()

    def test_roundtrip_empty_string(self):
        """Empty string must encode to empty list and decode back to empty string."""
        self.tokenizer.train("The quick brown fox", num_merges=5)
        encoded = self.tokenizer.encode("")
        self.assertEqual(encoded, [])
        self.assertEqual(self.tokenizer.decode(encoded), "")

    def test_roundtrip_ascii(self):
        """Standard ASCII text roundtrip property."""
        text = "Hello world! BPE tokenization works cleanly on English text."
        self.tokenizer.train(text, num_merges=15)
        encoded = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(encoded), text)

    def test_roundtrip_non_english_and_emojis(self):
        """Roundtrip must hold for non-English scripts, accents, and emojis."""
        test_strings = [
            "café crème brûlée",
            "नमस्ते भारत! आप कैसे हैं?",
            "日本語のテキスト: こんにちは世界 🌸",
            "Привет, мир! Это тест на кириллице.",
            "مرحبا بالعالم",
            "Emojis: 🚀 🤖 💻 🌟 🎉 🧠 🔥",
            "Mixed: Hello, 世界! नमस्ते 🌍 123 !@#$%",
        ]
        corpus = " ".join(test_strings)
        self.tokenizer.train(corpus, num_merges=25)

        for text in test_strings:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded)
                self.assertEqual(decoded, text)

    def test_token_ids_sequential(self):
        """Merged token IDs must start at 256 and increment sequentially."""
        text = "aa bb cc dd ee ff gg hh"
        num_merges = 5
        self.tokenizer.train(text, num_merges=num_merges)

        # Base tokens 0..255 exist
        for i in range(256):
            self.assertIn(i, self.tokenizer.vocab)
            self.assertEqual(self.tokenizer.vocab[i], bytes([i]))

        # Merges must receive IDs 256 through 260
        merged_ids = sorted(self.tokenizer.merges.values())
        expected_ids = list(range(256, 256 + len(self.tokenizer.merges)))
        self.assertEqual(merged_ids, expected_ids)

    def test_tie_breaking_by_first_id(self):
        """When frequencies tie, the pair with smaller first_id must be picked."""
        # 'ba ca'
        # Pairs:
        # ('b', 'a') -> (98, 97): count 1
        # ('a', ' ') -> (97, 32): count 1
        # (' ', 'c') -> (32, 99): count 1
        # ('c', 'a') -> (99, 97): count 1
        # All have frequency 1.
        # Smallest pair is (' ', 'c') = (32, 99) because 32 < 97, 98, 99.
        self.tokenizer.train("ba ca", num_merges=1)
        expected_pair = (ord(" "), ord("c"))
        self.assertIn(expected_pair, self.tokenizer.merges)
        self.assertEqual(self.tokenizer.merges[expected_pair], 256)

    def test_tie_breaking_by_second_id(self):
        """When frequencies tie and first_id is equal, pair with smaller second_id is picked."""
        # 'ab ac'
        # Pairs with count 1:
        # ('a', 'b') -> (97, 98)
        # ('b', ' ') -> (98, 32)
        # (' ', 'a') -> (32, 97)
        # ('a', 'c') -> (97, 99)
        # Smallest overall is (' ', 'a') = (32, 97).
        # Let's test with 'a b a c' with isolated tokens or specific pair comparison:
        # Instead, consider 'abab acac' where 'ab' has count 2, and 'ac' has count 2!
        # Pairs:
        # ('a', 'b') -> (97, 98): count 2
        # ('a', 'c') -> (97, 99): count 2
        # ('b', 'a') -> (98, 97): count 1
        # ('b', ' ') -> (98, 32): count 1
        # (' ', 'a') -> (32, 97): count 1
        # ('c', 'a') -> (99, 97): count 1
        # Max frequency is 2, tied between (97, 98) and (97, 99).
        # Both have first_id = 97.
        # Tie-breaker chooses (97, 98) because 98 < 99.
        self.tokenizer.train("abab acac", num_merges=1)
        pair_ab = (ord("a"), ord("b"))
        self.assertIn(pair_ab, self.tokenizer.merges)
        self.assertEqual(self.tokenizer.merges[pair_ab], 256)

    def test_training_edge_cases(self):
        """Edge cases such as empty text, single character, or zero merges."""
        # Empty text
        self.tokenizer.train("", num_merges=10)
        self.assertEqual(len(self.tokenizer.merges), 0)
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

        # Single character text
        self.tokenizer.train("x", num_merges=5)
        self.assertEqual(len(self.tokenizer.merges), 0)
        self.assertEqual(self.tokenizer.encode("x"), [ord("x")])
        self.assertEqual(self.tokenizer.decode([ord("x")]), "x")

        # num_merges = 0
        self.tokenizer.train("hello world", num_merges=0)
        self.assertEqual(len(self.tokenizer.merges), 0)
        self.assertEqual(self.tokenizer.encode("hello"), [ord(c) for c in "hello"])

        # num_merges much larger than possible merges
        self.tokenizer.train("ab", num_merges=100)
        # Only 1 merge is possible for 'ab'
        self.assertEqual(len(self.tokenizer.merges), 1)
        self.assertEqual(self.tokenizer.encode("ab"), [256])
        self.assertEqual(self.tokenizer.decode([256]), "ab")

    def test_unseen_text_inference(self):
        """Text containing unseen characters or words still roundtrips perfectly."""
        # Train on limited vocabulary
        self.tokenizer.train("apple banana orange", num_merges=5)
        unseen_text = "grape kiwi 123 !? 🚀"
        encoded = self.tokenizer.encode(unseen_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, unseen_text)

    def test_compression_on_repetitive_text(self):
        """Tokens length should be smaller than raw byte length after learning merges."""
        text = "banana " * 50
        raw_byte_length = len(text.encode("utf-8"))
        self.tokenizer.train(text, num_merges=10)
        encoded = self.tokenizer.encode(text)
        self.assertLess(len(encoded), raw_byte_length)
        self.assertEqual(self.tokenizer.decode(encoded), text)

    def test_unknown_token_id_raises_value_error(self):
        """Decoding an invalid/unknown token ID should raise ValueError."""
        self.tokenizer.train("abc", num_merges=1)
        with self.assertRaises(ValueError):
            self.tokenizer.decode([9999])


if __name__ == "__main__":
    unittest.main()
