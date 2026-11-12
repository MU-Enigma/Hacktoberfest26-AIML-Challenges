import unittest
from tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BPETokenizer()

    def test_initial_vocabulary_size(self):
        """Tokenizer should start with exactly 256 byte tokens."""
        self.assertEqual(len(self.tokenizer.vocab), 256)

    def test_initial_byte_ids(self):
        """Each byte ID should map to its corresponding byte."""
        for i in range(256):
            self.assertEqual(self.tokenizer.vocab[i], bytes([i]))

    def test_next_id_starts_at_256(self):
        self.assertEqual(self.tokenizer.next_id, 256)

    def test_training_creates_merges(self):
        """Training on repeated text should learn at least one merge."""
        self.tokenizer.train("banana banana banana", 5)

        self.assertGreater(len(self.tokenizer.merges), 0)

    def test_merge_ids_are_sequential(self):
        """New token IDs should be assigned sequentially from 256."""
        self.tokenizer.train("banana banana banana", 5)

        for i, merge in enumerate(self.tokenizer.merges):
            first, second, new_id = merge
            self.assertEqual(new_id, 256 + i)

    def test_new_token_represents_concatenated_bytes(self):
        """A merged token should contain the bytes of its components."""
        self.tokenizer.train("abababab", 3)

        for first, second, new_id in self.tokenizer.merges:
            expected = (
                self.tokenizer.vocab[first]
                + self.tokenizer.vocab[second]
            )

            self.assertEqual(
                self.tokenizer.vocab[new_id],
                expected
            )

    def test_tie_breaking(self):
        """
        If frequencies tie, choose the lexicographically
        smallest (first_id, second_id) pair.

        In 'ab cd', all adjacent pairs occur once.
        (32, 99) is the smallest pair because space = 32.
        """
        self.tokenizer.train("ab cd", 1)

        self.assertEqual(
            self.tokenizer.merges[0],
            (32, 99, 256)
        )

    def test_overlapping_pair_merge(self):
        """Overlapping occurrences should be merged left to right."""
        tokens = [97, 97, 97]

        result = self.tokenizer.merge_pair(
            tokens,
            (97, 97),
            256
        )

        self.assertEqual(result, [256, 97])

    def test_zero_merges(self):
        """Training with zero merges should preserve base vocabulary."""
        self.tokenizer.train("hello", 0)

        self.assertEqual(len(self.tokenizer.merges), 0)

        self.assertEqual(
            self.tokenizer.encode("hello"),
            list("hello".encode("utf-8"))
        )

    def test_no_pairs_available(self):
        """Training should stop when no adjacent pairs exist."""
        self.tokenizer.train("a", 10)

        self.assertEqual(len(self.tokenizer.merges), 0)

    def test_negative_merges(self):
        """Negative merge counts should raise an error."""
        with self.assertRaises(ValueError):
            self.tokenizer.train("hello", -1)

    def test_basic_round_trip(self):
        text = "hello world"

        self.tokenizer.train(text, 10)

        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertEqual(decoded, text)

    def test_empty_string(self):
        self.tokenizer.train("hello", 5)

        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_repeated_text(self):
        text = "banana banana banana"

        self.tokenizer.train(text, 10)

        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text
        )

    def test_hindi_text(self):
        text = "नमस्ते दुनिया"

        self.tokenizer.train(text, 20)

        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text
        )

    def test_emoji(self):
        text = "🚀🔥😀"

        self.tokenizer.train(text, 20)

        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text
        )

    def test_mixed_unicode(self):
        text = "Hello नमस्ते 🚀 123"

        self.tokenizer.train(text, 30)

        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text
        )

    def test_no_merges_encoding(self):
        """Without learned merges, encoding should return UTF-8 bytes."""
        text = "hello"

        self.tokenizer.train("something else", 0)

        expected = list(text.encode("utf-8"))

        self.assertEqual(self.tokenizer.encode(text), expected)

    def test_unknown_token_id(self):
        """Decoding an unknown token ID should raise ValueError."""
        with self.assertRaises(ValueError):
            self.tokenizer.decode([99999])

    def test_training_is_deterministic(self):
        """Same corpus and merge count should produce same merges."""
        tokenizer1 = BPETokenizer()
        tokenizer2 = BPETokenizer()

        text = "banana banana apple apple"

        tokenizer1.train(text, 10)
        tokenizer2.train(text, 10)

        self.assertEqual(
            tokenizer1.merges,
            tokenizer2.merges
        )

    def test_retraining_resets_vocabulary_and_merges(self):
        """A new training call should reset the previous state."""
        self.tokenizer.train("banana banana", 5)

        self.tokenizer.train("hello hello", 3)

        self.assertEqual(
            self.tokenizer.next_id,
            256 + len(self.tokenizer.merges)
        )

        self.assertEqual(len(self.tokenizer.vocab), 256 + len(self.tokenizer.merges))

        for i in range(256):
            self.assertEqual(self.tokenizer.vocab[i], bytes([i]))

    def test_round_trip_multiple_strings(self):
        """Any valid Python Unicode string should survive a round trip."""
        training_text = "hello banana नमस्ते 🚀 programming"

        self.tokenizer.train(training_text, 30)

        test_strings = [
            "",
            "hello",
            "banana",
            "hello world",
            "नमस्ते",
            "🚀",
            "Hello नमस्ते 🚀",
            "1234567890",
            "a" * 100,
            "The quick brown fox jumps over the lazy dog."
        ]

        for text in test_strings:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded)

                self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
