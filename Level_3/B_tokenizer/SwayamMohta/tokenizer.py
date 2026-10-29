"""Byte-Pair-Encoding (BPE) Tokenizer implementation for Hacktoberfest 2026 Level 3 Option B.

Implements byte-level BPE with deterministic tie-breaking.
"""

from typing import Dict, List, Tuple


class BPETokenizer:
    """A byte-level Byte-Pair-Encoding (BPE) tokenizer.

    Base vocabulary consists of the 256 individual byte values (0-255).
    New merged tokens are assigned sequential IDs starting at 256.
    """

    def __init__(self) -> None:
        # Maps an adjacent pair of token IDs (p0, p1) to the new merged token ID
        self.merges: Dict[Tuple[int, int], int] = {}
        # Maps token ID to its underlying bytes representation
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Counts frequency of all adjacent token pairs."""
        stats: Dict[Tuple[int, int], int] = {}
        for pair in zip(tokens, tokens[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replaces non-overlapping occurrences of `pair` with `new_id` from left to right."""
        new_tokens: List[int] = []
        i = 0
        n = len(tokens)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str, num_merges: int) -> None:
        """Trains BPE merges on the input text.

        Args:
            text: Input training text.
            num_merges: Number of merge operations to learn.
        """
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        if num_merges <= 0:
            return

        tokens = list(text.encode("utf-8"))
        if len(tokens) < 2:
            return

        for _ in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break

            # Find the most frequent adjacent pair.
            # Tie-breaking: pick the pair that is smallest as a (first_id, second_id) tuple.
            max_count = max(stats.values())
            candidates = [pair for pair, count in stats.items() if count == max_count]
            best_pair = min(candidates)

            new_id = 256 + len(self.merges)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            tokens = self._merge(tokens, best_pair, new_id)

            if len(tokens) < 2:
                break

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs using learned merges.

        Args:
            text: String to tokenize.

        Returns:
            List of integer token IDs.
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            # Find pairs that are in learned merges
            pairs_in_merges = [pair for pair in stats if pair in self.merges]
            if not pairs_in_merges:
                break

            # Pick the pair with the lowest merge ID (learned earliest in training)
            pair_to_merge = min(pairs_in_merges, key=lambda p: self.merges[p])
            new_id = self.merges[pair_to_merge]
            tokens = self._merge(tokens, pair_to_merge, new_id)

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token id: {token_id}")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys

    # Ensure UTF-8 output on Windows consoles if needed
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    sample_corpus = (
        "Byte-pair encoding is an algorithm originally used for data compression. "
        "In modern NLP and LLMs, BPE enables open-vocabulary subword tokenization. "
        "It handles English, हिन्दी, 日本語, and emojis like 🤖 🚀 seamlessly!"
    )
    tokenizer = BPETokenizer()
    tokenizer.train(sample_corpus, num_merges=30)

    test_sentence = "BPE handles हिन्दी and 🚀 emojis seamlessly!"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_sentence}")
    print(f"Encoded IDs ({len(encoded)} tokens): {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Roundtrip match: {test_sentence == decoded}")
