from collections import Counter

class BPETokenizer:

    def __init__(self):
        self.vocab = {
            i: bytes([i]) for i in range(256)
        }
        self.merges = []
        self.next_id = 256

    def get_pair_counts(self, tokens):
        pairs = Counter()

        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1

        return pairs

    def merge_pair(self, tokens, pair, new_id):
        result = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    def train(self, text, num_merges):
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")

        self.vocab = {
            i: bytes([i]) for i in range(256)
        }
        self.merges = []
        self.next_id = 256

        tokens = list(text.encode("utf-8"))

        for _ in range(num_merges):

            pair_counts = self.get_pair_counts(tokens)

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda p: (-pair_counts[p], p)
            )

            new_id = self.next_id

            tokens = self.merge_pair(
                tokens, best_pair, new_id
            )

            self.vocab[new_id] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

            self.merges.append(
                (best_pair[0], best_pair[1], new_id)
            )

            self.next_id += 1

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        for first, second, new_id in self.merges:
            tokens = self.merge_pair(
                tokens, (first, second), new_id
            )

        return tokens

    def decode(self, ids):
        output = b""

        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")

            output += self.vocab[token_id]

        return output.decode("utf-8")
