"""A small byte-level byte-pair-encoding (BPE) tokenizer.

How it works, in one paragraph:
    Text is first turned into UTF-8 bytes, so every possible string becomes a
    list of numbers between 0 and 255. Those 256 byte values are the starting
    tokens (ids 0..255). Training then repeatedly finds the most frequent pair
    of neighbouring tokens and replaces every occurrence of that pair with one
    new token. The first new token gets id 256, the next 257, and so on.
    Encoding replays those learned merges on new text; decoding turns ids back
    into bytes and the bytes back into text.

Rules taken from the challenge spec:
    * Base tokens are the 256 byte values (ids 0..255).
    * Merged tokens get ids 256, 257, ... in the order the merges are made.
    * Always merge the most frequent adjacent pair. On a tie, merge the pair
      that is smallest as a (first_id, second_id) tuple.
    * decode(encode(x)) == x for every string, including "" and non-English.

Example:
    >>> tok = BPETokenizer()
    >>> tok.train("abababab", 2)
    >>> tok.encode("abab")
    [257]
    >>> tok.decode([257])
    'abab'
"""

from collections import Counter


def get_pair_counts(ids):
    """Count how often each adjacent pair of ids appears in `ids`.

    Pairs are counted at every position, so they can overlap:
    [97, 97, 97] ("aaa") gives {(97, 97): 2}.
    """
    # zip(ids, ids[1:]) walks the list two at a time: (ids[0], ids[1]),
    # (ids[1], ids[2]), ... Counter tallies how many times each pair shows up.
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    """Return a new list where every occurrence of `pair` is replaced by `new_id`.

    The scan goes left to right and never reuses a token once it has been
    merged, so [97, 97, 97] with pair (97, 97) becomes [new_id, 97].
    """
    result = []
    i = 0
    while i < len(ids):
        # If the pair starts here (and there is a next element), merge it
        # and jump over both tokens.
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            # Otherwise keep this token as it is and move one step right.
            result.append(ids[i])
            i += 1
    return result


class BPETokenizer:
    """Byte-level BPE tokenizer with train, encode and decode."""

    def __init__(self):
        # merges maps a pair of ids to the id of the token that replaces it,
        # e.g. {(97, 98): 256}. Python dicts remember insertion order, so this
        # is also the order in which the merges were learned.
        self.merges = {}
        # vocab maps every id to the bytes it stands for. Ids 0..255 are the
        # single bytes themselves; merged ids are filled in by train().
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, num_merges):
        """Learn up to `num_merges` merges from `text`.

        Calling train again starts from scratch (old merges are forgotten).
        Training stops early if fewer than two tokens are left, because then
        there is no pair to merge. A pair is merged even if it appears only
        once, as long as it is the best pair available.
        """
        if num_merges < 0:
            raise ValueError("num_merges must be 0 or more")

        # Reset any state from a previous call to train().
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Turn the text into its UTF-8 bytes, then into a list of ints 0..255.
        ids = list(text.encode("utf-8"))

        for step in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                # Fewer than two tokens left: nothing to merge.
                break

            # Pick the most frequent pair. The key sorts by (-count, pair), so
            # a higher count wins first, and on a tie the smaller
            # (first_id, second_id) tuple wins. Using min() makes it
            # deterministic: it never depends on dictionary order.
            best = min(counts, key=lambda p: (-counts[p], p))

            new_id = 256 + step
            ids = merge(ids, best, new_id)
            self.merges[best] = new_id
            # The new token's bytes are just its two parts glued together.
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

    def encode(self, text):
        """Turn `text` into a list of token ids using the learned merges.

        Merges are applied in the order they were learned: at each step we
        find the pair in the current list that was learned earliest (lowest
        merge id) and merge it everywhere. Before training this simply
        returns the raw UTF-8 bytes.
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # Among the pairs present, find the one learned first. Pairs that
            # were never learned get "infinity" so they are never chosen.
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                # No pair in the list is a learned merge, so we are done.
                break
            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids):
        """Turn a list of token ids back into a string.

        Each id is looked up to get its bytes, the bytes are joined, and the
        result is decoded as UTF-8. Ids that came from encode() always form
        valid UTF-8. A hand-written list could cut a multi-byte character in
        half (e.g. [224] is only the first byte of a Hindi letter), so we use
        errors="replace" to show such bytes as the replacement character
        (U+FFFD) instead of crashing.
        """
        try:
            data = b"".join(self.vocab[i] for i in ids)
        except KeyError as err:
            raise ValueError(f"unknown token id: {err.args[0]}") from None
        return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # A tiny demo: python tokenizer.py
    sample = "the cat sat on the mat. the cat ate the rat. नमस्ते नमस्ते 🙂🙂"
    tok = BPETokenizer()
    tok.train(sample, 20)

    ids = tok.encode(sample)
    print("text bytes :", len(sample.encode("utf-8")))
    print("tokens     :", len(ids))
    print("first merges:")
    for pair, new_id in list(tok.merges.items())[:5]:
        print(f"  {pair} -> {new_id}  {tok.vocab[new_id]!r}")
    print("ids        :", ids)
    print("round trip :", tok.decode(ids) == sample)
    print("unseen     :", tok.encode("the bat"), "->", tok.decode(tok.encode("the bat")))
