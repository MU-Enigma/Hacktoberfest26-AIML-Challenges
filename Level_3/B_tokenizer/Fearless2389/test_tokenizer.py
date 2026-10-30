"""Tests for tokenizer.py. Run from this folder with:  python -m pytest -v"""

import pytest

from tokenizer import BPETokenizer

# A small multilingual training text used by several tests.
CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "नमस्ते दुनिया नमस्ते. నమస్కారం ప్రపంచం. 你好世界 你好. 🙂🙂 🚀 hello hello"
)


def trained(text=CORPUS, num_merges=50):
    """Helper: return a tokenizer trained on `text`."""
    tok = BPETokenizer()
    tok.train(text, num_merges)
    return tok


# ---------------------------------------------------------------- round trips

ROUNDTRIP_CASES = [
    "",                                   # empty string
    "hello world",                        # plain ASCII
    "नमस्ते दुनिया",                         # Hindi (3-byte UTF-8 chars)
    "నమస్కారం ప్రపంచం",                    # Telugu
    "你好世界",                             # Chinese
    "🙂🚀👨‍👩‍👧",                              # emoji, incl. a multi-codepoint family emoji
    "mixed: cat 猫 बिल्ली 🐱 ok\n\ttabs",    # everything at once
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip_after_training(text):
    tok = trained()
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip_untrained(text):
    # Before training, encode gives raw UTF-8 bytes and decode undoes it.
    tok = BPETokenizer()
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


def test_unseen_text_still_roundtrips():
    # Text with characters the tokenizer never saw in training still works,
    # because any byte 0..255 is always a valid token.
    tok = trained()
    text = "Zebras ¿qué? Ünïcødé ✓ 🦓 சென்னை"
    assert tok.decode(tok.encode(text)) == text


# ------------------------------------------------------------ ids and merges

def test_merge_ids_start_at_256_and_are_consecutive():
    tok = trained(num_merges=30)
    assert list(tok.merges.values()) == list(range(256, 256 + 30))
    # Every merged id has an entry in the vocab with at least 2 bytes.
    for new_id in tok.merges.values():
        assert len(tok.vocab[new_id]) >= 2


def test_hand_computed_merges_and_tie_breaking():
    # Text "cdcdabab". Bytes: a=97, b=98, c=99, d=100.
    # ids = [99, 100, 99, 100, 97, 98, 97, 98]
    #
    # Merge 1. Pair counts:
    #   (99,100)=2  (100,99)=1  (100,97)=1  (97,98)=2  (98,97)=1
    #   Tie between (97,98) "ab" and (99,100) "cd", both 2.
    #   (97,98) < (99,100) as tuples, so "ab" wins -> id 256,
    #   even though "cd" appears first in the text.
    #   ids = [99, 100, 99, 100, 256, 256]
    #
    # Merge 2. Pair counts:
    #   (99,100)=2  (100,99)=1  (100,256)=1  (256,256)=1
    #   (99,100) is the clear winner -> id 257.
    #   ids = [257, 257, 256, 256]
    #
    # Merge 3. Pair counts: (257,257)=1  (257,256)=1  (256,256)=1
    #   Three-way tie at count 1; smallest tuple is (256,256) -> id 258.
    #   ids = [257, 257, 258]
    #
    # Merge 4. Pair counts: (257,257)=1  (257,258)=1
    #   Tie; smallest is (257,257) -> id 259.   ids = [259, 258]
    #
    # Merge 5. Only pair (259,258) -> id 260.   ids = [260]
    #
    # Merge 6. One token left, no pairs: training stops early.
    tok = BPETokenizer()
    tok.train("cdcdabab", 10)
    assert tok.merges == {
        (97, 98): 256,
        (99, 100): 257,
        (256, 256): 258,
        (257, 257): 259,
        (259, 258): 260,
    }
    assert tok.encode("cdcdabab") == [260]
    assert tok.vocab[256] == b"ab"
    assert tok.vocab[260] == b"cdcdabab"


def test_overlapping_pairs():
    # "aaaa" = [97, 97, 97, 97]. The pair (97,97) is counted 3 times
    # (positions 0-1, 1-2, 2-3), but merging goes left to right without
    # reusing a token, giving [256, 256]. Merge 2 then joins those -> 257.
    tok = BPETokenizer()
    tok.train("aaaa", 5)
    assert tok.merges == {(97, 97): 256, (256, 256): 257}
    # Odd length: "aaa" -> [256, 97] (the last "a" has no partner left).
    assert tok.encode("aaa") == [256, 97]


def test_encode_applies_merges_in_learned_order():
    # Trained on "abababab": merge 1 is (97,98) "ab" -> 256,
    # merge 2 is (256,256) "abab" -> 257.
    tok = BPETokenizer()
    tok.train("abababab", 2)
    assert tok.merges == {(97, 98): 256, (256, 256): 257}
    assert tok.encode("abab") == [257]
    assert tok.encode("ababab") == [257, 256]
    assert tok.encode("ba") == [98, 97]


# ------------------------------------------------------------ edge cases

def test_zero_merges_gives_raw_bytes():
    tok = trained(num_merges=0)
    assert tok.merges == {}
    assert tok.encode("héllo") == list("héllo".encode("utf-8"))


@pytest.mark.parametrize("text", ["", "x"])
def test_tiny_training_text_does_not_crash(text):
    tok = BPETokenizer()
    tok.train(text, 10)
    assert tok.merges == {}
    assert tok.decode(tok.encode(text)) == text


def test_single_multibyte_char_can_still_merge():
    # "🙂" is ONE character but FOUR bytes, so there are pairs to merge.
    tok = BPETokenizer()
    tok.train("🙂", 10)
    assert len(tok.merges) == 3
    assert tok.encode("🙂") == [258]


def test_negative_num_merges_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", -1)


def test_retraining_resets_state():
    tok = trained()
    tok.train("aaaa", 1)
    assert tok.merges == {(97, 97): 256}
    assert max(tok.vocab) == 256


def test_encode_reduces_token_count_on_training_text():
    tok = trained()
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8"))


def test_training_is_deterministic():
    a, b = trained(), trained()
    assert a.merges == b.merges
    assert a.encode(CORPUS) == b.encode(CORPUS)


def test_decode_partial_character_does_not_crash():
    # 224 (0xE0) is only the first byte of a 3-byte character such as "न".
    # On its own it is not valid UTF-8, so decode shows U+FFFD instead of
    # raising an error.
    assert BPETokenizer().decode([224]) == "�"


def test_decode_unknown_id_raises():
    with pytest.raises(ValueError):
        BPETokenizer().decode([999])
