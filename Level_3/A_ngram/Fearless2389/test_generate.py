"""Tests for generate.py. Run from this folder with:  python -m pytest -q"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "generate.py"

# Load generate.py by its file path under a unique name. A plain
# "import generate" could pick up another contributor's generate.py
# if pytest is run over the whole Level_3 folder at once.
_spec = importlib.util.spec_from_file_location("fearless2389_generate", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

CORPUS = (HERE / "sample.txt").read_text(encoding="utf-8")
WORDS = gen.tokenize(CORPUS)


def run_cli(*args):
    """Run generate.py as a real command and return the finished process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8",
    )


# ---------- the model ----------

def test_build_model_counts_followers():
    model = gen.build_model(["a", "b", "a", "c"], n=1)
    assert model["counts"][()] == {"a": 2, "b": 1, "c": 1}  # unigram counts
    assert model["counts"][("a",)] == {"b": 1, "c": 1}
    assert ("c",) not in model["counts"]  # "c" is last, nothing follows it


def test_build_model_rejects_n_below_1():
    with pytest.raises(ValueError):
        gen.build_model(["a"], n=0)


# ---------- generate() ----------

@pytest.mark.parametrize("length", [1, 7, 50, 300])
def test_exact_word_count(length):
    model = gen.build_model(WORDS, n=2)
    assert len(gen.generate(model, length, seed=42)) == length


def test_length_zero_gives_no_words():
    assert gen.generate(gen.build_model(WORDS, n=2), 0, seed=1) == []


def test_same_seed_same_output():
    model = gen.build_model(WORDS, n=2)
    assert gen.generate(model, 60, seed=42) == gen.generate(model, 60, seed=42)


def test_different_seeds_usually_differ():
    model = gen.build_model(WORDS, n=1)
    outputs = {tuple(gen.generate(model, 30, seed=s)) for s in range(5)}
    assert len(outputs) > 1


def test_every_word_comes_from_the_corpus():
    model = gen.build_model(WORDS, n=3)
    assert set(gen.generate(model, 500, seed=7)) <= set(WORDS)


def test_n1_follows_the_only_possible_next_word():
    # In "x y x y x y", "x" is always followed by "y" and "y" by "x",
    # so after the first word the output must alternate.
    model = gen.build_model("x y x y x y".split(), n=1)
    out = gen.generate(model, 20, seed=3)
    for previous, current in zip(out, out[1:]):
        assert current != previous


# ---------- backoff ----------

def test_backoff_drops_unseen_old_words():
    model = gen.build_model("the cat sat on the mat".split(), n=2)
    # ("dog", "the") never appeared, but ("the",) did.
    assert gen.find_context(model, ["dog", "the"]) == ("the",)
    # A seen 2-word context is used as is, no backoff needed.
    assert gen.find_context(model, ["sat", "on"]) == ("sat", "on")


def test_backoff_reaches_unigrams_at_a_dead_end():
    model = gen.build_model("the cat sat".split(), n=2)
    # "sat" is the last word: nothing ever followed it, so fall back to ().
    assert gen.find_context(model, ["cat", "sat"]) == ()


def test_generation_continues_past_the_dead_end_word():
    model = gen.build_model("one two three".split(), n=2)
    assert len(gen.generate(model, 40, seed=0)) == 40


# ---------- short and empty corpora ----------

@pytest.mark.parametrize("text", ["hello", "hello world", "a b c"])
def test_corpus_shorter_than_n(text):
    model = gen.build_model(text.split(), n=5)
    out = gen.generate(model, 10, seed=1)
    assert len(out) == 10 and set(out) <= set(text.split())


def test_empty_corpus_raises_clear_error():
    with pytest.raises(ValueError):
        gen.generate(gen.build_model([], n=2), 5, seed=1)


# ---------- the command line ----------

def test_cli_prints_exactly_length_words():
    result = run_cli("--n", "2", "--seed", "42", "--length", "50", "--input", str(HERE / "sample.txt"))
    assert result.returncode == 0
    assert len(result.stdout.split()) == 50


def test_cli_is_reproducible():
    args = ("--n", "2", "--seed", "42", "--length", "25", "--input", str(HERE / "sample.txt"))
    assert run_cli(*args).stdout == run_cli(*args).stdout


def test_cli_short_and_unicode_corpus(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("café naïve 日本語", encoding="utf-8")
    result = run_cli("--n", "10", "--seed", "1", "--length", "8", "--input", str(corpus))
    assert result.returncode == 0
    assert len(result.stdout.split()) == 8
    assert set(result.stdout.split()) <= {"café", "naïve", "日本語"}


def test_cli_length_zero_prints_no_words():
    result = run_cli("--n", "2", "--seed", "1", "--length", "0", "--input", str(HERE / "sample.txt"))
    assert result.returncode == 0 and result.stdout.split() == []


def test_cli_empty_file_fails_cleanly(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("   \n  ", encoding="utf-8")
    result = run_cli("--n", "2", "--seed", "1", "--length", "5", "--input", str(corpus))
    assert result.returncode == 1
    assert "no words" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("bad", [["--n", "0"], ["--length", "-1"], ["--n", "two"]])
def test_cli_rejects_bad_arguments(bad):
    result = run_cli(*bad, "--input", str(HERE / "sample.txt"))
    assert result.returncode == 2 and "Traceback" not in result.stderr


def test_cli_missing_file_fails_cleanly(tmp_path):
    result = run_cli("--input", str(tmp_path / "missing.txt"))
    assert result.returncode == 2 and "Traceback" not in result.stderr
