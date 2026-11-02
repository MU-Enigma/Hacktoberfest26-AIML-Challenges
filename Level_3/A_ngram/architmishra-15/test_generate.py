import unittest
from generate import NGramGenerator

class TestNGramGenerator(unittest.TestCase):
    def setUp(self):
        self.corpus = "the quick brown fox jumps over the lazy dog".split()

    def test_exact_length(self):
        generator = NGramGenerator(n=2)
        generator.train(self.corpus)
        result = generator.generate(length=50, seed=42)
        self.assertEqual(len(result), 50)

    def test_reproducibility(self):
        generator = NGramGenerator(n=3)
        generator.train(self.corpus)
        
        result1 = generator.generate(length=20, seed=123)
        result2 = generator.generate(length=20, seed=123)
        result3 = generator.generate(length=20, seed=999)
        
        self.assertEqual(result1, result2)
        self.assertNotEqual(result1, result3)

    def test_backoff_and_dead_ends(self):
        # "dog" is at the end of the text. It has no subsequent transitions.
        generator = NGramGenerator(n=3)
        generator.train(self.corpus)
        
        # Generating 20 words guarantees we hit "dog" and must backoff.
        # It shouldn't crash; it should backoff to k=0 and pick a random word.
        result = generator.generate(length=20, seed=42)
        self.assertEqual(len(result), 20)

    def test_short_corpus_no_crash(self):
        # The corpus is much shorter than n.
        short_corpus = ["hello", "world"]
        generator = NGramGenerator(n=10)
        generator.train(short_corpus)
        
        result = generator.generate(length=5, seed=1)
        self.assertEqual(len(result), 5)
        
        # Unigram fallback will just bounce between "hello" and "world"
        for word in result:
            self.assertIn(word, short_corpus)

    def test_empty_corpus(self):
        generator = NGramGenerator(n=2)
        generator.train([])
        result = generator.generate(length=10, seed=42)
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
