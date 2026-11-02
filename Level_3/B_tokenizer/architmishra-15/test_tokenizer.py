import unittest
from tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def test_basic_roundtrip(self):
        tokenizer = BPETokenizer()
        text = "hello world! hello BPE!"
        tokenizer.train(text, num_merges=10)
        encoded = tokenizer.encode(text)
        
        self.assertLess(len(encoded), len(text.encode('utf-8')))
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_tie_breaking(self):
        tokenizer = BPETokenizer()
        # 'ab' (97, 98) and 'ac' (97, 99) both appear twice.
        # 'ab' has a smaller tuple, so it should be merged first and get ID 256.
        tokenizer.train("ababacac", num_merges=1)
        self.assertIn((97, 98), tokenizer.merges)
        self.assertEqual(tokenizer.merges[(97, 98)], 256)

    def test_non_ascii_and_emojis(self):
        tokenizer = BPETokenizer()
        text = "Hello 👋, how are you? こんにちは世界"
        tokenizer.train(text, num_merges=5)
        encoded = tokenizer.encode(text)
        
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_empty_string(self):
        tokenizer = BPETokenizer()
        tokenizer.train("some text", num_merges=5)
        
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

if __name__ == '__main__':
    unittest.main()
