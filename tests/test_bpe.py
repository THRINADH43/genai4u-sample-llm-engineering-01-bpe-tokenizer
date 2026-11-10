import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bpe import BPETokenizer, count_pairs, merge  # noqa: E402

TEXT = "low lower lowest newer wider new news " * 20 + "తెలుగు తెలుసు తెలుగులో " * 10


class HelperTests(unittest.TestCase):
    def test_count_pairs(self):
        self.assertEqual(count_pairs([1, 2, 1, 2]), {(1, 2): 2, (2, 1): 1})

    def test_merge_does_not_overlap(self):
        self.assertEqual(merge([5, 5, 5], (5, 5), 9), [9, 5])
        self.assertEqual(merge([1, 2, 3, 1, 2], (1, 2), 7), [7, 3, 7])


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer()
        cls.tok.train(TEXT, 320)

    def test_training_learns_merges(self):
        self.assertGreater(len(self.tok.merges), 10)
        self.assertIn(b"low", self.tok.vocab.values())

    def test_round_trip_english_and_telugu(self):
        for text in ["lowest newer", "తెలుగు", "unseen words like zebra!", "హిందీ: नमस्ते", ""]:
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_merges_make_text_shorter(self):
        text = "lower newest"
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")))

    def test_special_token_is_one_id_and_round_trips(self):
        ids = self.tok.encode("new<|endoftext|>తెలుగు")
        self.assertIn(self.tok.special_tokens["<|endoftext|>"], ids)
        self.assertEqual(self.tok.decode(ids), "new<|endoftext|>తెలుగు")

    def test_emoji_and_spaces(self):
        text = "  two  spaces 🙂\n"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_unknown_id(self):
        with self.assertRaises(ValueError):
            self.tok.decode([99999])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            self.tok.save(path)
            again = BPETokenizer.load(path)
            self.assertEqual(again.encode("lowest తెలుగు"), self.tok.encode("lowest తెలుగు"))

    def test_small_vocab_rejected(self):
        with self.assertRaises(ValueError):
            BPETokenizer().train("abc", 100)


if __name__ == "__main__":
    unittest.main()
