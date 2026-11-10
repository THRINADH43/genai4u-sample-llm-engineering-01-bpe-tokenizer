"""Train on English + Telugu text and report tokens per character for each language.

python compare.py [vocab_size]
"""

import sys

from bpe import BPETokenizer


def tokens_per_char(tok, text):
    return len(tok.encode(text)) / len(text)


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    english = open("data/english.txt", encoding="utf-8").read()
    telugu = open("data/telugu.txt", encoding="utf-8").read()
    tok = BPETokenizer()
    tok.train(english + "\n" + telugu, vocab_size)
    tok.save("tokenizer.json")
    print(f"vocabulary: {len(tok.vocab)} tokens ({len(tok.merges)} merges) + {len(tok.special_tokens)} special")
    print(f"{'text':<10}{'chars':>8}{'bytes':>8}{'tokens':>8}{'tokens/char':>13}")
    for name, text in (("English", english), ("Telugu", telugu)):
        n = len(tok.encode(text))
        print(f"{name:<10}{len(text):>8}{len(text.encode()):>8}{n:>8}{n / len(text):>13.2f}")
        assert tok.decode(tok.encode(text)) == text
    sample = "సంత<|endoftext|>market"
    print("round trip with a special token:", tok.decode(tok.encode(sample)) == sample)


if __name__ == "__main__":
    main()
