# Byte-pair encoding tokenizer from scratch

A BPE tokenizer in plain Python (standard library only, no tokenizer library). It trains a
vocabulary from text files, encodes text to token ids, decodes ids back to the same text,
keeps special tokens whole, and compares English with Telugu.

No API key is needed and an ordinary laptop is enough.

## Run

```
python compare.py        # trains on data/, prints the comparison, saves tokenizer.json
python -m unittest discover -s tests
```

## Training text

`data/english.txt` and `data/telugu.txt` are short paragraphs I wrote myself (about a village
market, learning to program, and how tokenizers work), so I am free to use them. The Telugu
file says the same things as the first part of the English file.

## How it works

1. Text is turned into UTF-8 bytes, so the starting vocabulary is the 256 byte values. Any
   text can be encoded, even characters never seen in training.
2. Training splits the text into chunks at spaces (so merges never join two words), counts
   every pair of neighbouring tokens (`count_pairs`), and merges the most common pair into a
   new token (`merge`). Ties go to the smaller pair, so training gives the same result every
   time. It stops at the requested vocabulary size, or when no pair appears twice.
3. Encoding applies the learned merges in the order they were learned.
4. Decoding joins the bytes of each token and turns them back into text.
5. Special tokens (`<|endoftext|>`) are split out before encoding and get their own id, so
   they are never broken into pieces.

## Results (vocab size 600)

```
vocabulary: 578 tokens (322 merges) + 1 special
text         chars   bytes  tokens  tokens/char
English       1394    1394     576         0.41
Telugu         761    2051     372         0.49
round trip with a special token: True
```

Training stopped at 322 merges because no pair appeared twice after that.

**What this shows:** Telugu needs about 20% more tokens per character than English with the
same tokenizer. Each Telugu character is 3 bytes in UTF-8 (2,051 bytes for 761 characters),
so it starts with three times as many tokens and needs many more merges to catch up. My
Telugu text is also shorter, so fewer Telugu pairs reached the top of the counts. With a
tokenizer trained mostly on English, as many real ones are, the gap would be much larger,
which means a model reads less Telugu for the same number of tokens.

## Tests

`tests/test_bpe.py` checks pair counting and merging, round trips for English, Telugu, Hindi,
emoji, extra spaces and empty text, special tokens, saving and loading, unknown ids, and that
merges really make text shorter.
