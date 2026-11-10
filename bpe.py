"""A byte-pair encoding tokenizer in plain Python.

Text is turned into UTF-8 bytes, so every string (English, Telugu, emoji) can be encoded.
Training repeatedly finds the most common pair of neighbouring tokens and merges it into a
new token. Special tokens such as <|endoftext|> are never split.
"""

import json
import re


def count_pairs(ids, counts=None):
    """Count how often each pair of neighbours appears: [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of pair in ids with new_id, left to right."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, special_tokens=("<|endoftext|>",)):
        self.merges = {}  # (id, id) -> new id, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}
        self._special_names = list(special_tokens)

    # ---- training -----------------------------------------------------------
    def train(self, text, vocab_size, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        # train per word-ish chunk so merges never cross a space boundary
        chunks = [list(c.encode("utf-8")) for c in re.findall(r"\s?\S+|\s+", text)]
        for new_id in range(256, vocab_size):
            counts = {}
            for chunk in chunks:
                count_pairs(chunk, counts)
            if not counts:
                break  # nothing left to merge
            # most common pair; ties broken by the smaller pair, so training is repeatable
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[pair] < 2:
                break
            chunks = [merge(c, pair, new_id) for c in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {new_id - 255}: {self.vocab[new_id]!r} ({counts[pair]} times)")
        self._add_special_tokens()

    def _add_special_tokens(self):
        next_id = max(self.vocab) + 1
        self.special_tokens = {}
        for name in self._special_names:
            self.special_tokens[name] = next_id
            next_id += 1

    # ---- encode / decode ------------------------------------------------------
    def _encode_ordinary(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = count_pairs(ids)
            # apply the earliest learned merge first, as in training
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        if not self.special_tokens:
            return self._encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids):
        inverse = {v: k for k, v in self.special_tokens.items()}
        out = bytearray()
        for i in ids:
            if i in inverse:
                out.extend(inverse[i].encode("utf-8"))
            elif i in self.vocab:
                out.extend(self.vocab[i])
            else:
                raise ValueError(f"unknown token id {i}")
        # a slice of ids may cut a multi-byte character; show it as the replacement character
        return out.decode("utf-8", errors="replace")

    # ---- saving -------------------------------------------------------------
    def save(self, path):
        data = {"merges": [[a, b, n] for (a, b), n in self.merges.items()],
                "special_tokens": self.special_tokens}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(special_tokens=list(data["special_tokens"]))
        for a, b, n in data["merges"]:
            tok.merges[(a, b)] = n
            tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = data["special_tokens"]
        return tok
