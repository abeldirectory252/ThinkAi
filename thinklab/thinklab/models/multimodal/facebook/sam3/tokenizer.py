"""
SAM3 Tokenizer — CLIP BPE tokenizer wrapper.

SAM3 uses CLIP text encoding (vocab=49408, max_len=32).
NO chat template — this is a segmentation model, not conversational.
Uses simple text prompts like "segment the red object".
SAM3 Tokenizer — CLIP BPE tokenizer.
tokenizer.py

"""

import json
import re
from pathlib import Path
from typing import List


class Sam3Tokenizer:
    HAS_CHAT_TEMPLATE = False
    BOS_ID = 49406
    EOS_ID = 49407
    PAD_ID = 49407
    VOCAB_SIZE = 49408
    MAX_LENGTH = 32

    def __init__(self, tokenizer_path):
        path = Path(tokenizer_path)
        if path.is_dir():
            vocab_file = path / "vocab.json"
            merges_file = path / "merges.txt"
        else:
            vocab_file = path.parent / "vocab.json"
            merges_file = path.parent / "merges.txt"

        if vocab_file.exists() and merges_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
            self.decoder_map = {v: k for k, v in self.encoder.items()}
            with open(merges_file, "r", encoding="utf-8") as f:
                merges = f.read().split("\n")[1:]
                merges = [tuple(m.split()) for m in merges if m.strip()]
            self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        else:
            self.encoder = None
            self.decoder_map = None
            self.bpe_ranks = None

        self._pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"""
            r"""|[a-zA-Z]+|[0-9]|[^\sa-zA-Z0-9]+""",
            re.IGNORECASE)

    def _bpe(self, token):
        if self.bpe_ranks is None:
            return [token + "</w>"]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text, add_bos=True, add_eos=True):
        text = text.lower().strip()
        if self.encoder is None:
            ids = []
            if add_bos:
                ids.append(self.BOS_ID)
            for word in text.split()[:self.MAX_LENGTH - 2]:
                ids.append(hash(word) % (self.VOCAB_SIZE - 2) + 1)
            if add_eos:
                ids.append(self.EOS_ID)
            return self._pad(ids)

        ids = []
        if add_bos:
            ids.append(self.BOS_ID)
        for match in self._pat.finditer(text):
            token = match.group().lower()
            for bt in self._bpe(token):
                if bt in self.encoder:
                    ids.append(self.encoder[bt])
        if add_eos:
            ids.append(self.EOS_ID)
        return self._pad(ids)

    def _pad(self, ids):
        if len(ids) > self.MAX_LENGTH:
            ids = ids[:self.MAX_LENGTH - 1] + [self.EOS_ID]
        while len(ids) < self.MAX_LENGTH:
            ids.append(self.PAD_ID)
        return ids

    def decode(self, ids):
        if self.decoder_map is None:
            return "<decode unavailable>"
        tokens = []
        for i in ids:
            if i in (self.BOS_ID, self.EOS_ID, self.PAD_ID):
                continue
            if i in self.decoder_map:
                tokens.append(self.decoder_map[i])
        return "".join(tokens).replace("</w>", " ").strip()

    def build_input(self, text, **kwargs):
        return self.encode(text)

    def get_attention_mask(self, ids: List[int]) -> List[int]:
        mask = []
        found_eos = False
        for idx, token_id in enumerate(ids):
            if not found_eos:
                mask.append(1)
                if token_id == self.EOS_ID and idx > 0:
                    found_eos = True
            else:
                mask.append(0)
        return mask