"""PaliGemma tokenizer — Image-prefix format. No chat template."""
from pathlib import Path
from typing import List, Optional
import sentencepiece as spm


class PaliGemmaTokenizer:
    HAS_CHAT_TEMPLATE = False
    BOS_ID, EOS_ID, PAD_ID = 2, 1, 0
    IMAGE_TOKEN_ID = 257152

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromFile(str(model_path))
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text, add_bos=True, add_eos=False):
        ids = self.sp.EncodeAsIds(text)
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids):
        special = {self.BOS_ID, self.EOS_ID, self.PAD_ID, self.IMAGE_TOKEN_ID}
        return self.sp.DecodeIds([i for i in ids if i not in special and i < self.vocab_size])

    def build_input(self, text, num_image_tokens=256, **kwargs):
        """[IMAGE_TOKENS...] <bos> text_tokens"""
        return [self.IMAGE_TOKEN_ID] * num_image_tokens + self.encode(text, add_bos=True)
