"""
Minimal Gemma tokenizer wrapping SentencePiece directly.
No HuggingFace dependency.
"""
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm


class GemmaTokenizer:
    """Load tokenizer.model from the downloaded PaliGemma weights."""

    BOS_ID = 2      # <bos>
    EOS_ID = 1      # <eos>
    PAD_ID = 0      # <pad>
    IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_ID = 257152   # PaliGemma's image placeholder token id

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromFile(str(model_path))
        self.vocab_size = self.sp.GetPieceSize()

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        # filter special tokens
        ids = [i for i in ids if i not in (self.BOS_ID, self.EOS_ID, self.PAD_ID)
               and i < self.vocab_size]
        return self.sp.DecodeIds(ids)

    def build_paligemma_input(
        self, text: str, num_image_tokens: int = 256
    ) -> List[int]:
        """Build input: [IMAGE_TOKENS...] <bos> text_tokens."""
        image_ids = [self.IMAGE_TOKEN_ID] * num_image_tokens
        text_ids = self.encode(text, add_bos=True, add_eos=False)
        return image_ids + text_ids
