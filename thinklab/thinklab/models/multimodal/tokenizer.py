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

    # Gemma 3 chat template special tokens
    START_OF_TURN = "<start_of_turn>"
    END_OF_TURN = "<end_of_turn>"

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
        """Build PaliGemma 1 input: [IMAGE_TOKENS...] <bos> text_tokens."""
        image_ids = [self.IMAGE_TOKEN_ID] * num_image_tokens
        text_ids = self.encode(text, add_bos=True, add_eos=False)
        return image_ids + text_ids

    def build_gemma3_input(
        self, text: str, num_image_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """Build Gemma 3 / MedGemma chat-template input.

        Format:
          [IMAGE_TOKENS...] <bos>
          <start_of_turn>system\\n{system}\\n<end_of_turn>   (optional)
          <start_of_turn>user\\n{text}<end_of_turn>
          <start_of_turn>model\\n
        """
        image_ids = [self.IMAGE_TOKEN_ID] * num_image_tokens

        # Build chat text
        parts = []
        if system_prompt:
            parts.append(f"{self.START_OF_TURN}system\n{system_prompt}\n{self.END_OF_TURN}\n")
        parts.append(f"{self.START_OF_TURN}user\n{text}{self.END_OF_TURN}\n")
        parts.append(f"{self.START_OF_TURN}model\n")
        chat_text = "".join(parts)

        text_ids = self.encode(chat_text, add_bos=True, add_eos=False)
        return image_ids + text_ids

    def build_input(
        self, text: str, num_image_tokens: int = 256,
        model_type: str = "gemma1",
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """Unified input builder - auto-selects format by model_type."""
        if model_type in ("gemma2", "gemma3"):
            return self.build_gemma3_input(text, num_image_tokens, system_prompt)
        return self.build_paligemma_input(text, num_image_tokens)
