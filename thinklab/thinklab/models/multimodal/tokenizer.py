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

    # MedGemma specific tokens
    MEDGEMMA_IMG_START = 255999    # <image>
    MEDGEMMA_IMG_SOFT = 262144     # <image_soft_token>
    MEDGEMMA_IMG_END = 256000      # <end_of_image>

    def build_gemma3_input(
        self, text: str, num_image_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """Build Gemma 3 / MedGemma chat-template input matching HF jinja exactly."""
        # MedGemma prepends the system prompt to the user turn
        prefix = f"{system_prompt}\n\n" if system_prompt else ""
        
        # Build text parts
        chat_text = f"{self.START_OF_TURN}user\n{prefix}{text}\n\n"
        
        # Encode text, with <bos>
        ids = self.encode(chat_text, add_bos=True, add_eos=False)
        
        # Append image tokens exactly as HF does:
        # <image> + (<image_soft_token> * 4096) + <end_of_image>
        ids.append(self.MEDGEMMA_IMG_START)
        ids.extend([self.MEDGEMMA_IMG_SOFT] * num_image_tokens)
        ids.append(self.MEDGEMMA_IMG_END)
        
        # Close the user turn and start the model turn
        suffix = f"\n\n{self.END_OF_TURN}\n{self.START_OF_TURN}model\n"
        ids.extend(self.encode(suffix, add_bos=False, add_eos=False))
        
        return ids

    def build_input(
        self, text: str, num_image_tokens: int = 256,
        model_type: str = "gemma1",
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """Unified input builder - auto-selects format by model_type."""
        if model_type in ("gemma2", "gemma3"):
            return self.build_gemma3_input(text, num_image_tokens, system_prompt)
        return self.build_paligemma_input(text, num_image_tokens)
