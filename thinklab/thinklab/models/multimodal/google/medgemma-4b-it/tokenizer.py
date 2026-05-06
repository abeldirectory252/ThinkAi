"""
MedGemma tokenizer — Gemma 3 chat template format.
Uses SentencePiece directly (no HuggingFace dependency).
"""
from pathlib import Path
from typing import List, Optional
import sentencepiece as spm


class MedGemmaTokenizer:
    """Gemma 3 / MedGemma tokenizer with chat template support."""

    BOS_ID = 2
    EOS_ID = 1
    PAD_ID = 0
    START_OF_TURN_ID = 105
    END_OF_TURN_ID = 106
    IMG_START_ID = 255999
    IMG_SOFT_ID = 262144
    IMG_END_ID = 256000
    MODEL_VOCAB_SIZE = 262145

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromFile(str(model_path))
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        SPECIAL = {self.BOS_ID, self.EOS_ID, self.PAD_ID,
                    self.START_OF_TURN_ID, self.END_OF_TURN_ID,
                    self.IMG_START_ID, self.IMG_SOFT_ID, self.IMG_END_ID}
        decodable = [i for i in ids if i not in SPECIAL and i < self.vocab_size]
        return self.sp.DecodeIds(decodable)

    def build_input(self, text: str, num_image_tokens: int = 256,
                    system_prompt: Optional[str] = None, **kwargs) -> List[int]:
        """Build Gemma 3 chat-template input:
        <bos><start_of_turn>user\\n{system}\\n\\n{text}\\n\\n
        <start_of_image><img_soft>*N<end_of_image>
        \\n\\n<end_of_turn>\\n<start_of_turn>model\\n
        """
        ids = [self.BOS_ID, self.START_OF_TURN_ID]
        if system_prompt:
            body = f"user\n{system_prompt}\n\n{text}\n\n"
        else:
            body = f"user\n{text}\n\n"
        ids.extend(self.sp.EncodeAsIds(body))
        ids.append(self.IMG_START_ID)
        ids.extend([self.IMG_SOFT_ID] * num_image_tokens)
        ids.append(self.IMG_END_ID)
        ids.extend(self.sp.EncodeAsIds("\n\n"))
        ids.append(self.END_OF_TURN_ID)
        ids.extend(self.sp.EncodeAsIds("\n"))
        ids.append(self.START_OF_TURN_ID)
        ids.extend(self.sp.EncodeAsIds("model\n"))
        return ids
