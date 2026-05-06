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

    # Gemma 3 chat template special token IDs
    # These are ADDED tokens — SentencePiece does NOT know them.
    # They MUST be injected as raw IDs, never encoded as text.
    START_OF_TURN_ID = 105    # <start_of_turn>
    END_OF_TURN_ID = 106      # <end_of_turn>

    # MedGemma / Gemma 3 image token IDs
    MEDGEMMA_IMG_START = 255999    # <start_of_image>
    MEDGEMMA_IMG_SOFT = 262144     # <image_soft_token>
    MEDGEMMA_IMG_END = 256000      # <end_of_image>

    # Full model vocab (larger than SentencePiece base vocab)
    MODEL_VOCAB_SIZE = 262145      # 262144 + 1

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
        """Decode token IDs back to text.

        Uses MODEL_VOCAB_SIZE (not SentencePiece vocab size) so that
        tokens in the extended range (e.g. 256001–262144) are not
        accidentally filtered out.  Special tokens that SentencePiece
        cannot decode are mapped to their string names.
        """
        SPECIAL = {
            self.BOS_ID: "",
            self.EOS_ID: "",
            self.PAD_ID: "",
            self.START_OF_TURN_ID: "",
            self.END_OF_TURN_ID: "",
            self.MEDGEMMA_IMG_START: "",
            self.MEDGEMMA_IMG_SOFT: "",
            self.MEDGEMMA_IMG_END: "",
        }

        decodable = []
        for i in ids:
            if i in SPECIAL:
                # Flush decodable buffer, then skip special token
                continue
            if i < self.vocab_size:
                decodable.append(i)
        return self.sp.DecodeIds(decodable)

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
        """Build Gemma 3 / MedGemma chat-template input matching HF exactly.

        Target token sequence (from HF processor debug):
          <bos> <start_of_turn> user \\n {system} \\n\\n {text} \\n\\n
          <start_of_image> <img_soft>*N <end_of_image>
          \\n\\n <end_of_turn> \\n <start_of_turn> model \\n

        CRITICAL: <start_of_turn> / <end_of_turn> are ADDED special
        tokens that SentencePiece cannot encode.  They must be injected
        as literal integer IDs.
        """
        ids = [self.BOS_ID]                    # <bos>
        ids.append(self.START_OF_TURN_ID)      # <start_of_turn>

        # Encode the user turn body.
        # HF concatenates: role + "\\n" + system_prefix + text
        if system_prompt:
            body = f"user\n{system_prompt}\n\n{text}\n\n"
        else:
            body = f"user\n{text}\n\n"
        ids.extend(self.sp.EncodeAsIds(body))

        # Image block
        ids.append(self.MEDGEMMA_IMG_START)                    # <start_of_image>
        ids.extend([self.MEDGEMMA_IMG_SOFT] * num_image_tokens)
        ids.append(self.MEDGEMMA_IMG_END)                      # <end_of_image>

        # Close user turn, open model turn
        ids.extend(self.sp.EncodeAsIds("\n\n"))                # \n\n
        ids.append(self.END_OF_TURN_ID)                        # <end_of_turn>
        ids.extend(self.sp.EncodeAsIds("\n"))                  # \n
        ids.append(self.START_OF_TURN_ID)                      # <start_of_turn>
        ids.extend(self.sp.EncodeAsIds("model\n"))             # model\n

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
