"""
Download model weights from HuggingFace Hub using raw HTTP.
No huggingface_hub / transformers dependency.
Supports safetensors parsing natively.
"""
import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"
HF_CDN = "https://huggingface.co"

DTYPE_MAP = {
    "F32":  (torch.float32,  4),
    "F16":  (torch.float16,  2),
    "BF16": (torch.bfloat16, 2),
    "I64":  (torch.int64,    8),
    "I32":  (torch.int32,    4),
    "I16":  (torch.int16,    2),
    "I8":   (torch.int8,     1),
    "U8":   (torch.uint8,    1),
    "BOOL": (torch.bool,     1),
}


# ════════════════════════════════════════════════════════════════════
#  SafeTensors Loader  (no external lib needed)
# ════════════════════════════════════════════════════════════════════
class SafeTensorsLoader:
    """Parse a .safetensors file without the safetensors library."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header: Dict = {}
        self.data_offset: int = 0
        self._parse_header()

    # ── internal ────────────────────────────────────────────────────
    def _parse_header(self):
        with open(self.path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            self.data_offset = 8 + header_size
        self.header = json.loads(header_json)
        # remove metadata key if present
        self.header.pop("__metadata__", None)

    # ── public API ──────────────────────────────────────────────────
    def keys(self) -> List[str]:
        return list(self.header.keys())

    def get_tensor(self, name: str) -> torch.Tensor:
        meta = self.header[name]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]

        pt_dtype, elem_size = DTYPE_MAP[dtype_str]
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + start)
            raw = f.read(end - start)

        # numpy intermediate for non-standard dtypes
        if pt_dtype == torch.bfloat16:
            # read as uint16 then view as bfloat16
            arr = np.frombuffer(raw, dtype=np.uint16).copy()
            tensor = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
        else:
            np_dtype = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.int64:   np.int64,
                torch.int32:   np.int32,
                torch.int16:   np.int16,
                torch.int8:    np.int8,
                torch.uint8:   np.uint8,
                torch.bool:    np.bool_,
            }[pt_dtype]
            arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape)
            tensor = torch.from_numpy(arr)
        return tensor

    def load_all(self) -> Dict[str, torch.Tensor]:
        return {k: self.get_tensor(k) for k in self.keys()}


# ════════════════════════════════════════════════════════════════════
#  HuggingFace Downloader
# ════════════════════════════════════════════════════════════════════
class HuggingFaceDownloader:
    """Download model files from HuggingFace using raw requests."""

    def __init__(
        self,
        repo_id: str,
        save_dir: str | Path,
        token: Optional[str] = None,
        revision: str = "main",
    ):
        self.repo_id = repo_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.revision = revision
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # ── list remote files ───────────────────────────────────────────
    def list_files(self) -> List[dict]:
        url = f"{HF_API}/{self.repo_id}/tree/{self.revision}"
        files = []
        params: dict = {}
        while True:
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            files.extend(batch)
            # HF paginates with ?cursor=
            if len(batch) < 50:
                break
            params["cursor"] = batch[-1].get("oid", "")
        return files

    # ── download single file ────────────────────────────────────────
    def download_file(
        self, filename: str, expected_size: Optional[int] = None
    ) -> Path:
        dest = self.save_dir / filename
        if dest.exists():
            if expected_size and dest.stat().st_size == expected_size:
                logger.info("Skipping %s (already downloaded)", filename)
                return dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = (
            f"{HF_CDN}/{self.repo_id}/resolve/{self.revision}/{filename}"
        )
        with self.session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=filename,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        logger.info("Downloaded %s → %s", filename, dest)
        return dest

    # ── download entire model ───────────────────────────────────────
    def download_model(
        self,
        patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Download files matching *patterns* (glob-style).
        Defaults to config + tokenizer + safetensors.
        """
        if patterns is None:
            patterns = [
                "config.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "*.safetensors",
            ]
        remote = self.list_files()
        downloaded: List[Path] = []
        for entry in remote:
            name = entry.get("path", entry.get("rfilename", ""))
            size = entry.get("size", None)
            if not name:
                continue
            if self._matches(name, patterns):
                downloaded.append(self.download_file(name, size))
        return downloaded

    # ── load all safetensors into one state dict ────────────────────
    def load_state_dict(self) -> Dict[str, torch.Tensor]:
        st_files = sorted(self.save_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(
                f"No .safetensors files in {self.save_dir}"
            )
        state: Dict[str, torch.Tensor] = {}
        for sf in st_files:
            logger.info("Loading %s …", sf.name)
            loader = SafeTensorsLoader(sf)
            state.update(loader.load_all())
        return state

    # ── helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _matches(name: str, patterns: List[str]) -> bool:
        from fnmatch import fnmatch
        return any(fnmatch(name, p) for p in patterns)
