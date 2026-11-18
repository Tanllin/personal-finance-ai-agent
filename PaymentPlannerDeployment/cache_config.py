"""Put downloadable, non-user caches inside the deploy artifact.

Chroma 1.5.9 has no cache-directory constructor argument; set its pinned ONNX
implementation's download path before any embedding instance is created.
"""
import os
from pathlib import Path


def configure_caches():
    root = Path(__file__).resolve().parent / ".cache"
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root / "tiktoken"))
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = root / "all-MiniLM-L6-v2"
