import os
import pickle
import numpy as np

from scipy.sparse import load_npz
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from media_rs.utils.load_data import load_faiss_index

from pathlib import Path
from typing import Optional, Dict, Any


HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO_ID")
CACHE_FOLDER = os.getenv("CACHE_FOLDER")

if not HF_REPO:
    raise ValueError("HF_REPO_ID environment variable not defined.")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not defined.")
if not CACHE_FOLDER:
    raise ValueError("CACHE_FOLDER environment variable must be set")

print(f"[MovieDataCache] Using CACHE_FOLDER={CACHE_FOLDER}")

class MovieDataCache:
    """
    EAGER, IN-MEMORY cache.
    Everything loaded once during warmup().
    Optimized for fast requests, slow startup.
    """

    _instance = None

    FILES_ORDERED = [
        # ---- indices / mappings (cheap) ----
        "item_index.pkl",
        "user_index.pkl",

        # ---- sparse matrices ----
        "user_item_matrix.npz",

        # ---- embeddings ----
        "tfidf/item_embeddings.npy",
        "tfidf/user_embeddings.npy",
        "sbert/item_embeddings.npy",
        "sbert/user_embeddings.npy",

        # ---- FAISS ----
        "tfidf/faiss_index_users.index",
        "sbert/faiss_index_users.index",

        # ---- precomputed graphs ----
        "item_topk_cf.pkl",
        "tfidf/item_topk_content.pkl",
        "sbert/item_topk_content.pkl",

        # ---- ML artifacts ----
        "tfidf/tfidf_vectorizer.pkl",
        "tfidf/svd.pkl",

        # ---- models LAST ----
        "sbert/sbert_model",
    ]

    def __new__(cls, repo_id: str, local_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.repo_id = repo_id
            cls._instance.local_dir = Path(local_dir) if local_dir else None
            cls._instance.paths = {}
            cls._instance.data = {}
            cls._instance._resolve_paths()
        return cls._instance

    # ------------------------------------------------
    # Resolve file paths only
    # ------------------------------------------------

    def _resolve_paths(self):
        for f in self.FILES_ORDERED:
            if self.local_dir:
                path = self.local_dir / f
                if not path.exists():
                    raise FileNotFoundError(path)
                self.paths[f] = path
            else:
                if "." not in f:
                    root = snapshot_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=HF_TOKEN,
                        allow_patterns=[f + "/*"],
                        cache_dir=CACHE_FOLDER,
                    )
                    self.paths[f] = Path(root) / f
                else:
                    self.paths[f] = Path(
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=f,
                            repo_type="dataset",
                            token=HF_TOKEN,
                            cache_dir=CACHE_FOLDER,
                        )
                    )

    # ------------------------------------------------
    # WARMUP (called once)
    # ------------------------------------------------

    def warmup(self):
        """
        Load ALL assets into memory.
        Call once before serving traffic.
        """
        if self.data:
            return  # already warm

        print("MovieDataCache warmup started")

        for f in self.FILES_ORDERED:
            print(f"Loading {f}")
            self.data[f] = self._load_file(f)

        print("MovieDataCache warmup complete")

    # ------------------------------------------------
    # Access
    # ------------------------------------------------

    def get(self, filename: str):
        if filename not in self.data:
            raise RuntimeError(
                f"{filename} not loaded. Call warmup() first."
            )
        return self.data[filename]

    # ------------------------------------------------
    # Internal loading logic
    # ------------------------------------------------

    def _load_file(self, filename: str):
        path = self.paths[filename]

        if filename.endswith(".npy"):
            # fully load into RAM (no mmap)
            return np.load(path)

        if filename.endswith(".npz"):
            return load_npz(path)

        if filename.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)

        if filename.endswith(".index"):
            return load_faiss_index(str(path))

        if filename == "sbert/sbert_model":
            return SentenceTransformer(str(path))

        return path
# ----------------------------------------------------------------------
# Global accessor
# ----------------------------------------------------------------------

_CACHE: Optional[MovieDataCache] = None


def get_movie_data_cache(local_dir: Optional[str] = None) -> MovieDataCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = MovieDataCache(
            repo_id=None if local_dir else HF_REPO,
            local_dir=local_dir,
        )
    return _CACHE
