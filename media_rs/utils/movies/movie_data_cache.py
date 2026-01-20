import os
import pickle
import numpy as np
import shutil

from scipy.sparse import load_npz
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from media_rs.utils.load_data import load_faiss_index

from pathlib import Path
from typing import Optional


HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO_ID")
CACHE_FOLDER = os.getenv("CACHE_FOLDER")

if not HF_REPO:
    raise ValueError(f"Environment variable HF_REPO not defined.")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")
if not CACHE_FOLDER:
    raise ValueError("CACHE_FOLDER environment variable must be set")

print(f"Using CACHE_FOLDER at {CACHE_FOLDER}")

class MovieDataCache:
    """
    Singleton class to manage precomputed files either from Hugging Face or a local folder,
    with type-aware loading.
    """
    _instance = None

    FILES_TO_DOWNLOAD = [
        "tfidf/faiss_index_users.index",
        "sbert/faiss_index_users.index",
        "user_item_matrix.npz",
        "tfidf/item_embeddings.npy",
        "sbert/item_embeddings.npy",
        "sbert/sbert_model",
        "item_index.pkl",
        "item_topk_cf.pkl",
        "tfidf/item_topk_content.pkl",
        "sbert/item_topk_content.pkl",
        "tfidf/user_embeddings.npy",
        "sbert/user_embeddings.npy",
        "user_index.pkl",
        "tfidf/tfidf_vectorizer.pkl",
        "tfidf/svd.pkl"
    ]

    def __new__(cls, repo_id: str = None, local_dir: str = None):
        if cls._instance is None:
            cls._instance = super(MovieDataCache, cls).__new__(cls)
            cls._instance.repo_id = repo_id
            cls._instance.local_dir = Path(local_dir) if local_dir else None
            cls._instance._preload_files()
            cls._instance._load_all_files()
        return cls._instance

    def _preload_files(self):
        self.paths = {}
        
        if self.local_dir:
            # Use local files
            print(f"Using local files from {self.local_dir}...")
            for f in self.FILES_TO_DOWNLOAD:
                file_path = self.local_dir / f
                if not file_path.exists():
                    raise FileNotFoundError(f"Expected file {file_path} does not exist.")
                self.paths[f] = file_path
            print("All local files mapped!")
            
        else:
            # Download from HF Hub
            if not self.repo_id:
                raise ValueError("repo_id must be provided if local_dir is not set.")
            print(f"Preloading Hugging Face files from {self.repo_id}...")
            
            for f in self.FILES_TO_DOWNLOAD:
                if "." not in f:
                    # folder -> use snapshot_download
                    path = Path(
                        snapshot_download(
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            token=HF_TOKEN,
                            allow_patterns=[f + "/*"],
                            cache_dir=CACHE_FOLDER
                        )
                    )
                    # snapshot_download returns the root snapshot folder, append subfolder
                    self.paths[f] = path / f
                else:
                    path = Path(hf_hub_download(
                        repo_id=self.repo_id, 
                        filename=f, 
                        repo_type="dataset",
                        token=HF_TOKEN,
                        cache_dir=CACHE_FOLDER
                    ))
                    self.paths[f] = path
                    
                print(f"Cached {f} at {path}")
                
            print("All files cached!")

    def _load_all_files(self):
        """
        Load all files into memory when the singleton is instantiated.
        """
        self.data = {}
        print("Loading files into memory...")
        for f in self.FILES_TO_DOWNLOAD:
            self.data[f] = self._load_file(f)
        print("All files loaded into memory!")

    def get_path(self, filename: str) -> Path:
        """
        Returns the path of a file (either local or cached from HF Hub)
        """
        return self.paths.get(filename)
    
    def _load_file(self, filename: str):
        """Load a file according to its type."""
        path = self.get_path(filename)
        if path is None:
            raise ValueError(f"File {filename} not found in cache.")

        if filename.endswith(".npy"):
            return np.load(path)
        elif filename.endswith(".npz"):
            return load_npz(path)
        elif filename.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif filename.endswith(".index"):
            return load_faiss_index(str(path))
        elif filename == "sbert/sbert_model":
            return SentenceTransformer(str(path))
        else:
            # fallback: return Path object
            return path
        
    def _is_cache_complete(self) -> bool:
        for f in self.FILES_TO_DOWNLOAD:
            path = self.paths.get(f)
            if not path or not path.exists():
                return False
        return True
        
    def get(self, filename: str):
        """
        Get the in-memory object for a file.
        """
        if filename not in self.data:
            raise ValueError(f"{filename} not loaded in memory.")
        return self.data[filename]
    
_CACHE: MovieDataCache | None = None

def get_movie_data_cache(local_dir: Optional[str]=None) -> MovieDataCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = MovieDataCache(repo_id=HF_REPO, local_dir=local_dir)
    return _CACHE