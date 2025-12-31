import os
import pickle
import numpy as np

from scipy.sparse import load_npz
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from media_rs.utils.load_data import load_faiss_index

from pathlib import Path


HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO_ID")
cache_folder = "data/movies/raw/ml-latest/hf_cache"

if not HF_REPO:
    raise ValueError(f"Environment variable HF_REPO not defined.")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")

class MovieDataCache:
    """
    Singleton class to manage precomputed files either from Hugging Face or a local folder,
    with type-aware loading.
    """
    _instance = None

    FILES_TO_DOWNLOAD = [
        "faiss_index_users.index",
        "user_item_matrix.npz",
        "movies_item_embeddings.npy",
        "movie_sbert_model",
        "item_index.pkl",
        "movies_item_topk_cf.pkl",
        "movies_item_topk_content.pkl",
        "movies_user_embeddings.npy",
        "user_index.pkl"
    ]

    def __new__(cls, repo_id: str = None, local_dir: str = None):
        if cls._instance is None:
            cls._instance = super(MovieDataCache, cls).__new__(cls)
            cls._instance.repo_id = repo_id
            cls._instance.local_dir = Path(local_dir) if local_dir else None
            cls._instance._preload_files()
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
                            cache_dir=cache_folder
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
                        cache_dir=cache_folder
                    ))
                    self.paths[f] = path
                    
                print(f"Cached {f} at {path}")
                
            print("All files cached!")

    def get_path(self, filename: str) -> Path:
        """
        Returns the path of a file (either local or cached from HF Hub)
        """
        return self.paths.get(filename)
    
    def load(self, filename: str):
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
        elif "sbert" in str(filename):
            return SentenceTransformer(str(path))
        else:
            # fallback: return Path object
            return path
        
MOVIE_DATA_CACHE = MovieDataCache(HF_REPO)