import os
from pathlib import Path
from huggingface_hub import login, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
LOCAL_FOLDER = "data/movies/cache/"
REPO_TYPE = "dataset"

if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")
login(HF_TOKEN)

def upload_folder_recursive(folder_path: str, repo_id: str, repo_type: str = "dataset", prefix=""):
    folder = Path(folder_path)
    for file_path in folder.iterdir():
        if file_path.is_file():
            remote_path = f"{prefix}/{file_path.name}" if prefix else file_path.name
            print(f"Uploading {remote_path}...")
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=repo_type
            )
        elif file_path.is_dir():
            new_prefix = f"{prefix}/{file_path.name}" if prefix else file_path.name
            upload_folder_recursive(str(file_path), repo_id, repo_type, prefix=new_prefix)

upload_folder_recursive(LOCAL_FOLDER, HF_REPO_ID, REPO_TYPE)