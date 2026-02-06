import pandas as pd
from pathlib import Path

from media_rs.utils.books.load_book_data import load_all_book_data

save_dir = Path("data/books/cache/")
file_dir = Path("data/books/book_dataset/raw/")

sbert_dir = save_dir / "sbert"
tfidf_dir = save_dir / "tfidf"

sbert_dir.mkdir(parents=True, exist_ok=True)
tfidf_dir.mkdir(parents=True, exist_ok=True)

# Load data
books, ratings, tags = load_all_book_data(file_dir)

books.to_csv(save_dir / "metadata.csv", index=False)