from pathlib import Path

def parse_openlibrary_ratings(path: str):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            # Defensive unpacking
            work_id = parts[0] if len(parts) > 0 else None
            book_id = parts[1] if len(parts) > 1 and parts[1] else None
            rating = int(parts[2]) if len(parts) > 2 else None
            date = parts[3] if len(parts) > 3 else None

            if work_id and rating is not None:
                records.append({
                    "work_id": work_id.replace("/works/", ""),
                    "book_id": book_id.replace("/books/", "") if book_id else None,
                    "rating": rating,
                    "date": date
                })

    return records