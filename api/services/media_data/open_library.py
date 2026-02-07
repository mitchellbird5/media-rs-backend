import requests


def get_open_library_details(title: str):
    url = f"https://openlibrary.org/search.json?title={title}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if len(data["docs"]) == 0:
        raise ValueError(f"No book found for title: {title}")

    book_info = None
    for doc in data["docs"]:
        if doc["title"].lower() == title.lower():
            book_info = doc
            break
          
    if book_info is None:
        raise ValueError(f"No exact match found for title: {title}")
    
    return book_info
      
def get_multiple_book_data(titles: list) -> list:
    return [get_open_library_details(title) for title in titles]