from dataclasses import dataclass, field
from typing import Optional, List, Dict
from fastapi import Response

from media_rs.utils.session import get_or_create_session_id
from media_rs.rs_types.model import Medium
from api.services.database_query import query_database
from api.services.media_data.open_library import get_open_library_details

@dataclass
class BookData:
    title: str = ""
    itemId: Optional[int] = None
    url: Optional[str] = None
    authors: Optional[str] = None
    lang: Optional[str] = None
    cover_id: Optional[int] = None
    img: Optional[str] = None
    year: Optional[str] = None
    description: Optional[str] = None
    
def get_book_data(title: str) -> BookData:
    response = Response()
    results = query_database(
        response=response,
        title=title,
        medium=Medium.BOOKS,
        limit=1
    )
    if results:
        result = results[0]
        ol_details = get_open_library_details(title)
        return BookData(
            title=result.get("title", title),
            itemId=result.get("itemId"),
            url=result.get("url"),
            authors=result.get("authors"),
            lang=result.get("lang"),
            cover_id=ol_details.get("cover_i"),
            img=result.get("img"),
            year=result.get("year"),
            description=result.get("description")
        )
    else:
        return BookData(title=title)

def get_multiple_book_data(titles: List[str]) -> List[BookData]:
    data: List[BookData] = []
    for title in titles:
        try:
            data.append(get_book_data(title))
        except Exception:
            data.append(BookData(title=title))
    return data