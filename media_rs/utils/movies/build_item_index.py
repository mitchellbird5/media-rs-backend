
def build_item_index(movies, links):

  idx_to_movieId = dict(enumerate(movies["movieId"].values))
  movieId_to_idx = {mid: idx for idx, mid in idx_to_movieId.items()}

  idx_to_title = dict(enumerate(movies["title"].values))
  title_to_idx = {title: idx for idx, title in idx_to_title.items()}

  movieId_to_imdbId = dict(zip(links["movieId"], links["imdbId"]))
  movieId_to_tmdbId = dict(zip(links["movieId"], links["tmdbId"]))


  imdbId_to_movieId = {imdb: mid for mid, imdb in movieId_to_imdbId.items()}
  tmdbId_to_movieId = {tmdb: mid for mid, tmdb in movieId_to_tmdbId.items()}

  return {
      "num_items": len(movies),

      # forward
      "idx_to_movieId": idx_to_movieId,
      "idx_to_title": idx_to_title,
      "movieId_to_imdbId": movieId_to_imdbId,
      "movieId_to_tmdbId": movieId_to_tmdbId,

      # reverse
      "movieId_to_idx": movieId_to_idx,
      "title_to_idx": title_to_idx,
      "imdbId_to_movieId": imdbId_to_movieId,
      "tmdbId_to_movieId": tmdbId_to_movieId,
  }