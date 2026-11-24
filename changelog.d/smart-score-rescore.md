- Fixed: on a large library, removing or re-weighting a Penalised Tag in the
  smart score settings spun for a long time and then said "Failed to update
  smart score tags". The change now re-scores only the pictures that carry the
  tag, and the next start no longer re-scores the whole library for a change
  that was already applied.
- Fixed: giving a picture 1 star, or 4 stars or more, could re-score your whole
  library. That now happens only when the rating changes which pictures the
  smart score compares against, and the log says what changed. Undo, the
  Scrapheap and auto-stacking now refresh smart scores in the same way when
  they change those pictures; before, they left the old scores in place.
- Fixed: when auto-stacking copied a flaw tag such as "malformed eyes" to the
  other pictures in a group, those pictures kept the smart score they had
  before.
- Fixed: the Smart score row in the Tasks panel showed every picture as done
  while a re-score was running. It now counts the pictures left to score.
- Fixed: a picture you had rated 1 star, or 4 stars or more, before it
  finished importing did not count towards smart scores until the next rating
  change. Smart scores now refresh about a minute after the import's
  embeddings finish.
- Re-scoring a library with many rated pictures finishes sooner: each batch
  reads the pictures it compares against in about half the time.
