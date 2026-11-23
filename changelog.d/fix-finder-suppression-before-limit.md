- Fixed: pictures PixlStash cannot read (a damaged file, or one on a drive that
  is not connected) no longer hold up the background work on the rest of your
  library. A run of them used to leave descriptions, quality scores, hashes and
  the other stages idle with work still waiting, and tens of thousands of them
  (a whole disconnected drive) stopped thumbnails, faces and tags with a
  database error. The remaining counts for thumbnails, descriptions and quality
  now leave those pictures out, so they reach zero.
