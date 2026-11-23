- Fixed: a picture moved while PixlStash was checking its folder, or while it
  was clearing out pictures whose files had gone, could be deleted with its
  tags, score and sets, or pointed at a place the file had already left.
  PixlStash now checks again just before it moves or removes a picture, and
  leaves it for the next check if it has moved since or its file is back.
