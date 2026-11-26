- Fixed: a person a picture gains from merging duplicates, from Keep cover
  only, or from undo and redo is now given to the face already found on it,
  where before it was never given.
- Fixed: a person waiting for their picture's faces to be found could stay
  unassigned when the server stopped at the wrong moment. Start-up now
  finishes the assignment.
- Fixed: taking faces away from a person now takes exactly the faces chosen,
  whether they are picked by face or by picture, and re-indexes for text
  search only the pictures those faces were on. A request that matches none of
  that person's faces, or names a face that no longer exists, now changes
  nothing instead of failing with a server error.
