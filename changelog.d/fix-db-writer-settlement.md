- Fixed: a database error while saving a change could leave the app waiting
  forever, with later saves and background work stuck behind it. A failed
  save now reports its error and the next one goes ahead; if saving stops
  working altogether, changes are refused with an error straight away instead
  of hanging, and shutting down no longer pulls the library out from under a
  save that is still finishing.
