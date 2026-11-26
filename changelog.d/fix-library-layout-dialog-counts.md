- Fixed: reopening the library layout dialog could show the count from the last
  time it was open instead of counting again.
- Fixed: in the library layout dialog, a count that could not be read, or a
  layout save that was refused while the server could not be reached, left the
  move bar on "counting…" for good. It now says "count unavailable". A refused
  save whose layout could be read again counts that layout.
