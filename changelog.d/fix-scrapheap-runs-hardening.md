- Fixed: Delete forever could also destroy pictures that reached the Scrapheap
  after you saw its confirmation, such as pictures moved there from another
  tab, or put back there by Undo, while the dialog was open. It now deletes
  only what was in the Scrapheap when the confirmation was shown, and anything
  that arrived later stays in the Scrapheap.
- Changed: through the API, emptying or restoring the Scrapheap in a single
  request now takes at most 1,000 pictures, named one by one. Emptying or
  restoring the whole Scrapheap, which could run past the request's time limit
  on a large one, is done by starting a Scrapheap run, which works in the
  background.
- Fixed: a tag text search wrote two lines to the server log for every picture
  it compared, which flooded the log on a large library.
- Fixed: on a large Scrapheap, the check behind the Delete forever dialog
  listed every protected file. It now lists at most 200 of each kind and still
  counts all of them.
