- Fixed: emptying a large Scrapheap could fail with "Couldn't empty the
  scrapheap. timeout of 60000ms exceeded" and leave it partly emptied. Empty
  scrapheap, Restore all, and Delete forever or Restore on a selection now run
  in the background. The bar above the Scrapheap counts the pictures as they
  go, and you can stop the run from there. A run also shows in the task
  manager on every screen, with its progress and a Stop. Leaving the Scrapheap
  or reloading the page does not stop it.
- A Restore all or a restore of a selection can be undone from its notice,
  even when it took several History steps. If History has moved on, the
  notice says what is in the way and offers Finish undo. Ctrl+Z undoes a
  large restore up to 1,000 pictures at a time. Through the API, a whole
  restore run can be undone in the background, newest part first; it stops
  without changing anything further if a later edit in History is in the way.
