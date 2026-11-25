- Fixed: emptying a large Scrapheap could fail with "Couldn't empty the
  scrapheap. timeout of 60000ms exceeded" and leave it partly emptied. Empty
  scrapheap, Restore all, and Delete forever or Restore on a selection now run
  in the background. The bar above the Scrapheap counts the pictures as they
  go, and you can stop the run from there. Leaving the Scrapheap or reloading
  the page does not stop it. A large restore is undone with Ctrl+Z a few
  thousand pictures at a time.
