- Fixed: a snapshot restore that was going to be refused no longer starts
  first. Other open tabs no longer reload, and the app no longer turns
  requests away while the restore decides.
- Fixed: a restore that cannot start because too many picture files are
  missing, or because the library folder cannot be read, now says so instead
  of reporting a server error, and changes nothing. The missing-files refusal
  explains that the library folder may not be fully reachable (for example, a
  network drive that is not connected).
- Fixed: a waiting duplicate scan now stops a restore instead of being lost to
  it. The refusal names the work in plain words ("Duplicate scan") rather than
  internal task names.
- Changed: if you start something while a restore is running (a face
  re-detect, reading text again, object detection), it is refused with "Not
  started: a library restore is in progress. Try again once it has finished."
  Before, it started and was then stopped without a word. Pages refused during
  a restore no longer say that PixlStash is switching library.
- Changed: runs that a restore ends now say "The library was restored to a
  snapshot taken before this run finished." instead of saying the server
  stopped.
- Fixed: restoring several pictures, sets or characters at once could bring
  back a character, picture set or project that was deleted while the restore
  ran. Those items are now left out and listed as not restored, and the line
  of any item that is refused now says why, instead of only "could not be
  restored".
- Fixed: emptying the Scrapheap while a restore is running no longer uses up
  your confirmation. The refusal comes first, so you can empty it again once
  the restore has finished.
