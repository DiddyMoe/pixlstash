- Fixed: background work (tagging, captions, quality scores) could stop for good
  when a task worker was lost to an unexpected error, with nothing on screen to
  say so, and imports could be refused meanwhile. A lost worker is now started
  again within about ten seconds, even when nothing new is queued, and while one
  is missing the Tasks panel says which work is waiting and shows the last
  error.
- Fixed: a task could be counted as finished twice, or never, when work was
  cancelled or the app shut down just as the task was being queued or was
  waiting for GPU memory. A task that never finished kept its pictures out of
  later passes until a restart. Every task now finishes exactly once.
