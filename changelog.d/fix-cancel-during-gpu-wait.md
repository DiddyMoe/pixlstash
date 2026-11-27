- Fixed: a background task that was cancelled or paused for other GPU work just
  as it was about to start could still run, or be reported as finished.
