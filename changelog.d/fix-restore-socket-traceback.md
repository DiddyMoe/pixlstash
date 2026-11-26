- Fixed: restoring a snapshot no longer writes an error with a traceback to the
  server log for each open browser tab. The tabs were always closed cleanly;
  only the log said otherwise.
