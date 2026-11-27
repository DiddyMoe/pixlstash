- Fixed: with a log file configured, the server wrote nothing to it at the
  default log level, and at other levels the console showed nothing. The file
  now gets the log at every level, and the console keeps it too.
