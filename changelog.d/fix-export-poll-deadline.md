- Fixed: an export that took more than about ten minutes was reported as
  "Export timed out" while the server carried on with it. An export is now
  followed until the server says it has finished, however long that takes.
- Fixed: a dropped connection during an export was reported as a failed
  export. The progress card now says it is reconnecting and carries on, and if
  the server has forgotten the export it says so instead of calling it a
  failure.
- Fixed: Abort could leave the export card up for a moment longer. It now
  closes at once.
- Leaving the pictures view for another screen during an export now stops
  following it and says so: a ZIP export the server is still preparing won't
  download, and a folder export keeps being written on the server.
