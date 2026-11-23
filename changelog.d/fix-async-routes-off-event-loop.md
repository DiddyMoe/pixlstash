- Fixed: while the library was busy saving (a large background batch, a
  restore or a library switch), saving an edit, changing a setting, uploading a
  file, opening a thumbnail or signing in with a share link could freeze the
  whole app until the library was free again. Those requests now wait on their
  own, and the rest of the app stays responsive meanwhile.
