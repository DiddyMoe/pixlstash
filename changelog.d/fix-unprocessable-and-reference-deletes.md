- Fixed: a picture PixlStash could not read because of a permission problem, a
  file still being written or a disk that came back was reported as corrupt.
  It is now retried by itself once the file can be read again, with no restart
  needed. Only a file whose contents really do not decode is set aside as
  unprocessable, and it is tried again when the file changes. Pictures
  PixlStash could not read or reach are remembered across a restart, so the
  background tasks no longer try every one of them again each time the server
  starts.
- Fixed: when quality scoring filled in a missing size for a rotated photo, it
  swapped the width and height. It now records them the way import does.
- Fixed: a picture in a reference folder whose file is a link to a drive that
  is not connected was removed by the missing-file cleanup. The folder scan
  already kept it, and now the cleanup does too.
- Uploads and watch-folder imports no longer wait while a scan of the library
  folder reads and renders new files.
