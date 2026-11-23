- Fixed: an upload whose files had all been sent could be stopped as "Upload
  stalled" while the server was still saving them, and then sent again. The
  upload now waits for the server's answer once every byte is out.
- Fixed: an import that took more than about ten minutes on the server was
  reported as "Import timed out" while it carried on. The import is now
  followed until the server says it has finished, however long that takes.
- Fixed: switching away from the library view, or reloading the tab, while
  pictures were being imported lost track of the import. Its progress row now
  keeps going and says when the import has finished.
- Fixed: a dropped connection during an import was reported as a failed
  import. The row now says it is reconnecting, and if the server has forgotten
  the import it says so instead of calling it a failure; pictures already
  imported are kept.
- Fixed: an upload the server refused, for example because the disk was full,
  was sent twice more before failing, and its uploaded files stayed on the
  server until they expired. A refusal now ends the upload at once with the
  server's reason, and the uploaded files are removed.
- Fixed: if the answer to starting an import was lost, the import could be
  cancelled even though it had already started. PixlStash now asks the server
  whether the import started and follows it if it did.
- Fixed: the import progress row said "Checking for duplicates" for the whole
  import, whatever the server was doing. It now says which step the server is
  on: waiting, checking pictures, adding them to the library, finishing up.
