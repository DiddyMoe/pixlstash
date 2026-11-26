- Fixed: a watch folder set to delete files after import now deletes only the
  file it imported. A file outside the folder (through a link), a file changed
  or replaced since it was read, and a file in a place PixlStash may not touch
  are kept, and the log says why.
- Changed: a watch folder reached through a link (the folder itself or a folder
  above it) is still imported from, but its files are no longer deleted after
  import, because the link could point somewhere else by the time of the
  delete.
- Fixed: a watch folder could import the wrong file, or another file's caption
  text, when a file or folder inside it was replaced by a link at the moment it
  was read. It now reads only the file it checked. A caption file that is a
  link to a file outside the folder is left out, and the picture is imported
  without it.
- Fixed: one damaged entry in a zip no longer fails the upload or drops the
  rest of the archive. The other entries import, and the result names the
  entry that was skipped.
- Fixed: the same file arriving through two imports at once (a watch folder,
  an upload, a plugin output), or two identical files in a watch folder or
  dropped into the library folder together, became two pictures. They now
  become one, and an extra file is left where it is.
- Fixed: an import whose pictures were saved no longer reports a failure when
  the model engine cannot start afterwards; the pictures wait for tagging.
- Fixed: a full disk or a server that is shutting down now gives a clear
  "try again" error on import instead of an internal server error.
- Fixed: WebM, MOV and MKV videos keep their own format and extension on
  import instead of being recorded as MP4.
- Fixed: a copy of a library picture placed in the library folder is no longer
  imported as a second picture. It is left where it is and logged.
- The import folders list in Settings now says when a folder is refused, and
  why, apart from a folder that is missing or cannot be read.
- Fixed: moving a file in a reference folder while its picture is in the
  Scrapheap keeps that picture, with its tags, in the Scrapheap, instead of
  deleting it and bringing the file back as a new picture.
- Fixed: a scan of a reference folder, or of a library folder that syncs
  caption files, could undo a tag edit made at the same moment, or overwrite
  captions or tags imported or exported from the folder while it ran. The scan
  now leaves a picture changed meanwhile for the next scan.
- Fixed: a slow or sleeping disk under a reference folder no longer holds up
  every other change in the library while a scan checks it.
