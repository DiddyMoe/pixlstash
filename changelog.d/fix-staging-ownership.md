- Fixed: switching to another library while an upload import was still running
  could close the library under the import. The switch now waits, and if the
  import is still going it is refused with a message saying so.
- Fixed: starting the same upload import twice, for example by retrying after a
  slow reply, could import the pictures twice. The second request now follows
  the import that is already running.
- Fixed: a picture still uploading when the import started could be reported as
  received and then never imported. Starting the import is now refused until
  every upload has finished, and a cancel no longer pulls files from under an
  upload.
- Fixed: running out of disk space while uploading was reported as skipped
  files. The upload now stops with an out-of-space error and leaves no partial
  files behind.
- Fixed: a failed import kept its uploaded copies on disk. They are now removed,
  and leftover upload folders from an interrupted session are cleaned up.
- Fixed: an import could start even when PixlStash could not tell how much
  free space the library drive had. It now refuses and says why.
