- Fixed: on Windows, a folder typed with a `\\?\` prefix, such as
  `\\?\C:\Windows`, was not refused as a system folder, and a folder spelled
  that way was not seen to overlap the PixlStash data folder or an existing
  reference folder. A prefixed folder is now checked as the folder it names.
