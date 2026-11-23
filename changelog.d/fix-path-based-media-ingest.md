- Fixed: importing a picture or video read the whole file into memory first,
  whether it was added in the app, found in a reference or watched folder,
  brought in by a local folder import or made by a ComfyUI run started in
  PixlStash, so a large video needed that much free memory to import. PixlStash
  now works from the file on disk, and a ComfyUI run's outputs are saved to disk
  as they arrive.
- Fixed: a picture or video could take its creation date from stray bytes inside
  it that happened to look like a video's date field, giving it a date years
  off. A video is now dated from its own movie header, a picture from its EXIF
  date, and either from the file's modification time when it has neither.
- Fixed: an import cut short (a full disk, or the app stopping mid-write)
  could leave a half-written file in the library, and an import whose file name
  was already taken could adopt the file that was there as its own. A file is
  now written under a hidden name and renamed into place once it is complete,
  and an import no longer takes over a different file at its name.
