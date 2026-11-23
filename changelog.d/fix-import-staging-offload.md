- Fixed: uploading a large zip froze PixlStash, in every open window, while the
  server unpacked it, and on a slow drive any large upload made it stutter.
  Uploads are now written to disk in the background, and the pictures in a zip
  are unpacked a piece at a time instead of each being held whole in memory.
- Caption `.txt` files larger than 1 MiB are now left out of an import instead
  of being held in memory until it runs.
