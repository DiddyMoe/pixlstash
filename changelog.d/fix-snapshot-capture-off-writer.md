- Fixed: taking a snapshot, including the automatic daily one, paused imports,
  edits and background work until the snapshot had been copied and
  compressed, which on a large library could take minutes. They now carry on
  while the snapshot is made, and the snapshot still holds the library exactly
  as it was when the snapshot started.
- Fixed: a snapshot that failed part-way, ran out of disk space or was
  interrupted by quitting could leave a half-written file in the snapshots
  folder under a finished snapshot's name. A snapshot that fails or is
  cancelled now removes everything it wrote, and quitting while one is being
  made stops it cleanly instead of closing the library under it. If PixlStash
  is killed or the computer loses power while a snapshot is being made, no
  snapshot in the list is left half-written, but files of the interrupted
  snapshot can stay in the snapshots folder and take up space until removed by
  hand.
