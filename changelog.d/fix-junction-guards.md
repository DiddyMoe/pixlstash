- Fixed: on Windows, a folder junction (a kind of folder link) was not
  treated like other folder links. Moving a model folder, installing a ComfyUI
  node pack, restoring a backup, switching libraries, removing a plugin,
  discarding an export, and cleaning up edited or shared copies could reach
  into the folder it points at, and so move or delete files outside the
  library. They now refuse or skip a junction, as does choosing a model
  folder, backup or reference folder that is one.
- Fixed: on Windows, watch-folder scans, picture counts, folder renames and
  importing the pictures already in a library's folder followed a junction,
  which can lead anywhere on the disk. They now skip it, as they skip a folder
  link. A watch folder reached through a junction is still imported from, but
  its files are no longer deleted after import.
