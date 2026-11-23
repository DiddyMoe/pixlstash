- Fixed: dropping a folder, or importing a zip, that held pictures with the
  same name in different subfolders gave every one of them the tags from just
  one of the caption files. Each picture now takes its tags only from the
  caption file in its own folder.
- Fixed: a dropped folder that the browser could not read could leave the
  import waiting forever, and files the browser could not read were skipped
  without a word. The rest of the drop is now imported, and a message says how
  many items could not be read and were skipped.
