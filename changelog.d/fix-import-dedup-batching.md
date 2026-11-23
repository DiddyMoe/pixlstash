- Fixed: two upload imports, or an upload import and a ComfyUI generation
  started from PixlStash, running at the same time could both add the same new
  picture. Upload imports and these generations now check for duplicates one
  at a time.
- Fixed: a watch folder could skip a new picture as already imported when only
  a sample of its bytes matched a different picture. A file is now skipped only
  when its full content is already in the library.
- Large upload imports now add their pictures as they go instead of all at the
  end, and check each group of files for duplicates in one pass. If an import
  fails partway, the pictures it already added stay in the library and the
  error says how many there are.
