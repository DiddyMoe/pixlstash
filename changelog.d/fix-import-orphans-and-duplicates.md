- Fixed: when saving an import failed partway, whether from a watch folder, an
  upload, a plugin result or a ComfyUI generation, its files stayed in the
  library folder. The next scan then added them as pictures without their tags
  or captions. Those files are now removed when the save fails.
- Fixed: a watch-folder file that failed to import, or that an import stopped
  before reaching, was not offered again until the file changed. It is now
  picked up again on a later scan.
- Fixed: a plugin, edit or ComfyUI result identical to a picture in the
  Scrapheap was added again beside it, and one whose matching picture was moved
  to the Scrapheap while the result was being saved left that picture there.
  Such a result now brings the Scrapheap picture back. Two identical results
  from one plugin or edit run now become one picture.
- Fixed: a plugin result that only resembled an existing picture could be
  treated as a duplicate of it and dropped. A result now counts as a duplicate
  only when its whole content matches.
- Fixed: a ComfyUI generation that matched a picture deleted for good while the
  run was being saved was listed under an id that no longer exists. It is now
  added as a new picture.
- Improved: an import no longer holds up other imports while it reads and
  copies a large file. Only the final duplicate check and save wait their turn.
