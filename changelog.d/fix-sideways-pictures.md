- Fixed: a photo whose camera information (EXIF) held a value PixlStash could
  read but not write back got no thumbnail, and was tagged and scored sideways.
  Such photos are now turned upright like any other.
- Fixed: the same photos also failed in other places. A resized or watermarked
  share copy skipped them as unreadable, a training dataset could leave them
  out, the full-size view of such a PNG or WebP failed to load, a selection
  could not be drawn on them in the Edit tab, and the text in them was not read.
  These all work now. A share copy that keeps the generation data leaves out
  only the camera information it cannot write back.
- Fixed: a picture whose camera information (EXIF) was too damaged to read
  failed to import, got no thumbnail, and was skipped by a resized or
  watermarked share copy. Such pictures are now shown as they are stored, as a
  web browser shows them.
- Fixed: a resized or watermarked share copy that keeps the generation data,
  the full-size view of a rotated PNG, and a picture a plugin made from a
  rotated PNG are turned upright but could still carry the rotation in their
  XMP or text data, so an app that reads it turned the picture a second time.
  They now keep that data without the rotation.
