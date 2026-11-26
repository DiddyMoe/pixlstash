- Fixed: actions over a very large number of pictures no longer fail with an
  error: searching within a selection, asking which of a selection are shared,
  assigning or removing the faces of a large selection, moving a selection to
  another library, filtering the grid by a very large number of sets, training
  suggestions for a person with many stacks, renaming a reference folder with
  many files, re-adding a folder whose files were removed before, the
  thumbnail labels of a large batch, restoring a large Scrapheap, sorting a
  big set by smart score, library statistics for a large set or person,
  likeness search, tag suggestions and reviews over a large scope, unstacking
  a very large stack, stacking a large duplicate group and the start-up
  cleanup of missing files. On most systems this took tens of thousands of
  pictures; on some older Linux systems about a thousand were enough.
- Fixed: imports, runs and restores with very many pictures no longer fail
  with an error: a plugin run on a very large number of pictures, a single API
  import of very many files, a ComfyUI run with very many outputs, opening,
  re-running or reading the settings of a workflow run with very many
  different models or input pictures, restoring a picture set with very many
  members, a full restore that drops many permanently deleted pictures, and
  comparing many pictures with a snapshot.
- Fixed: a search for one person's pictures without the others' no longer
  comes back empty when the request names that first person twice.
- Fixed: the People list and the project list no longer fail to load in a
  library with very many people or projects.
