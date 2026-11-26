- Fixed: error messages could repeat the value that was sent, quote an
  internal error, or show server folder paths (including another library's),
  database error text or free-space figures. They now give a plain sentence,
  and the server log keeps the full reason. Where something failed for a
  reason only the server can see, the message says "The server log says why."
  This covers the library screens and library switching, imports and import
  folders, stacks, moving pictures to another library, ComfyUI connections and
  runs, node packs, the model shelf, model folders and model downloads, folder
  mapping, training, the duplicates queue, reference folders, picture sets,
  tag groups imported from .xlsx or .csv, the library layout, and image
  plugins.
- Changed: a refused request's error list no longer repeats the values that
  were sent. Where a field has a limit, the error now states the limit, such as
  "at most 5". A refused workflow change or model swap names the entry by its
  position in the list, an unrecognised sort or matching mode is refused
  without repeating it, and a folder name Windows reserves is refused with the
  list of reserved names.
- A failed import now says what went wrong in a plain sentence, such as "Not a
  picture or video this library can read." or "The file could not be written
  to the library.", and a failed import or model folder scan in the task list
  says that the task failed.
- ComfyUI messages no longer quote what was sent. Checking an address that
  nothing answers at says "Nothing answered at that address. Check that
  ComfyUI is running and that the port matches the one in its console.", an
  address another connection already has is "Another connection already has
  that address.", and a node pack PixlStash does not list is "PixlStash does
  not know that node pack." A run whose pictures cannot be filed because of a
  library fault reads "PixlStash could not import this run's outputs. The
  server log says why.", and a run that skips a LoRA it cannot take out names
  the node as PixlStash knows it, without quoting file names from the
  workflow.
- When an image plugin fails, the notice reads "The plugin failed; the log
  says why." instead of the plugin's internal error, and a plugin's report
  about one image shows only its own message.
