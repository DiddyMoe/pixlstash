- Fixed: the picture count above the grid now matches the grid in the
  Unassigned view and in a folder in the Scrapheap. In the Unassigned view a
  stack counts once, as the grid shows it once. A folder in the Scrapheap
  counts only that folder's pictures. The Scrapheap with the Inbox or the
  Unassigned view on shows every deleted picture, as the Scrapheap alone does.
- Fixed: the Details panel now applies the same filters as the grid: models
  and LoRAs, workflows, stacks, hidden tags, shared pictures, a person's
  likeness group, reference folders, a project, a difference of sets (with the
  set you started from first) and the "No character" and "In no set" views. It
  also checks the same picture formats as the grid.
- Fixed: with hidden tags on, the sidebar's All media, Favourites,
  Scrapheap, Not in a project and Inbox counts leave out hidden pictures, as
  the grid and the Details panel do.
- Fixed: after the connection to the server drops and comes back, the sidebar
  counts and the Details panel are read again, so they no longer show numbers
  from before the drop.
- Fixed: the likeness step no longer asks for work again and again in a
  library where it has nothing to compare.
- Fixed: when an upload is cancelled or fails, the sidebar counts and the
  Details panel catch up with the pictures that changed or arrived while it
  ran, instead of keeping the numbers from before.
- Fixed: with a reference folder selected, the Inbox and the Unassigned view
  list only that folder's pictures, as the Details panel counts them, instead
  of every unassigned picture in the library.
- Fixed: sorting a folder or a reference folder by similarity to a person
  lists only that folder's pictures, as every other sort does, instead of
  pictures from across the library.
- Fixed: with stacks collapsed, filtering the grid by picture format,
  resolution, workflow or unresolved duplicates no longer hides a stack whose
  cover does not match while another picture in it does. The stack shows,
  represented by its best matching picture, and the count above the grid
  includes it.
- Fixed: a picture whose stored rating, size or other detail is damaged no
  longer stops the grid, the picture's details or a search from loading. That
  one detail shows as empty and every other picture shows as before.
- Fixed: a picture whose stored rating, size or other detail is damaged no
  longer stops a stack's pictures, a set in any order, or the grid sorted by
  likeness to a person, from loading. That one detail shows as empty.
- Fixed: the "Shared only" filter narrows the grid sorted by likeness to a
  person, and the Unassigned view when nothing is shared, instead of listing
  every picture. The count above the grid agrees.
