- Fixed: keys pressed while a dialog or a menu is open no longer reach the
  picture grid behind it. Delete or Backspace in a confirmation such as
  "Delete forever?", or Delete or Enter on a menu item, could delete or open
  the selection underneath, and Page Up/Down, Home, End and the search and
  shortcut keys moved the grid instead of the dialog. In the lightbox, Delete
  or an arrow key in one of its menus no longer deletes or changes the
  picture, and on the Duplicates page a key in a menu no longer stacks or
  skips the group behind it, and Escape closes the menu.
- Fixed: deleting or moving pictures while a large view was still loading
  could leave gaps or repeat pictures further down the grid.
- Fixed: after a delete far down a large view, the pictures on screen no
  longer turn into blank tiles while the view reloads.
- Fixed: scrolling into a large view while it was still loading could leave
  some pictures as blank tiles until the view was reloaded. Their thumbnails
  now appear as soon as the pictures arrive.
- Fixed: removing pictures above the part of the grid you are looking at, or
  a large view catching up after a delete, no longer jumps the view or makes
  the pictures on screen fade in again. The pictures on screen keep their
  places, or move along their row by as few places as the removal allows.
- Fixed: a search opened after the grid reloaded (after a delete, for
  example) no longer opens scrolled to where the grid was before.
- Fixed: pressing Enter to confirm Japanese, Chinese or Korean input in a
  dialog's text field no longer accepts the dialog.
- Fixed: in search results, if you scrolled while the grid was refreshing
  (after a delete or an edit, for example), the view jumped back to where it
  was when the refresh began. It now stays where you scrolled.
- Fixed: after the connection to the server drops and comes back, the grid
  catches up with what changed meanwhile, in All media, the Scrapheap, search
  results and the other views you browse. Pictures that left the view leave
  the grid without moving the pictures you are looking at, and new or
  re-sorted pictures are offered on the "new pictures - Load" and "View
  changed externally - Refresh" buttons instead of staying out of sight until
  the next reload.
- Fixed: pressing Escape to cancel Japanese, Chinese or Korean input no longer
  closes the dialog, and Enter or Escape while composing such input in a
  search, tag or rename field no longer runs the search, adds the tag, or
  saves or cancels the rename.
