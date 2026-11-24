- Fixed: on a large duplicate queue, Auto-stack could stop answering partway
  through ("Auto-stack did not answer") and leave most groups unstacked. It now
  runs in the background: a banner above the queue counts the groups as it
  stacks them, and you can stop the run or undo all of it from there. Leaving
  the screen or reloading the page does not stop it, and Ctrl+Z steps back
  through it a few hundred groups at a time.
- Fixed: Redo after several Undos in a row replayed the changes in the wrong
  order, so a later change to the same pictures could be rolled back while
  History showed it as redone.
