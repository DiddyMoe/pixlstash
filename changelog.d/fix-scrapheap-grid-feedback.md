- Fixed: after Delete forever, the Scrapheap's grid went blank for a moment
  and reloaded every thumbnail. The pictures still there now stay on screen
  while the grid catches up, and the deleted ones leave as the server confirms
  them.
- Fixed: clicking Delete forever or Empty scrapheap a second time while
  PixlStash was still working out what would be destroyed could do nothing, or
  open the dialog on the wrong pictures. The bar now says "Checking what would
  be destroyed…", and a second click is told the first check is still running.
- Fixed: deleting pictures while a large view was still loading stopped the
  rest of it from loading. The view now keeps filling in.
- Fixed: when restoring a picture from the lightbox failed, the picture
  vanished from the Scrapheap and the lightbox closed anyway. Both now stay, so
  you can try again.
- Restore all now asks in PixlStash's own dialog and says how many pictures
  will return to your library. Screen readers announce PixlStash's
  confirmation dialogs as alerts and read out their question.
- Large counts in the Delete forever dialog, the new-pictures pill, the grid's
  notices and the locked-set messages now show thousands separators.
- Fixed: the receipt after Keep cover only or Keep recipes only ended in a
  stray count ("… to the Scrapheap · 13").
- Fixed: stepping through the lightbox while the grid was still loading could
  leave it blank, with the arrow keys doing nothing. It now waits for the next
  picture and shows it.
- Fixed: restoring pictures while the Images or Videos filter was on could
  shift cards one place out, and restoring many pictures into a large library
  held the page for seconds.
- Fixed: with the Images or Videos filter on, opening a stack showed members
  of the other kind.
- Fixed: after deleting or restoring a picture in the Scrapheap, the grid
  showed only one picture of each stack, hiding the others until the view
  reloaded. Every scrapped picture now stays on screen.
