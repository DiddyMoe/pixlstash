- Fixed: exporting a large selection to a ZIP no longer fails. The selected
  pictures are sent in the request itself rather than its address, so a
  selection of thousands of pictures exports in full. Through the API, the
  export now takes the picture ids in the request body.
- Fixed: an export now packages the pictures in the order the grid shows them,
  with a stack's cover in the stack's place.
- Fixed: exporting a search exports exactly the pictures the search shows, in
  its order. Filters such as a minimum score now apply to the export as they do
  to the search, and a search in the Scrapheap exports the Scrapheap. The export
  used to run a search of its own (with "A photo of" put in front of your
  words), so it could package pictures the search did not show; the same
  search may now export a different set of pictures than it did before.
- Fixed: exporting a selection made in the Scrapheap exports only those deleted
  pictures, and exporting a set follows the filters and sort you have on.
- Fixed: exporting a large selection to a folder no longer fails. As with a
  ZIP, the selected pictures are sent in the request itself, and a selection
  made in the Scrapheap exports only those deleted pictures.
- Fixed: with a search narrowed to "In text", an export packages only the
  pictures whose text matches, as the grid shows them. Turning the switch on
  or off now asks the server again rather than filtering the results already
  shown.
