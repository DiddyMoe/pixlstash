- Fixed: searching inside the Scrapheap searched your live library instead,
  and exporting from the Scrapheap failed without packaging anything. Both now
  work on the Scrapheap's pictures, as do its statistics and likeness groups.
- Fixed: the Details panel could show the statistics of the view you had just
  left, and did not update when pictures were added or removed. It now shows
  the view on screen, refreshes as pictures come and go, and says "No pictures
  in this view" for an empty one.
- The Details panel now says why pictures are untagged: waiting for the
  tagger, couldn't be read, or no tags apply. Click a reason to narrow the
  grid and its statistics to those pictures, in every view including the
  Inbox; the reason shows as a filter chip you can clear. Through the API,
  the picture list can be narrowed to one of these reasons in the same way.
- Fixed: the tagging progress could stay short of done for ever, because it
  counted pictures the tagger never picks up (in the Scrapheap, locked, or
  unreadable).
- Fixed: when a count in the sidebar or the project dropdown could not be
  refreshed, it kept showing the old number as if it were current. It is now
  underlined, says "Couldn't refresh this count" when you point at it, and is
  retried. Large counts now have thousands separators.
- Deleting a project, person or picture set from the sidebar now asks in
  PixlStash's own dialog, which names the action on its button and says what
  is kept: the pictures stay in your library.
- Fixed: the background task progress counted pictures in the Scrapheap, so a
  library of 500 pictures with 1,000 in the Scrapheap showed 1,500 pictures for
  every task. Progress now counts only your library.
- Face detection and most other background work no longer spend time on
  pictures in the Scrapheap. A picture you restore is picked up where it left
  off.
- Fixed: one picture whose file couldn't be read could stop likeness grouping
  from ever running, because likeness waited for that picture's details, and
  those need a quality score it can never get. Likeness now waits only for
  pictures that can be finished. It also no longer compares pictures in the
  Scrapheap; a picture you restore is compared then.
- Fixed: a search ignored most of the grid's filters, such as the star
  rating's upper limit, the resolution, the folder and the workflow, and
  searched past them. A search now narrows by every filter the grid does, and
  refuses a filter value the grid refuses.
- Fixed: through the API, some unusual options on the picture list caused a
  server error, and one of them could add Scrapheap pictures to the live list.
  Unknown options are now ignored and the live list never holds Scrapheap
  pictures.
