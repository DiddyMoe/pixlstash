- Fixed: the duplicate queue's counts now match what the button next to them
  acts on. Opened on a person, set or project, the queue's headline, its tier
  menu and the *Auto-stack* button count that scope's matches, not the whole
  library's.
- Fixed: a duplicate group you had already decided on could come back after a
  rescan, or after some of its pictures went to the Scrapheap. A decision now
  stays until a new copy joins the group.
- Fixed: *Auto-stack* could fail a group whose suggested cover had been sent to
  the Scrapheap after the preview, and its report could count fewer pictures
  than the preview said it would stack.
- Fixed: after a duplicate verdict whose outcome was unknown, the queue said it
  had been reloaded even when the reload failed. It now says the reload failed
  and asks you to reload the page.
- Fixed: in the duplicate queue, scrolling for more groups after a reload could
  load nothing, the scan banner could jump back to "complete" while a scan was
  running, and a bulk *Stack* or *Keep separate* could be sent while another
  verdict was still saving.
- Fixed: when a duplicate scan finished while you were partway through the
  queue, the queue jumped back to the top and dropped your selection. The
  scan's last finds now join the list in place, and the group you are on and
  your selection stay.
- Fixed: a picture whose stored likeness fingerprint was damaged could pair as
  a near duplicate, score as identical in likeness, or drop every other
  candidate from a likeness comparison. Such a picture is now treated as not
  yet fingerprinted.
- Stopping a duplicate scan takes effect sooner on a large library.
- *Keep cover only* and stacking several duplicate groups at once are faster.
