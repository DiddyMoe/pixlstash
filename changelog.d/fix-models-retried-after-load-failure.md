- Fixed: when the tagger could not be loaded, or a tagging plugin was missing a
  package it needs, pictures could come out untagged and were never tagged
  again. They now stay waiting for tags and are tagged once the tagger loads or
  the package is installed. Meanwhile the tagger is no longer loaded from disk
  again for every batch of pictures: it waits between attempts, up to ten
  minutes, and the reason is in the server log.
- Fixed: when the aesthetic scoring model could not be downloaded or loaded
  (for example while offline), pictures were scored without an aesthetic score
  until the app was restarted. It is now tried again after a short wait, and
  scoring resumes on its own once it loads.
