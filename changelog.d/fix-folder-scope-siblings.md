- Fixed: a folder no longer takes in the folders whose names merely start with
  its own. Filtering the grid or the stats sidebar by "photos", or finding
  duplicates in it, used to include "photos-old" and "photos2" as well; on
  Linux, and on a Mac drive formatted as case-sensitive, it also included
  "PHOTOS", which is a different folder there.
- Fixed: "About your library" could refuse to compare the duplicates of folders
  that share a parent when a neighbouring folder had a similar name.
- Fixed: when part of a model folder cannot be read, only that part is marked
  unreachable, not neighbouring folders with similar names or another case.
