- Fixed: on a large library, Auto-stack in the duplicate queue could spend over
  a minute working out what it would do and then give up. It now answers
  quickly, and when it cannot it says why rather than offering to create 0
  stacks. It also stacks the scope it showed you, even if the queue moved to
  another one while the dialog was open.
- Fixed: picking one snapshot to restore, going back and quickly picking
  another could show the first one's figures above a Restore button that
  restores the second.
- Fixed: the library layout dialog could show a count for a layout or sweep you
  had already changed, and let Move them now go ahead on it.
