- Fixed: saving a setting with a value it cannot hold (a word for a column
  count, a number too large to store) could fail with a server error or save a
  broken value. It is now refused with a message naming the setting.
