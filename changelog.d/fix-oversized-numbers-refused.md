- Fixed: a request naming an id or number too large for the library to store
  failed with a server error. This covered page offsets and limits, the ids of
  people, projects, pictures, stacks and import folders, and training
  settings. Such a request is now refused as a bad request. A share link scoped
  to one set or picture that names such an id is refused like any other
  picture it may not see.
