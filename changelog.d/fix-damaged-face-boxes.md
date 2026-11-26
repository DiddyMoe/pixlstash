- Fixed: a picture with a damaged face or object box saved in its library
  failed its face list, its object list, its thumbnail, a person's avatar, an
  export, its score, a rotate, and giving it a person. A person's training set
  could also fail to load its suggested pictures. The damaged box is now left
  out (a rotate turns the other boxes and leaves it as it was, and a person
  goes to a face whose box can be read), it is noted once in the log, and the
  picture is no longer retried on every pass.
- Fixed: a picture whose face box was saved in an older format no longer stops
  the tagger partway through a batch.
