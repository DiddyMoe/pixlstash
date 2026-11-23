- Fixed: when automatic captioning could not run at all (the captioning model
  failed to load, or the service it calls was unreachable), every picture in
  the batch was left with an empty description and never captioned again.
  Those pictures now keep their description and are tried again later, less
  often after every failure.
- Fixed: a picture whose file could not be opened for a moment (a network share
  that dropped out, an unplugged drive) lost its caption for good. It is now
  captioned once the file can be read again.
- Fixed: a caption you typed, or a request to caption a picture with another
  model, could be overwritten by a background captioning batch that was already
  running. Your change is now kept.
- Fixed: pictures in the Scrapheap were still sent for automatic captioning,
  and one moved there while it was being captioned got the caption anyway.
