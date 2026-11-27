- Fixed: a requested duplicate scan never started while any picture without a
  duplicate fingerprint had a file PixlStash could not read, such as one in a
  reference folder on an unplugged drive. Such a picture is now skipped until
  the next restart, and the scan goes ahead.
