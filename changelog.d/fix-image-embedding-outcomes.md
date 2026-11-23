- Fixed: a picture that could not be read for a moment (too many open files, a
  drive that dropped out, low memory) was marked as an image that cannot be
  decoded, and similarity indexing gave up on it. Similarity indexing now gives
  up only on a file that reads but cannot be decoded, and tries the others again
  later.
- Fixed: background similarity indexing could keep retrying the same failing
  pictures every few seconds. Each picture that fails is now held back, for a
  little longer after every failure, while the rest of the library carries on.
- Fixed: one picture whose loading never finished (a stalled network share, a
  pathological file) could stop similarity indexing, and every other GPU-backed
  background job, indefinitely. The other jobs now carry on after a limited
  wait. Similarity indexing starts no new batch until that file's loading
  finishes, or PixlStash restarts, and then retries the picture.
- Fixed: the background progress log counted pictures that failed as done.
