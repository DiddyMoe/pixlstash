- Fixed: search could fail on the first query after start-up. When the first
  search arrived while PixlStash was still preparing pictures in the
  background, both could start loading the AI libraries at the same moment,
  and the search answered with an error. The libraries now load one at a time
  on first use.
