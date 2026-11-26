- Fixed: restoring a snapshot while background work was running (tagging,
  captioning) could let that work keep writing into the library during the
  restore. A restore now pauses the work PixlStash does on its own first, and
  it picks up again afterwards. If something will not stop within 30 seconds,
  the restore does not start, changes nothing, and tells you what was still
  running so you can try again. Work you started yourself is never stopped:
  while an import, a re-detect, a tag-health rebuild or another action you
  started is still queued or running, the restore does not start, changes
  nothing, and tells you what is still going.
- Fixed: work that started before a full restore, such as reading an edited
  copy back into the library, could finish writing after the restore and
  change the restored library. A restore now waits for that work to finish
  first, and pauses new work until it is done; work that was paused runs again
  afterwards. If something is still using the library after 30 seconds, the
  restore does not start and changes nothing.
- Fixed: restoring a library while a tag health rebuild was waiting to start
  could leave the tag board's rebuild button stuck as busy until PixlStash was
  restarted.
- Fixed: after a library restore, a training run's previews could go without
  their likeness until PixlStash was restarted.
- Fixed: a LoRA training run that finished while a backup was being restored or
  the library was being switched is now added to the model shelf once the
  library is back, instead of failing.
