- Fixed: when PixlStash's database was busy, a training run could lose track
  of where it was. A run that started could keep training while it still said
  "Queued"; a run that ended could stay "Training" until the next restart; a
  run whose LoRA reached the model shelf then, or just before PixlStash quit
  unexpectedly, could show as failed after the next start ("could not be
  imported" or "no final checkpoint"); Train longer could say the run was
  queued although it never trained; and starting a run that may quit ComfyUI
  could lose that permission, so the run waited for the GPU as though you had
  said no. These changes are now saved again after a short
  wait. When one still cannot be saved, the run is stopped and says why, and
  starting it, Train longer, or stopping a queued run reports an error instead
  of success.
- Fixed: stopping a training run at the moment it finished could leave the
  finished run showing "Stopping" until the next restart. A stop now changes
  only a run that is still training.
- Fixed: a training step whose loss was not a number made the run's latest
  loss disappear and dipped its loss chart to zero. Those steps are now left
  out of the curve.
