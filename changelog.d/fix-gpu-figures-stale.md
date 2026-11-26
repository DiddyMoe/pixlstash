- Fixed: the Tasks panel could stop updating for several seconds at a time
  while a connected ComfyUI was not answering. It now shows the last known GPU
  state instead of waiting, and the task list and its Run anyway confirmation
  say when the GPU memory figures may be out of date, instead of showing old
  figures as current.
