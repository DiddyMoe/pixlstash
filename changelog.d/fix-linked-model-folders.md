- Fixed: on Linux and macOS, models inside a linked folder in a model folder
  (a checkpoints folder that links to another drive, say) were missing from
  the Models shelf, although ComfyUI loads them. The shelf now follows such
  links as ComfyUI does. A folder reached twice, or one that is already a
  model folder of its own, is still listed only once.
