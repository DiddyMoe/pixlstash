- Fixed: A batch of ComfyUI runs no longer has each run ask ComfyUI how it is
  doing every second. The runs share one look at ComfyUI's queue, so a long
  batch asks no more often than a single run.
- Fixed: A ComfyUI run's pictures, prompt and results now pass straight
  between PixlStash and ComfyUI, never through a proxy set in your environment
  and never by following a redirect to another address.
