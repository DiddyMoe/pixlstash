- Fixed: a slow ComfyUI could hold up PixlStash's run tracking and GPU
  sharing. PixlStash now gives up on the read after a few seconds and follows
  the run from ComfyUI's history instead. Switching a ComfyUI address between
  spellings with and without a trailing slash no longer forgets what
  PixlStash had measured of that ComfyUI.
- Fixed: setting a model's icon or a workflow's thumbnail no longer makes the
  rest of the app wait while the picture is processed.
- Fixed: the PixlStash MCP server's tools no longer wait for ever on a
  PixlStash that sends its answer a trickle at a time.
- Fixed: recording or clearing an answer about a workflow set naming thousands
  of models could fail with a server error. A set of more than 5,000 models is
  now refused with a clear message.
- Fixed: data nested thousands of levels deep, in a picture's ComfyUI
  workflow, in a model file or in a message to the PixlStash MCP server, could
  fail the picture's Recipe read with a server error, stop a model folder scan
  or end the MCP session. It is now treated like any other data that does not
  read.
