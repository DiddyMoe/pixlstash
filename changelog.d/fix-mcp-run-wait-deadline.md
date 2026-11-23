- Fixed: The MCP `run_workflow` tool now stops waiting for its runs once its
  `wait_seconds` is up, even when PixlStash is slow to answer. It could take
  minutes longer, and every other request from the AI client waited behind it.
