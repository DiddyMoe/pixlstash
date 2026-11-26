- Fixed: the server could hand out files from a folder beside its own web
  page files whose name begins the same way to anyone who could reach it,
  without signing in. It now serves only its own page files, and never one
  reached through a link placed among them.
