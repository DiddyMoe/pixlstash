- Fixed: stopping the PixlStash server reloaded every open tab onto an error
  page. Tabs now wait for the server and reconnect; only a database restore
  reloads them.
