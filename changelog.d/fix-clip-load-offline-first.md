- Fixed: PixlStash contacted Hugging Face every time it loaded the model behind
  search and picture embeddings, even when that model was already downloaded.
  On a slow or blocked network each load could wait for that request first.
  A downloaded model now loads from disk without going online; only the first
  download needs the network.
