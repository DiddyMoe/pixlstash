- Fixed: a ComfyUI, a model host or the plugins repository that answers
  slowly, without end or with a redirect no longer holds PixlStash up or fills
  its memory. Every such request now has a time limit and a size limit. When
  ComfyUI sends something PixlStash cannot use, you see a short message saying
  so instead of ComfyUI's raw reply.
- Fixed: a ComfyUI address is now saved in one spelling, so an address written
  with or without "http://" or a trailing slash is the same connection. An
  address with a user name, a query or a stray backslash is refused when you
  save it. If an address saved by an older version is no longer accepted,
  PixlStash sends it nothing and asks you to set the address again in
  Settings.
- Fixed: model downloads follow a redirect only to another public https
  address, and a download that runs past the size it gave is stopped.
- Fixed: installing plugins from the command line writes the plugins
  repository to disk as it arrives, stops after 10 minutes or 256 MB, and
  refuses a redirect.
- Fixed: a plugin archive can no longer fill the disk or memory while
  `pixlstash-cli plugins install` unpacks it. An archive with too many
  entries, an oversized index or more than 512 MB of contents is refused, and
  whatever was partly unpacked is removed.
- Fixed: a package from outside PyPI no longer reads as an ordinary PyPI
  package in the list `pixlstash-cli plugins install` asks you to approve. A
  plugin whose requirements hide a host in an address, behind a user name or
  a backslash or spelled so that programs read it differently, is refused
  before anything is downloaded, and the message names the file and line.
  When the requirements name another package index or a page of links, each
  package in the list says it may come from there. A package fetched over
  plain http, from a file, over FTP or from a code repository is listed with
  its address.
- Fixed: when `pixlstash-cli plugins install` cannot download the plugins
  repository, it says in one sentence whether the proxy, the secure
  connection, the connection or the wait failed, instead of printing the raw
  error. Other install errors say what went wrong in plain words and name the
  file or line.
- Fixed: when a redirect is followed, PixlStash connects to the address it
  checked, so a host whose address changes in between cannot steer the
  request into your network. The certificate is still checked against the
  host's name.
- Fixed: a model download, and the aesthetic scorer download, now check the
  link itself as well as each redirect. The link must be https to a public
  address, and PixlStash connects to the address it checked.
- Fixed: a model download from a ComfyUI template can no longer be sent to a
  host outside Hugging Face by a redirect whose address hides that host behind
  a backslash.
- Fixed: the optional telemetry ping has the same time and size limits and
  follows no redirect.
- Fixed: every part of PixlStash that talks to ComfyUI now checks a saved
  address first, including the model shelf, model folders, merging model
  copies and training. A request refused because the saved address is no
  longer accepted shows a notice with an Open Settings button that goes
  straight to Settings > ComfyUI, and that page names the connection to set
  again. Training and the other places that failed because of it say what
  failed and leave the reason to that one notice. Training's background
  check logs the problem once instead of on every poll.
- Fixed: the connection PixlStash keeps open to ComfyUI for a run's step
  progress no longer follows a redirect. If ComfyUI's address answers it with
  one, PixlStash refuses to connect to wherever it points; the run still
  finishes and its pictures still arrive, only without step progress in the
  Tasks tab.
- Fixed: cancelling a model download now stops it straight away, even when
  the link has stopped sending or sends only a trickle, and the file can be
  downloaded again at once. A download cancelled while it was still
  connecting takes a few seconds to stop, and trying it again meanwhile says
  so. A download that brings in almost nothing for two minutes is given up as
  failed instead of staying in progress for good.
- Model downloads now run three at a time. Any more wait in the Tasks tab as
  Queued, where each can be cancelled before it starts, and a download button's
  tooltip says it is queued.
- Fixed: moving the model store, the face recognition models or the engines'
  folder, and moving or removing a folder inside a model folder, are refused
  while a model is downloading into it, instead of losing the download or
  leaving its partial file behind. A download into a folder that is being
  moved or removed is refused until that is done.
- PixlStash now needs urllib3 2.8.0 or later, which fixes advisories in
  earlier versions, among them one where an answer could make it use
  unbounded memory.
