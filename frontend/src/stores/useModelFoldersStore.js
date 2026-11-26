import { computed, onScopeDispose, ref } from "vue";
import { defineStore } from "pinia";

import {
  createModelFolder,
  createModelFolderDir,
  DOWNLOADS_KIND,
  forgetModelFolder,
  getComfyuiReads,
  listModelFolderDevices,
  listModelFolderDirs,
  listModelFolders,
  MANAGED_KIND,
  renameModelFolder,
  setDefaultModelFolder,
  setDownloadModelFolder,
  SOURCE_KIND,
} from "../api/modelFolders";
import { onSessionReset } from "../utils/apiClient";
import { errorDetail } from "../utils/apiError";
import { isBehindLink } from "../utils/modelShelf";
import { useModelShelfStore } from "./useModelShelfStore";
import { useNoticeStore } from "./useNoticeStore";

/** How often the list is re-read while a folder is being scanned. */
const POLL_MS = 3000;

/** The `scan_status` values that mean a scan is queued or walking. */
const SCAN_IN_FLIGHT = new Set(["pending", "running"]);

/** Where ComfyUI loads from, before it is read or when it cannot be. */
const NO_COMFYUI_READS = Object.freeze({ known: false, reads: [] });

/**
 * How many listings `reloadDirs` re-reads at once. Each read walks a directory
 * on disk, possibly on a slow or offline mount; a conservative bound.
 */
const DIRS_READ_SLOTS = 4;

/** How many listings nothing shows are kept before the oldest are dropped. */
const DORMANT_LISTINGS = 256;

/**
 * The registered model folders, and the scans running against them.
 *
 * Nobody starts a scan: the server scans a folder when it is added, whenever a
 * model file in it changes, and every ten minutes besides, and announces what
 * arrived as `models_arrived`, which the shelf listens for. This store only
 * shows it: a folder is scanning while its `scan_status` says so, and the list
 * is re-read until none is. It is a STORE rather than dialog state because the
 * notice for a folder just added must still fire when its scan lands after
 * the dialog has closed.
 *
 * The list is owner-only host data (absolute paths on this machine), so a
 * session reset drops it whole. Nothing here is persisted.
 */
export const useModelFoldersStore = defineStore("modelFolders", () => {
  const folders = ref([]);
  /** One entry per DRIVE the folders sit on, for the shelf's bands. */
  const devices = ref([]);
  const loading = ref(false);
  const error = ref("");
  /** True once a read has completed, so "none registered" and "not asked" differ. */
  const loaded = ref(false);

  /** Folders added this session whose first scan has not been announced yet. */
  const announcing = new Set();

  let pollHandle = null;

  /**
   * Every read this store has out, by what it reads: `"folders"`, `"devices"`,
   * `"comfyui"`, or a listing's `dirsKey` (always `"<id>:<path>"`, so the two
   * kinds never collide). See `readOnce`.
   */
  const reads = new Map();

  /**
   * Run `start` as the one read of `key`, so two reads of the same thing are
   * never out at once and an older answer never lands after a newer one.
   *
   * With `join`, a caller shares the read already out: right for a mount or a
   * poll, which needs an answer, not one newer than anything. Without it (the
   * default, for a caller that has just changed something) the read out may
   * have left before the change, so it stops being current, which keeps its
   * answer off the screen, and one more read starts once it settles. Every
   * caller in between shares that one, and every promise handed out resolves
   * to the newest answer.
   *
   * @param {string} key - what is read.
   * @param {(isCurrent: () => boolean) => Promise<*>} start - issues the read
   *   and publishes its answer only while `isCurrent()`. Must not reject.
   * @param {{join?: boolean}} [options]
   * @returns {Promise<*>} what the newest read resolved to, or null for a read
   *   a session reset dropped.
   */
  function readOnce(key, start, { join = false } = {}) {
    const out = reads.get(key);
    if (!out) return launch(key, start);
    if (join) return out.promise;
    if (!out.next) {
      out.current = false;
      out.next = out.settled.then(() =>
        out.dropped ? null : launch(key, start),
      );
    }
    return out.next;
  }

  /** Start one read of `key`; `readOnce` decides when. */
  function launch(key, start) {
    const read = { current: true, dropped: false, next: null };
    // Registered before `start` runs, so what it writes first already counts
    // this key as being read.
    reads.set(key, read);
    read.settled = start(() => read.current).finally(() => {
      // Identity-checked: a read a reset dropped must not unregister the one
      // that replaced it. One with a read to follow stays until that starts.
      if (reads.get(key) === read && !read.next) reads.delete(key);
    });
    read.promise = read.settled.then(
      (answer) => read.next ?? (read.dropped ? null : answer),
    );
    return read.promise;
  }

  /** Abandon every read out: none may publish, and none is shared again. */
  function dropReads() {
    for (const read of reads.values()) {
      read.current = false;
      read.dropped = true;
    }
    reads.clear();
  }

  const scanningIds = computed(
    () =>
      new Set(
        folders.value
          .filter((folder) => SCAN_IN_FLIGHT.has(folder.scan_status))
          .map((folder) => folder.id),
      ),
  );

  /**
   * folder id → the drive it sits on.
   *
   * The shelf bands by this rather than by path prefix: a bind mount and a
   * symlinked folder look like different drives by path and are one, and two
   * folders under one root can be different drives when a mount sits between
   * them. The server measured it; the client must not re-derive it.
   */
  const deviceByFolderId = computed(() => {
    const byId = new Map();
    for (const device of devices.value) {
      for (const folderId of device.folder_ids || []) {
        byId.set(Number(folderId), device);
      }
    }
    return byId;
  });

  /** folder id → its row name, told apart from its namesakes. */
  const labelByFolderId = computed(() => folderLabels(folders.value));

  /** The one folder that is PixlStash's own storage. Always exactly one. */
  const managedFolder = computed(() =>
    folders.value.find((folder) => folder.kind === MANAGED_KIND),
  );

  /**
   * The registered ai-toolkit output root, or `undefined` while none is set.
   *
   * Singular on purpose. The server permits several `source` folders, but
   * ai-toolkit writes every run under one output root, so a second would be a
   * setting nobody asked for: the offer to set one disappears once this exists,
   * and the training-runs view reads exactly this folder. `find`, so a registry
   * that somehow holds two still resolves rather than reporting an error the
   * owner has no control to fix.
   */
  const sourceFolder = computed(() =>
    folders.value.find((folder) => folder.kind === SOURCE_KIND),
  );

  /** The watched Downloads folder, or `undefined` while none is set. */
  const downloadsFolder = computed(() =>
    folders.value.find((folder) => folder.kind === DOWNLOADS_KIND),
  );

  /** Where a new model goes when nothing else is named. */
  const defaultFolder = computed(() =>
    folders.value.find((folder) => folder.is_default),
  );

  /** Where a model PixlStash downloads goes; the default folder until chosen. */
  const downloadFolder = computed(() =>
    folders.value.find((folder) => folder.is_download_target),
  );

  /**
   * The folders that can be the default: the ones the shelf catalogues and
   * writes into. Never a taken-from folder, never PixlStash's own `foreign`.
   */
  const defaultCandidates = computed(() =>
    folders.value.filter(
      (folder) => folder.kind === "user" || folder.kind === MANAGED_KIND,
    ),
  );

  /** Paths already registered, for the picker to disable rather than 409 on. */
  const registeredPaths = computed(() =>
    folders.value.map((folder) => folder.path).filter(Boolean),
  );

  /**
   * One level of subfolders per `"<folder id>:<path>"`: `{loading, listing,
   * error, stale}`. Shared by the Folders trees and the folder view, so a
   * folder shown in both is read once. Entries are written in place (a
   * reactive Map tracks `get`, `set` and iteration), never by copying the map.
   * `stale` marks a listing a change on disk may have outdated while nothing
   * showed it: it is still drawn, and re-read on its next use.
   */
  const dirListings = ref(new Map());

  /**
   * The listings something on screen shows: owner → its keys, and key → how
   * many owners show it. Only these are re-read when a change lands.
   */
  const dirsShownBy = new Map();
  const dirsShown = new Map();

  /** Re-reads waiting for a slot. Replaced on a session reset. */
  let dirsQueue = { waiting: new Set(), running: 0 };

  /**
   * The sidebar tree's open rows, kept here so they outlive the sidebar: a
   * trip to another view and back finds the tree as it was left. Expanded
   * folder keys, and the drives closed (drives start open).
   */
  const treeExpanded = ref(new Set());
  const treeClosedDrives = ref(new Set());

  /** The cache key for one listing. */
  function dirsKey(id, path = "") {
    return `${Number(id)}:${path}`;
  }

  /** The folder id and path a `dirsKey` names. */
  function splitDirsKey(key) {
    const at = key.indexOf(":");
    return { id: Number(key.slice(0, at)), path: key.slice(at + 1) };
  }

  /**
   * Whether a directory is a link (a symlink or junction to a directory, which
   * the scanner follows) or lies behind one, by the listings read so far. The
   * server refuses every move, rename and new folder through a link, so the
   * tree and the sidebar offer none.
   *
   * @param {number} id - the registered folder.
   * @param {string} [path=""] - POSIX, relative to it.
   * @returns {boolean}
   */
  function behindLink(id, path = "") {
    return isBehindLink(dirListings.value, id, path);
  }

  /**
   * Read one level of a registered folder: once, unless `force` or the cached
   * listing is stale.
   *
   * A read of the same level already out is shared. A forced read (asked for
   * after a change in the folder) cannot use one that may have left before
   * the change, so it waits for that one and reads once more: see `readOnce`.
   *
   * @param {number} id - the registered folder.
   * @param {string} [path=""] - POSIX, relative to it.
   * @param {{force?: boolean}} [options]
   * @returns {Promise<Object|null>} the listing, or null when it could not be
   *   read (a remote owner is refused the route; the tree then offers no
   *   subfolders rather than an error per row).
   */
  async function loadDirs(id, path = "", { force = false } = {}) {
    const key = dirsKey(id, path);
    const known = dirListings.value.get(key);
    if (!force && known?.listing && !known.stale) return known.listing;
    return readOnce(key, (isCurrent) => readDirs(id, path, key, isCurrent), {
      join: !force,
    });
  }

  /** One read of one level, written to `dirListings` while it is current. */
  async function readDirs(id, path, key, isCurrent) {
    const before = dirListings.value.get(key);
    // The listing already drawn stays up while it is re-read.
    putDirs(key, {
      loading: true,
      listing: before?.listing ?? null,
      error: "",
      stale: Boolean(before?.stale),
    });
    try {
      const listing = await listModelFolderDirs(id, path);
      if (isCurrent()) {
        putDirs(key, { loading: false, listing, error: "", stale: false });
      }
      return listing;
    } catch (err) {
      console.warn(`Could not list model folder ${id} at "${path}"`, err);
      if (isCurrent()) {
        putDirs(key, {
          loading: false,
          listing: null,
          error: errorDetail(err) || "Could not read this folder.",
          stale: false,
        });
      }
      return null;
    }
  }

  /**
   * Write one listing in place. A new key drops the oldest listings nothing
   * shows or reads, so that counting it there are at most `DORMANT_LISTINGS`:
   * browsing a large tree does not grow the cache without end. Only a read
   * writes here, so the new key itself is never dropped.
   */
  function putDirs(key, entry) {
    const listings = dirListings.value;
    const added = !listings.has(key);
    listings.set(key, entry);
    if (!added) return;
    const dormant = [];
    for (const known of listings.keys()) {
      if (!dirsShown.has(known) && !reads.has(known)) dormant.push(known);
    }
    // `+ 1`: the new key, being read now, is not in `dormant` but soon will be.
    for (let i = 0; i < dormant.length + 1 - DORMANT_LISTINGS; i += 1) {
      listings.delete(dormant[i]);
    }
  }

  /**
   * Say which listings `owner` shows, replacing what it said before. A listing
   * some owner shows is kept and re-read by `reloadDirs`; one that turns up
   * stale or not cached is re-read now. Each Folders tree (pickers included)
   * and the folder view registers, and releases on unmount.
   *
   * @param {symbol} owner - one per consumer.
   * @param {Iterable<string>} keys - `dirsKey`s.
   */
  function setActiveDirs(owner, keys) {
    const before = dirsShownBy.get(owner) ?? new Set();
    const now = new Set(keys);
    for (const key of before) {
      if (now.has(key)) continue;
      const owners = dirsShown.get(key) - 1;
      if (owners) dirsShown.set(key, owners);
      else dirsShown.delete(key);
    }
    for (const key of now) {
      if (before.has(key)) continue;
      dirsShown.set(key, (dirsShown.get(key) ?? 0) + 1);
      const known = dirListings.value.get(key);
      if ((!known || known.stale) && !reads.has(key)) queueDirs(key);
    }
    if (now.size) dirsShownBy.set(owner, now);
    else dirsShownBy.delete(owner);
  }

  /** `owner` shows no listing any more. */
  function releaseActiveDirs(owner) {
    setActiveDirs(owner, []);
  }

  /**
   * Re-read a listing as soon as one of `DIRS_READ_SLOTS` is free. One being
   * read then gets one more read after that one (see `readOnce`), and keeps its
   * slot until that read settles, so a trailing read counts against the bound.
   */
  function queueDirs(key) {
    dirsQueue.waiting.add(key);
    pumpDirs(dirsQueue);
  }

  function pumpDirs(queue) {
    for (const key of queue.waiting) {
      if (queue.running >= DIRS_READ_SLOTS) return;
      queue.waiting.delete(key);
      queue.running += 1;
      const { id, path } = splitDirsKey(key);
      loadDirs(id, path, { force: true }).finally(() => {
        queue.running -= 1;
        pumpDirs(queue);
      });
    }
  }

  /**
   * Where ComfyUI loads each kind from (#2205): `{known, reads}`. Unknown
   * until read, and unknown when the read fails (a remote owner is refused
   * it), so no folder is marked and no model called misplaced on a guess.
   */
  const comfyuiReads = ref(NO_COMFYUI_READS);

  /**
   * Ask once per shelf visit, and again after a rename; ComfyUI's folders
   * rarely move under it. Not shared: after a rename the answer must be newer
   * than the rename.
   */
  function loadComfyuiReads() {
    return readOnce("comfyui", readComfyui);
  }

  async function readComfyui(isCurrent) {
    let answer = NO_COMFYUI_READS;
    try {
      const body = await getComfyuiReads();
      if (body?.known && Array.isArray(body.reads)) answer = body;
    } catch (err) {
      console.warn("Could not read where ComfyUI loads models from", err);
    }
    if (isCurrent()) comfyuiReads.value = answer;
  }

  /**
   * After a change on disk (a scan, a move, an added file): re-read in place
   * the listings something shows, and mark the rest stale so each is re-read
   * when next used. A listing being read gets one more read after that one,
   * which may have left before the change. Every read this starts waits in
   * the queue, so at most `DIRS_READ_SLOTS` of them are out at a time.
   * Clearing instead would fold every open row shut.
   *
   * @param {{join?: boolean}} [options] - `join` for a mount, which brings
   *   back listings from an earlier visit rather than reporting a change: a
   *   listing already being read is left to that read.
   */
  function reloadDirs({ join = false } = {}) {
    for (const [key, entry] of dirListings.value) {
      if (reads.has(key)) {
        if (!join) queueDirs(key);
      } else if (dirsShown.has(key)) {
        queueDirs(key);
      } else if (!entry.stale) {
        dirListings.value.set(key, { ...entry, stale: true });
      }
    }
  }

  /**
   * Re-read the drive capacities.
   *
   * Its own call, and its own failure. The route stats the filesystem, so an
   * offline mount can make it slow or make it fail, and neither may take the
   * folder list down with it: without capacities the shelf still draws its
   * bands, it just cannot draw a meter in them. Always shares a read already
   * out: its callers are mounts.
   */
  function refreshDevices() {
    return readOnce("devices", readDevices, { join: true });
  }

  async function readDevices(isCurrent) {
    try {
      const next = await listModelFolderDevices();
      if (isCurrent()) devices.value = next;
    } catch (err) {
      console.warn("Could not read the model drives' capacity", err);
      // Not surfaced as `error`: that field drives the folder dialog's alert,
      // and a drive we could not measure is not a reason to tell the owner
      // their folders could not be read.
      if (isCurrent()) devices.value = [];
    }
  }

  /**
   * Re-read the registry.
   *
   * @param {Object} [options]
   * @param {boolean} [options.quiet=false] - true for a poll, which must not
   *   flash the list's loading state on every tick.
   * @param {boolean} [options.join=false] - share a read already out: for a
   *   mount or a poll. A caller that has just changed a folder leaves it
   *   false, so the answer it waits for left after the change.
   * @returns {Promise<void>}
   */
  function refresh({ quiet = false, join = false } = {}) {
    if (!quiet) loading.value = true;
    error.value = "";
    return readOnce("folders", readFolders, { join });
  }

  async function readFolders(isCurrent) {
    try {
      const next = await listModelFolders();
      if (!isCurrent()) return;
      folders.value = next;
      loaded.value = true;
      announceFinishedScans();
      if (scanningIds.value.size) startPolling();
      else stopPolling();
    } catch (err) {
      if (isCurrent()) {
        error.value =
          errorDetail(err) || "Could not read the registered model folders.";
      } else {
        console.warn("A superseded read of the model folders failed", err);
      }
    } finally {
      if (isCurrent()) loading.value = false;
    }
  }

  /**
   * Say how the first scan of a folder added here went, once it has finished.
   *
   * Only those: every other scan is the server keeping up by itself, and a
   * notice each time would be noise. What it found reaches the shelf as
   * `models_arrived`. A folder that vanished from the list (forgotten
   * mid-scan) is dropped silently.
   */
  function announceFinishedScans() {
    if (!announcing.size) return;
    const notices = useNoticeStore();
    const byId = new Map(folders.value.map((folder) => [folder.id, folder]));
    for (const id of [...announcing]) {
      const folder = byId.get(id);
      if (folder && SCAN_IN_FLIGHT.has(folder.scan_status)) continue;
      announcing.delete(id);
      if (folder?.scan_status === "completed") {
        notices.push({
          level: "success",
          text: `Scanned ${basename(folder.path)}. ${countLabel(folder.file_count)} listed.`,
        });
      } else if (folder?.scan_status === "failed") {
        notices.push({
          level: "error",
          text: `Could not scan ${basename(folder.path)}${folder.scan_error ? `: ${folder.scan_error}` : "."}`,
        });
      }
    }
  }

  function startPolling() {
    if (pollHandle !== null) return;
    pollHandle = setInterval(
      () => refresh({ quiet: true, join: true }),
      POLL_MS,
    );
  }

  function stopPolling() {
    if (pollHandle === null) return;
    clearInterval(pollHandle);
    pollHandle = null;
  }

  /**
   * Register a folder. The server queues its first scan itself, so the row
   * comes back already `pending` rather than reading as an empty failure.
   *
   * A `source` or Downloads folder is never scanned (it is taken from, not
   * catalogued), so its notice promises no scan and none is waited for.
   *
   * @param {Object} options - forwarded to the API's `createModelFolder`.
   * @returns {Promise<boolean>} true when the folder was registered.
   */
  async function add(options) {
    const notices = useNoticeStore();
    const takenFrom =
      options?.kind === SOURCE_KIND || options?.kind === DOWNLOADS_KIND;
    try {
      const created = await createModelFolder(options);
      await refresh();
      const name = basename(created?.path || options.path);
      const text =
        options?.kind === DOWNLOADS_KIND
          ? `Watching ${name}. New models saved there move to your default folder.`
          : takenFrom
            ? `Added ${name}. Its training runs are ready to import.`
            : `Added ${name}. Looking for models in it now.`;
      notices.push({ level: "success", text });
      if (!takenFrom && created?.id != null) {
        announcing.add(created.id);
        announceFinishedScans();
      }
      return true;
    } catch (err) {
      notices.push({
        level: "error",
        text: errorDetail(err) || "Could not add that folder.",
      });
      return false;
    }
  }

  /**
   * Forget a folder, offering the way back.
   *
   * No confirmation prompt: nothing on disk is touched and the models keep the
   * names, triggers and attachments the owner gave them, so this is cheap to
   * reverse. It is only cheap because the undo exists, which is why the row's
   * fields are captured BEFORE the request rather than read back after it.
   *
   * @param {Object} folder - the row as listed.
   * @returns {Promise<boolean>} true when the folder was forgotten.
   */
  async function forget(folder) {
    const notices = useNoticeStore();
    const restore = {
      path: folder.path,
      kind: folder.kind,
      hostPath: folder.host_path || undefined,
      deleteAfterImport: folder.delete_after_import ?? undefined,
    };
    try {
      const body = await forgetModelFolder(folder.id);
      await refresh();
      const gone = Number(body?.tombstoned_files || 0);
      notices.push({
        level: "success",
        text: gone
          ? `Forgot ${basename(folder.path)}. ${countLabel(gone)} left the shelf; their names and trigger words are kept.`
          : `Forgot ${basename(folder.path)}. Nothing was listed from it.`,
        action: { label: "Add it back", handler: () => add(restore) },
      });
      useModelShelfStore().fetchRows();
      return true;
    } catch (err) {
      notices.push({
        level: "error",
        text: errorDetail(err) || "Could not forget that folder.",
      });
      return false;
    }
  }

  /**
   * Make a folder the default destination for a new model.
   *
   * Choosing the managed store clears the owner's mark rather than setting
   * one: with nothing marked the managed store is the default, so the two say
   * the same thing and only one of them is stored.
   *
   * @param {Object} folder - a row from `defaultCandidates`.
   * @returns {Promise<boolean>} true when the default changed.
   */
  async function setDefault(folder) {
    const notices = useNoticeStore();
    try {
      await setDefaultModelFolder(folder.id, true);
      await refresh({ quiet: true });
      notices.push({
        level: "success",
        text: `New models now go to ${basename(folder.path)}.`,
      });
      return true;
    } catch (err) {
      notices.push({
        level: "error",
        text: errorDetail(err) || "Could not change the default folder.",
      });
      return false;
    }
  }

  /**
   * Send the models PixlStash downloads to *folder*. Unlike the default the
   * mark is always stored, so a later change of default does not move it.
   *
   * @param {Object} folder - a row from `defaultCandidates`.
   * @returns {Promise<boolean>} true when it changed.
   */
  async function setDownloadTarget(folder) {
    const notices = useNoticeStore();
    try {
      await setDownloadModelFolder(folder.id, true);
      await refresh({ quiet: true });
      notices.push({
        level: "success",
        text: `Downloaded models now go to ${basename(folder.path)}.`,
      });
      return true;
    } catch (err) {
      notices.push({
        level: "error",
        text: errorDetail(err) || "Could not change the download folder.",
      });
      return false;
    }
  }

  /**
   * Make one directory inside a registered folder (#2206).
   *
   * Refusals throw, with the server's `detail` in the response, so the field
   * that asked can show them under itself and stay open. Only the parent's
   * listing is re-read here; the caller reveals the new row in the tree it
   * owns, since the tree's open state is its own.
   *
   * @param {number} folderId
   * @param {string} parentPath - POSIX, `""` for the folder's root.
   * @param {string} name - one segment.
   * @returns {Promise<{folder_id: number, path: string}>} the new directory.
   */
  async function createDir(folderId, parentPath, name) {
    const created = await createModelFolderDir(folderId, {
      path: parentPath,
      name,
    });
    await loadDirs(folderId, parentPath, { force: true });
    useNoticeStore().push({ level: "success", text: `Created ${name}.` });
    return created;
  }

  /**
   * Rename a registered folder on disk. Refusals throw, with the server's
   * `detail`, so the field that asked shows them and stays open. The list
   * (and with it the settings dialog), where ComfyUI loads from and the
   * shelf's copies are all re-read, since each names the folder's path.
   *
   * @param {number} folderId
   * @param {string} name - one segment.
   * @returns {Promise<Object>} the folder as now registered.
   */
  async function renameFolder(folderId, name) {
    const renamed = await renameModelFolder(folderId, name);
    await refresh({ quiet: true });
    void loadComfyuiReads();
    useModelShelfStore().fetchRows();
    useNoticeStore().push({ level: "success", text: `Renamed to ${name}.` });
    return renamed;
  }

  /**
   * Drop everything the previous credential could see.
   *
   * The paths are host facts about this machine and the registry is owner-only,
   * so none of it may survive into a share or read-only session that is never
   * allowed to ask for it. Any scan being waited on is abandoned with it: the
   * server carries on, but this session no longer has standing to poll.
   *
   * A read already out answers for the previous credential: it must not
   * publish, and a read asked for after this must not share it. What each
   * consumer shows is kept: consumers are still mounted, and say what they
   * show as it changes.
   */
  function resetForSession() {
    dropReads();
    dirsQueue.waiting.clear();
    dirsQueue = { waiting: new Set(), running: 0 };
    stopPolling();
    folders.value = [];
    devices.value = [];
    announcing.clear();
    dirListings.value = new Map();
    treeExpanded.value = new Set();
    treeClosedDrives.value = new Set();
    comfyuiReads.value = NO_COMFYUI_READS;
    loading.value = false;
    loaded.value = false;
    error.value = "";
  }

  const unsubscribeSessionReset = onSessionReset(resetForSession);
  onScopeDispose(() => {
    unsubscribeSessionReset();
    stopPolling();
  });

  return {
    folders,
    devices,
    deviceByFolderId,
    labelByFolderId,
    loading,
    loaded,
    error,
    scanningIds,
    managedFolder,
    sourceFolder,
    downloadsFolder,
    defaultFolder,
    downloadFolder,
    defaultCandidates,
    registeredPaths,
    dirListings,
    treeExpanded,
    treeClosedDrives,
    dirsKey,
    behindLink,
    comfyuiReads,
    loadComfyuiReads,
    loadDirs,
    reloadDirs,
    setActiveDirs,
    releaseActiveDirs,
    refresh,
    refreshDevices,
    add,
    forget,
    setDefault,
    setDownloadTarget,
    createDir,
    renameFolder,
    resetForSession,
  };
});

/** The last segment of a path, for a sentence that must not run to 60 chars. */
export function basename(path) {
  const trimmed = String(path || "").replace(/[\\/]+$/, "");
  const parts = trimmed.split(/[\\/]/);
  return parts[parts.length - 1] || trimmed || "the folder";
}

/**
 * folder id → the name a folder row goes by.
 *
 * A folder is named for its tool, else its last segment. Folders that would
 * share a name (every StabilityMatrix folder reads "StabilityMatrix") are told
 * apart by the shortest tail of their paths no other in the group shares, with
 * the tool after it: "Lora (StabilityMatrix)", "Models/Lora", "Other/Lora".
 *
 * @param {Array<Object>} folders - the registry rows.
 * @returns {Map<number, string>}
 */
export function folderLabels(folders) {
  const nameOf = (folder) => folder.tool || basename(folder.path);
  const groups = new Map();
  for (const folder of folders) {
    const name = nameOf(folder);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(folder);
  }
  const labels = new Map();
  for (const [name, group] of groups) {
    if (group.length === 1) {
      labels.set(Number(group[0].id), name);
      continue;
    }
    const segments = group.map((folder) =>
      String(folder.path || "")
        .split(/[\\/]/)
        .filter(Boolean),
    );
    group.forEach((folder, i) => {
      const own = segments[i];
      let tail = own.join("/") || name;
      for (let n = 1; n <= own.length; n += 1) {
        const candidate = own.slice(-n).join("/");
        const shared = segments.some(
          (other, j) => j !== i && other.slice(-n).join("/") === candidate,
        );
        if (!shared) {
          tail = candidate;
          break;
        }
      }
      labels.set(Number(folder.id), folder.tool ? `${tail} (${folder.tool})` : tail);
    });
  }
  return labels;
}

/** "1 file" / "12 files", so a sentence never reads "1 files". */
export function countLabel(n) {
  const count = Number(n) || 0;
  return `${count.toLocaleString()} ${count === 1 ? "file" : "files"}`;
}
