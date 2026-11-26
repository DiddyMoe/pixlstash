// The model-folder registry, and the scans the server runs against it.
//
// The assertions worth having are the ones that pin decisions which are easy to
// "simplify" back into bugs: nobody starts a scan, so "scanning" is whatever the
// server's `scan_status` says and the poll must stop once none is in flight;
// only the first scan of a folder added here is announced, never the server's
// own rescans; and forgetting a folder is only cheap to undo because the row's
// fields are captured BEFORE the request that destroys them.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { computed } from "vue";

const listModelFolders = vi.fn();
const createModelFolder = vi.fn();
const forgetModelFolder = vi.fn();
const setDefaultModelFolder = vi.fn();
const listModelFolderDirs = vi.fn();
const listModelFolderDevices = vi.fn();
const getComfyuiReads = vi.fn();
const renameModelFolder = vi.fn();

vi.mock("../api/modelFolders", () => ({
  MANAGED_KIND: "managed",
  SOURCE_KIND: "source",
  DOWNLOADS_KIND: "downloads",
  CREATABLE_KINDS: ["user", "source", "downloads"],
  listModelFolders: (...a) => listModelFolders(...a),
  createModelFolder: (...a) => createModelFolder(...a),
  forgetModelFolder: (...a) => forgetModelFolder(...a),
  setDefaultModelFolder: (...a) => setDefaultModelFolder(...a),
  listModelFolderDirs: (...a) => listModelFolderDirs(...a),
  listModelFolderDevices: (...a) => listModelFolderDevices(...a),
  getComfyuiReads: (...a) => getComfyuiReads(...a),
  renameModelFolder: (...a) => renameModelFolder(...a),
}));

const fetchRows = vi.fn();
vi.mock("./useModelShelfStore", () => ({
  useModelShelfStore: () => ({ fetchRows }),
}));

import {
  useModelFoldersStore,
  basename,
  countLabel,
  folderLabels,
} from "./useModelFoldersStore";
import { useNoticeStore } from "./useNoticeStore";

/** One row of the shape `GET /model-folders` really returns. */
function folder(overrides = {}) {
  return {
    id: 1,
    path: "/home/me/loras",
    kind: "user",
    owner: null,
    movable: "per_item",
    host_path: null,
    delete_after_import: false,
    last_checked: null,
    created_at: "2026-08-01T10:00:00",
    file_count: 91,
    ...overrides,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
  fetchRows.mockReset();
  listModelFolders.mockReset().mockResolvedValue([folder()]);
  createModelFolder.mockReset().mockResolvedValue(folder({ id: 2 }));
  forgetModelFolder
    .mockReset()
    .mockResolvedValue({ status: "success", id: 1, tombstoned_files: 91 });
});

afterEach(() => {
  vi.useRealTimers();
});

/** Let every settled read run its continuations (the timers are fake). */
async function drain() {
  for (let i = 0; i < 50; i += 1) await Promise.resolve();
}

/** Make `api` answer each call only when the test says so, in call order. */
function holdAnswers(api) {
  const answers = [];
  api
    .mockReset()
    .mockImplementation(() => new Promise((resolve) => answers.push(resolve)));
  return answers;
}

/** A listing that names the level it lists, so the tests can tell them apart. */
function answerEveryListing() {
  listModelFolderDirs
    .mockReset()
    .mockImplementation(async (id, path) => ({ folder_id: id, path, dirs: [] }));
}

describe("scanning", () => {
  it("is scanning while the server says so, and stops polling when it is not", async () => {
    listModelFolders.mockResolvedValue([folder({ scan_status: "running" })]);
    const store = useModelFoldersStore();
    await store.refresh();
    expect(store.scanningIds.has(1)).toBe(true);

    await vi.advanceTimersByTimeAsync(3000);
    expect(listModelFolders).toHaveBeenCalledTimes(2);

    listModelFolders.mockResolvedValue([folder({ scan_status: "completed" })]);
    await vi.advanceTimersByTimeAsync(3000);
    expect(store.scanningIds.size).toBe(0);
    const reads = listModelFolders.mock.calls.length;
    await vi.advanceTimersByTimeAsync(9000);
    expect(listModelFolders).toHaveBeenCalledTimes(reads);
  });

  it("announces the first scan of a folder it added, and not the server's own", async () => {
    // Folder 1 is being rescanned by the server; folder 2 was just added.
    createModelFolder.mockResolvedValue(
      folder({ id: 2, scan_status: "pending" }),
    );
    listModelFolders.mockResolvedValue([
      folder({ id: 1, path: "/home/me/other", scan_status: "running" }),
      folder({ id: 2, scan_status: "pending" }),
    ]);
    const store = useModelFoldersStore();
    await store.add({ path: "/home/me/loras", kind: "user" });
    expect(useNoticeStore().notices.at(-1).text).toContain(
      "Looking for models",
    );

    listModelFolders.mockResolvedValue([
      folder({ id: 1, path: "/home/me/other", scan_status: "completed" }),
      folder({ id: 2, scan_status: "completed", file_count: 91 }),
    ]);
    await vi.advanceTimersByTimeAsync(3000);
    const texts = useNoticeStore().notices.map((n) => n.text);
    expect(texts.at(-1)).toBe("Scanned loras. 91 files listed.");
    expect(texts.some((t) => t.includes("other"))).toBe(false);

    // Announced once: the next read does not say it again.
    await store.refresh();
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual(texts);
  });

  it("says so when the first scan of a folder it added failed", async () => {
    createModelFolder.mockResolvedValue(
      folder({ id: 2, scan_status: "pending" }),
    );
    listModelFolders.mockResolvedValue([
      folder({ id: 2, scan_status: "failed", scan_error: "disk on fire" }),
    ]);
    const store = useModelFoldersStore();
    await store.add({ path: "/home/me/loras", kind: "user" });
    const notice = useNoticeStore().notices.at(-1);
    expect(notice.level).toBe("error");
    expect(notice.text).toBe("Could not scan loras: disk on fire");
  });

  it("promises no scan for an output root, which is taken from", async () => {
    createModelFolder.mockResolvedValue(
      folder({ id: 2, kind: "source", path: "/runs" }),
    );
    const store = useModelFoldersStore();

    expect(await store.add({ path: "/runs", kind: "source" })).toBe(true);
    const notice = useNoticeStore().notices.at(-1);
    expect(notice.text).toContain("ready to import");
    expect(notice.text).not.toContain("Looking for models");
  });

  it("names the one output root, and nothing else", async () => {
    // What every "is it set yet" control in the UI reads.
    listModelFolders.mockResolvedValue([
      folder({ id: 1, kind: "user" }),
      folder({ id: 2, kind: "source", path: "/runs" }),
      folder({ id: 3, kind: "managed", path: "/store" }),
    ]);
    const store = useModelFoldersStore();
    await store.refresh();
    expect(store.sourceFolder?.path).toBe("/runs");
  });

  it("has no output root until one is registered", async () => {
    listModelFolders.mockResolvedValue([folder({ id: 1, kind: "user" })]);
    const store = useModelFoldersStore();
    await store.refresh();
    expect(store.sourceFolder).toBeUndefined();
  });
});

describe("forgetting", () => {
  it("offers the way back with the fields the row had before the delete", async () => {
    // The undo is the whole reason this needs no confirmation prompt. Reading
    // the path back after the DELETE would read nothing.
    const store = useModelFoldersStore();
    listModelFolders.mockResolvedValue([
      folder({ host_path: "/mnt/host/loras" }),
    ]);
    await store.refresh();
    const row = store.folders[0];

    listModelFolders.mockResolvedValue([]);
    await store.forget(row);

    const notice = useNoticeStore().notices.at(-1);
    expect(notice.text).toContain("91 files left the shelf");
    expect(notice.action.label).toBe("Add it back");

    await notice.action.handler();
    expect(createModelFolder).toHaveBeenCalledWith({
      path: "/home/me/loras",
      kind: "user",
      hostPath: "/mnt/host/loras",
      deleteAfterImport: false,
    });
  });

  it("says nothing was listed rather than 0 files", async () => {
    forgetModelFolder.mockResolvedValue({ tombstoned_files: 0 });
    const store = useModelFoldersStore();
    await store.refresh();
    await store.forget(store.folders[0]);
    expect(useNoticeStore().notices.at(-1).text).toContain(
      "Nothing was listed from it",
    );
  });
});

describe("subfolder listings", () => {
  it("reads once more after a read out when forced, and never shows the older answer", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    const first = store.loadDirs(1, "");
    // A scan landed while the first read was out: it must read again, but
    // not alongside it, or the older answer could land last.
    const forced = store.loadDirs(1, "", { force: true });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);

    answers[0]({ dirs: [{ name: "old" }] });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(2);
    const entry = store.dirListings.get(store.dirsKey(1, ""));
    expect(entry.listing).toBe(null);
    expect(entry.loading).toBe(true);

    answers[1]({ dirs: [{ name: "new" }] });
    expect(await forced).toEqual({ dirs: [{ name: "new" }] });
    expect(await first).toEqual({ dirs: [{ name: "new" }] });
    expect(store.dirListings.get(store.dirsKey(1, "")).listing.dirs).toEqual([
      { name: "new" },
    ]);
  });

  it("re-reads only the listings something shows, and marks the rest stale", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    for (let i = 0; i < 64; i += 1) await store.loadDirs(1, `d${i}`);
    store.setActiveDirs(Symbol("test-view"), [store.dirsKey(1, "d0")]);
    listModelFolderDirs.mockClear();

    store.reloadDirs();
    await drain();
    expect(listModelFolderDirs.mock.calls).toEqual([[1, "d0"]]);
    const stale = [...store.dirListings].filter(([, entry]) => entry.stale);
    expect(stale).toHaveLength(63);
    expect(store.dirListings.get("1:d0").stale).toBe(false);
  });

  it("re-reads a stale listing on its next use, still drawing the old one meanwhile", async () => {
    const store = useModelFoldersStore();
    listModelFolderDirs.mockReset().mockResolvedValue({ dirs: [{ name: "old" }] });
    await store.loadDirs(1, "");
    store.reloadDirs();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);

    const answers = holdAnswers(listModelFolderDirs);
    const next = store.loadDirs(1, "");
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);
    const entry = store.dirListings.get("1:");
    expect(entry.loading).toBe(true);
    expect(entry.listing.dirs).toEqual([{ name: "old" }]);

    answers[0]({ dirs: [{ name: "new" }] });
    expect(await next).toEqual({ dirs: [{ name: "new" }] });
    expect(store.dirListings.get("1:").stale).toBe(false);
  });

  it("re-reads a stale listing as soon as something shows it", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    await store.loadDirs(1, "");
    store.reloadDirs();
    listModelFolderDirs.mockClear();

    store.setActiveDirs(Symbol("test-picker"), ["1:"]);
    await drain();
    expect(listModelFolderDirs.mock.calls).toEqual([[1, ""]]);
    expect(store.dirListings.get("1:").stale).toBe(false);
  });

  it("reads a listing something starts showing after the cache dropped it", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    await store.loadDirs(1, "open");
    // Nothing shows it (the tree is unmounted), and 256 newer listings push
    // it out of the cache.
    for (let i = 0; i < 256; i += 1) await store.loadDirs(1, `d${i}`);
    expect(store.dirListings.has("1:open")).toBe(false);
    listModelFolderDirs.mockClear();

    // The tree comes back and says it shows its root and the row left open.
    store.setActiveDirs(Symbol("test-tree"), ["1:", "1:open"]);
    await drain();
    expect(listModelFolderDirs.mock.calls).toEqual([
      [1, ""],
      [1, "open"],
    ]);
    expect(store.dirListings.get("1:open").listing).toEqual({
      folder_id: 1,
      path: "open",
      dirs: [],
    });
  });

  it("stops re-reading a listing once the last thing showing it lets go", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    await store.loadDirs(1, "");
    const tree = Symbol("test-tree");
    const picker = Symbol("test-picker");
    store.setActiveDirs(tree, ["1:"]);
    store.setActiveDirs(picker, ["1:"]);

    store.releaseActiveDirs(picker);
    listModelFolderDirs.mockClear();
    store.reloadDirs();
    await drain();
    expect(listModelFolderDirs.mock.calls).toEqual([[1, ""]]);

    store.releaseActiveDirs(tree);
    listModelFolderDirs.mockClear();
    store.reloadDirs();
    await drain();
    expect(listModelFolderDirs).not.toHaveBeenCalled();
    expect(store.dirListings.get("1:").stale).toBe(true);
  });

  it("re-reads at most four listings at once", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    const shown = [];
    for (let i = 0; i < 10; i += 1) {
      await store.loadDirs(1, `d${i}`);
      shown.push(store.dirsKey(1, `d${i}`));
    }
    store.setActiveDirs(Symbol("test-tree"), shown);
    const answers = holdAnswers(listModelFolderDirs);

    store.reloadDirs();
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(4);

    answers[0]({ dirs: [] });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(5);

    for (let i = 1; i < 10; i += 1) {
      answers[i]({ dirs: [] });
      await drain();
    }
    expect(listModelFolderDirs).toHaveBeenCalledTimes(10);
    expect(listModelFolderDirs.mock.calls.map(([, path]) => path)).toEqual(
      shown.map((key) => key.slice(2)),
    );
  });

  it("keeps the reads a change adds behind listings being read to four at once", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    let answered = 0;
    const answer = (i) => {
      answers[i]({ dirs: [] });
      answered += 1;
    };
    const inFlight = () => listModelFolderDirs.mock.calls.length - answered;
    // Ten open rows being read (a tree mounting), then a change lands.
    const shown = [];
    for (let i = 0; i < 10; i += 1) {
      store.loadDirs(1, `d${i}`);
      shown.push(store.dirsKey(1, `d${i}`));
    }
    store.setActiveDirs(Symbol("test-tree"), shown);
    store.reloadDirs();
    await drain();
    expect(inFlight()).toBe(10);

    for (let i = 0; i < 10; i += 1) answer(i);
    await drain();
    expect(inFlight()).toBe(4);

    // Each still gets its one more read, four at a time.
    for (let i = 10; i < 20; i += 1) {
      answer(i);
      await drain();
      expect(inFlight()).toBeLessThanOrEqual(4);
    }
    expect(listModelFolderDirs.mock.calls.slice(10).map(([, path]) => path)).toEqual(
      shown.map((key) => key.slice(2)),
    );
    expect(inFlight()).toBe(0);
  });

  it("reads a listing once more when a change lands while it is being read", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    const first = store.loadDirs(1, "");
    // Two changes in a row still cost one more read, not two.
    store.reloadDirs();
    store.reloadDirs();
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);

    answers[0]({ dirs: [{ name: "before the change" }] });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(2);
    expect(store.dirListings.get("1:").listing).toBe(null);

    answers[1]({ dirs: [{ name: "after the change" }] });
    expect(await first).toEqual({ dirs: [{ name: "after the change" }] });
    expect(listModelFolderDirs).toHaveBeenCalledTimes(2);
  });

  it("leaves a listing being read to that read when a mount reloads", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    store.loadDirs(1, "");
    store.reloadDirs({ join: true });
    answers[0]({ dirs: [] });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);
  });

  it("writes a listing into the map it already has", async () => {
    listModelFolderDirs.mockReset().mockResolvedValue({ dirs: [{ name: "a" }] });
    const store = useModelFoldersStore();
    const listings = store.dirListings;
    const drawn = computed(
      () => store.dirListings.get("1:")?.listing?.dirs.length ?? -1,
    );
    expect(drawn.value).toBe(-1);
    await store.loadDirs(1, "");
    expect(store.dirListings).toBe(listings);
    expect(drawn.value).toBe(1);
  });

  it("keeps at most 256 listings nothing shows or reads, dropping the oldest first", async () => {
    answerEveryListing();
    const store = useModelFoldersStore();
    await store.loadDirs(1, "shown");
    store.setActiveDirs(Symbol("test-tree"), ["1:shown"]);
    let answerHeld;
    listModelFolderDirs.mockImplementationOnce(
      () => new Promise((resolve) => (answerHeld = resolve)),
    );
    const held = store.loadDirs(1, "held");
    for (let i = 0; i < 300; i += 1) await store.loadDirs(1, `d${i}`);

    const keys = [...store.dirListings.keys()];
    expect(keys).toHaveLength(258);
    expect(keys).toContain("1:shown");
    expect(keys).toContain("1:held");
    expect(keys).not.toContain("1:d43");
    expect(keys).toContain("1:d44");
    expect(keys).toContain("1:d299");
    answerHeld({ dirs: [] });
    await held;
  });

  it("neither shows nor shares a read from before a session reset", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    const before = store.loadDirs(1, "");
    store.resetForSession();
    const after = store.loadDirs(1, "");
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(2);

    answers[0]({ dirs: [{ name: "from-the-old-session" }] });
    expect(await before).toBe(null);
    expect(store.dirListings.get("1:").listing).toBe(null);
    expect(store.dirListings.get("1:").loading).toBe(true);

    // The read after the reset is still the one shared.
    const joined = store.loadDirs(1, "");
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(2);
    answers[1]({ dirs: [{ name: "fake-new" }] });
    expect(await after).toEqual({ dirs: [{ name: "fake-new" }] });
    expect(await joined).toEqual({ dirs: [{ name: "fake-new" }] });
    expect(store.dirListings.get("1:").listing.dirs).toEqual([
      { name: "fake-new" },
    ]);
  });

  it("never starts a read asked for behind one a session reset dropped", async () => {
    const store = useModelFoldersStore();
    const answers = holdAnswers(listModelFolderDirs);
    store.loadDirs(1, "");
    const forced = store.loadDirs(1, "", { force: true });
    store.resetForSession();
    answers[0]({ dirs: [{ name: "from-the-old-session" }] });
    await drain();
    expect(listModelFolderDirs).toHaveBeenCalledTimes(1);
    expect(await forced).toBe(null);
    expect(store.dirListings.size).toBe(0);
  });

  it("knows a place at or under a linked directory by the listings read", async () => {
    listModelFolderDirs.mockReset().mockImplementation(async (id, path) => ({
      folder_id: id,
      path,
      dirs:
        path === ""
          ? [
              { name: "Lora", path: "Lora" },
              { name: "Shared", path: "Shared", link: true },
            ]
          : [],
    }));
    const store = useModelFoldersStore();
    // Nothing read yet: nothing is known to be a link.
    expect(store.behindLink(1, "Shared")).toBe(false);
    await store.loadDirs(1, "");
    expect(store.behindLink(1, "Shared")).toBe(true);
    expect(store.behindLink(1, "Shared/loras/flux")).toBe(true);
    expect(store.behindLink(1, "Lora")).toBe(false);
    expect(store.behindLink(1, "")).toBe(false);
    // Another folder's listing says nothing about this one.
    expect(store.behindLink(2, "Shared")).toBe(false);
  });
});

describe("registry reads", () => {
  it("never brings back a folder forgotten while a read was out", async () => {
    const answers = holdAnswers(listModelFolders);
    const store = useModelFoldersStore();
    // A poll's read is out when the owner forgets the folder.
    const polled = store.refresh({ quiet: true, join: true });
    const forgot = store.forget(folder());
    await drain();
    expect(listModelFolders).toHaveBeenCalledTimes(1);

    // It left before the forget, so it still lists the folder, mid-scan.
    answers[0]([folder({ scan_status: "running" })]);
    await drain();
    expect(store.folders).toEqual([]);
    expect(store.scanningIds.size).toBe(0);
    // The forget's own read is out, so the list is still loading.
    expect(store.loading).toBe(true);
    expect(listModelFolders).toHaveBeenCalledTimes(2);

    answers[1]([]);
    expect(await forgot).toBe(true);
    await polled;
    expect(store.folders).toEqual([]);
    expect(store.loaded).toBe(true);
    expect(store.loading).toBe(false);
    listModelFolders.mockClear();
    await vi.advanceTimersByTimeAsync(9000);
    expect(listModelFolders).not.toHaveBeenCalled();
  });

  it("shares a read already out with a mount or a poll, and drives' reads always", async () => {
    const folderAnswers = holdAnswers(listModelFolders);
    const driveAnswers = holdAnswers(listModelFolderDevices);
    const store = useModelFoldersStore();
    store.refresh({ join: true });
    store.refresh({ quiet: true, join: true });
    store.refreshDevices();
    store.refreshDevices();

    folderAnswers[0]([folder()]);
    driveAnswers[0]([{ device_id: "fake-drive", folder_ids: [1] }]);
    await drain();
    expect(listModelFolders).toHaveBeenCalledTimes(1);
    expect(listModelFolderDevices).toHaveBeenCalledTimes(1);
    expect(store.folders).toEqual([folder()]);
    expect(store.devices).toEqual([{ device_id: "fake-drive", folder_ids: [1] }]);
  });

  it("lets a poll tick share a registry read still out rather than queue another", async () => {
    listModelFolders.mockResolvedValueOnce([folder({ scan_status: "running" })]);
    const store = useModelFoldersStore();
    await store.refresh();
    const answers = holdAnswers(listModelFolders);
    await vi.advanceTimersByTimeAsync(3000);
    await vi.advanceTimersByTimeAsync(3000);

    answers[0]([folder({ scan_status: "running" })]);
    await drain();
    expect(listModelFolders).toHaveBeenCalledTimes(1);
    expect(store.folders).toEqual([folder({ scan_status: "running" })]);
  });

  it("shows nothing read before a session reset, and shares none of it after", async () => {
    const folderAnswers = holdAnswers(listModelFolders);
    const driveAnswers = holdAnswers(listModelFolderDevices);
    const store = useModelFoldersStore();
    const before = store.refresh();
    const drives = store.refreshDevices();
    store.resetForSession();
    store.refresh({ join: true });
    store.refreshDevices();
    await drain();
    expect(listModelFolders).toHaveBeenCalledTimes(2);
    expect(listModelFolderDevices).toHaveBeenCalledTimes(2);

    folderAnswers[0]([folder({ scan_status: "running" })]);
    driveAnswers[0]([{ device_id: "fake-drive", folder_ids: [1] }]);
    await before;
    await drives;
    expect(store.folders).toEqual([]);
    expect(store.loaded).toBe(false);
    expect(store.devices).toEqual([]);
    expect(store.scanningIds.size).toBe(0);
    listModelFolders.mockClear();
    await vi.advanceTimersByTimeAsync(9000);
    expect(listModelFolders).not.toHaveBeenCalled();
  });

  it("reads where ComfyUI loads from again after a rename, past a read out", async () => {
    const answers = holdAnswers(getComfyuiReads);
    renameModelFolder.mockReset().mockResolvedValue(folder({ path: "/home/me/renamed" }));
    const store = useModelFoldersStore();
    const visit = store.loadComfyuiReads();
    await store.renameFolder(1, "renamed");
    await drain();
    expect(getComfyuiReads).toHaveBeenCalledTimes(1);

    answers[0]({ known: true, reads: [{ kind: "loras", path: "/home/me/loras" }] });
    await drain();
    expect(store.comfyuiReads.known).toBe(false);
    expect(getComfyuiReads).toHaveBeenCalledTimes(2);

    answers[1]({ known: true, reads: [{ kind: "loras", path: "/home/me/renamed" }] });
    await visit;
    expect(store.comfyuiReads.reads).toEqual([
      { kind: "loras", path: "/home/me/renamed" },
    ]);
  });
});

describe("session reset", () => {
  it("drops the host paths and abandons the poll", async () => {
    // Absolute paths on this machine are owner-only, so none of it may survive
    // into a share or read-only session that could never have asked for it.
    listModelFolders.mockResolvedValue([folder({ scan_status: "running" })]);
    const store = useModelFoldersStore();
    await store.refresh();
    expect(store.scanningIds.has(1)).toBe(true);
    store.resetForSession();

    expect(store.folders).toEqual([]);
    expect(store.loaded).toBe(false);
    expect(store.scanningIds.size).toBe(0);

    listModelFolders.mockClear();
    await vi.advanceTimersByTimeAsync(9000);
    expect(listModelFolders).not.toHaveBeenCalled();
  });
});

describe("sentence helpers", () => {
  it("never reads '1 files'", () => {
    expect(countLabel(1)).toBe("1 file");
    expect(countLabel(1204)).toBe("1,204 files");
  });

  it("names the folder by its last segment, trailing slash or not", () => {
    expect(basename("/home/me/loras/")).toBe("loras");
    expect(basename("C:\\models\\lora")).toBe("lora");
  });
});

describe("folderLabels", () => {
  const row = (id, path, tool = null) => ({ id, path, tool });

  it("keeps a lone tool's name and tells namesakes apart by path", () => {
    const labels = folderLabels([
      row(1, "/home/me/StabilityMatrix/Models/Lora", "StabilityMatrix"),
      row(2, "/home/me/StabilityMatrix/Models/VAE", "StabilityMatrix"),
      row(3, "/home/me/.cache/huggingface/hub", "HuggingFace"),
      row(4, "/home/me/loras"),
    ]);
    expect(labels.get(1)).toBe("Lora (StabilityMatrix)");
    expect(labels.get(2)).toBe("VAE (StabilityMatrix)");
    expect(labels.get(3)).toBe("HuggingFace");
    expect(labels.get(4)).toBe("loras");
  });

  it("reaches up a segment when the last ones match", () => {
    const labels = folderLabels([
      row(1, "/home/me/pixlstash/models", "PixlStash"),
      row(2, "/home/me/.insightface/models/", "PixlStash"),
      row(3, "D:\\SD\\Lora"),
      row(4, "/mnt/other/Lora"),
    ]);
    expect(labels.get(1)).toBe("pixlstash/models (PixlStash)");
    expect(labels.get(2)).toBe(".insightface/models (PixlStash)");
    expect(labels.get(3)).toBe("SD/Lora");
    expect(labels.get(4)).toBe("other/Lora");
  });

  it("never shows a rooted path, even when one path ends another", () => {
    const labels = folderLabels([
      row(1, "/Lora", "StabilityMatrix"),
      row(2, "/a/Lora", "StabilityMatrix"),
    ]);
    expect(labels.get(1)).toBe("Lora (StabilityMatrix)");
    expect(labels.get(2)).toBe("a/Lora (StabilityMatrix)");
  });
});
