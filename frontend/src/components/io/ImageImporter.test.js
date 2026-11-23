import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  apiClient: { get: vi.fn() },
  isReadOnly: { value: false },
  onSessionReset: () => () => {},
}));
vi.mock("../../api/workers", () => ({
  getWorkerProgress: vi.fn(async () => ({ workers: {} })),
}));
vi.mock("../../api/pictureImport", () => ({
  openStagingSession: vi.fn(),
  stageFiles: vi.fn(),
  commitStaging: vi.fn(),
  cancelStaging: vi.fn(),
  fetchStagingStatus: vi.fn(),
}));

import {
  cancelStaging,
  commitStaging,
  fetchStagingStatus,
  openStagingSession,
  stageFiles,
} from "../../api/pictureImport";
import { useTasksStore } from "../../stores/useTasksStore";
import { extractSupportedImportFilesFromDataTransfer } from "../../utils/media";
import ImageImporter from "./ImageImporter.vue";

// What the import API functions reject with: axios errors, with a `response`
// only when the server answered.
function abortError() {
  return Object.assign(new Error("canceled"), {
    isAxiosError: true,
    name: "CanceledError",
    code: "ERR_CANCELED",
  });
}
function networkError() {
  return Object.assign(new Error("Network Error"), {
    isAxiosError: true,
    code: "ERR_NETWORK",
  });
}
// The server's structured refusal: `detail` is `{code, message, ...}`.
function refusal(status, code, message, extra = {}) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    isAxiosError: true,
    response: { status, data: { detail: { code, message, ...extra } } },
  });
}
function httpError(status) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    isAxiosError: true,
    response: { status, data: { detail: "Staging session not found" } },
  });
}

/** Sends every byte of the batch at once, then answers after `ms`. */
function uploadAnsweringAfter(ms) {
  return ({ files, signal, onUploadProgress }) =>
    new Promise((resolve, reject) => {
      onUploadProgress?.({ loaded: files.length, total: files.length });
      const timer = setTimeout(
        () =>
          resolve({
            staged: files.length,
            received: [],
            skipped: [],
            replayed: false,
          }),
        ms,
      );
      signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(abortError());
        },
        { once: true },
      );
    });
}

/** A dead connection: no byte goes out and no answer comes back. */
function stalledUpload({ signal }) {
  return new Promise((_resolve, reject) => {
    signal?.addEventListener("abort", () => reject(abortError()), {
      once: true,
    });
  });
}

function serverStatus(overrides = {}) {
  return {
    stage: "importing",
    phase: "processing",
    staged: 2,
    total: 2,
    processed: 0,
    taskId: "task-1",
    importedCount: null,
    duplicateCount: null,
    scrapheapedCount: null,
    scrapheapedPictureIds: [],
    failedCount: null,
    cancelledCount: null,
    error: null,
    outcomes: [],
    ...overrides,
  };
}
const COMPLETED = serverStatus({
  stage: "completed",
  phase: null,
  processed: 2,
  importedCount: 2,
});

function pictures(count) {
  return Array.from(
    { length: count },
    (_, i) => new File(["x"], `fake-${i}.png`, { type: "image/png" }),
  );
}

/**
 * A drop of folders as the browser's entries API hands it over, for example
 * `{ "/a": ["x.png"] }`: each folder lists its files once, then nothing more.
 */
function folderDrop(tree) {
  const folders = Object.entries(tree).map(([folder, names]) => {
    const children = names.map((name) => {
      const file = new File(["x"], name, { lastModified: 1700000000000 });
      return {
        isFile: true,
        isDirectory: false,
        name,
        fullPath: `${folder}/${name}`,
        file: (ok) => ok(file),
      };
    });
    return {
      isFile: false,
      isDirectory: true,
      name: folder.slice(1),
      fullPath: folder,
      createReader: () => {
        let listed = false;
        return {
          readEntries: (ok) => {
            const batch = listed ? [] : children;
            listed = true;
            ok(batch);
          },
        };
      },
    };
  });
  return {
    types: ["Files"],
    files: [],
    items: folders.map((entry) => ({
      kind: "file",
      webkitGetAsEntry: () => entry,
    })),
  };
}

function onlyRun(store) {
  const rows = Object.values(store.importRuns);
  expect(rows).toHaveLength(1);
  return rows[0];
}

let wrapper = null;

function mountImporter() {
  wrapper = mount(ImageImporter);
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
  vi.resetAllMocks();
  window.sessionStorage.clear();
  openStagingSession.mockResolvedValue({
    stagingId: "staging-1",
    safeThreshold: 0,
    raw: {},
  });
  stageFiles.mockImplementation(uploadAnsweringAfter(10));
  commitStaging.mockResolvedValue({
    taskId: "task-1",
    stagedCount: 2,
    replayed: false,
  });
  cancelStaging.mockResolvedValue({ stage: "cancelled" });
  fetchStagingStatus.mockResolvedValue(COMPLETED);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  useTasksStore().stopPolling();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("ImageImporter upload", () => {
  it("commits a batch whose reply is slow after every byte is sent", async () => {
    stageFiles.mockImplementation(uploadAnsweringAfter(30000));
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(31000);

    // Every byte was out: the wait is the server's, not a stall.
    expect(stageFiles).toHaveBeenCalledTimes(1);
    expect(commitStaging).toHaveBeenCalledTimes(1);
    await run;
    expect(cancelStaging).not.toHaveBeenCalled();
    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")[0][0]).toMatchObject({
      importedCount: 2,
      phase: "done",
    });
  });

  it("sends a stalled batch again under the same key, then discards the staging", async () => {
    stageFiles.mockImplementation(stalledUpload);
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();
    const store = useTasksStore();

    const run = importer.vm.startImport(pictures(2));
    // Three 15 s stalls and two waits (1 s, 2 s) end at 48 s; the failed row
    // stays 8 s after that.
    await vi.advanceTimersByTimeAsync(50000);

    expect(stageFiles).toHaveBeenCalledTimes(3);
    const keys = stageFiles.mock.calls.map(([args]) => args.idempotencyKey);
    expect(keys[0]).toEqual(expect.any(String));
    expect(keys[0]).not.toBe("");
    expect(new Set(keys).size).toBe(1);
    // Its staged files are discarded, not left for the reaper.
    expect(cancelStaging).toHaveBeenCalledTimes(1);
    expect(cancelStaging).toHaveBeenCalledWith(
      expect.objectContaining({ stagingId: "staging-1" }),
    );
    expect(commitStaging).not.toHaveBeenCalled();
    await run;
    expect(importer.emitted("import-error")).toEqual([
      [{ message: "Upload stalled (no progress)" }],
    ]);
    expect(onlyRun(store)).toMatchObject({
      status: "failed",
      message: "Import failed: Upload stalled (no progress)",
    });
  });

  it("sends a batch again under its own key, and the next batch under a new one", async () => {
    const answer = uploadAnsweringAfter(10);
    stageFiles.mockRejectedValueOnce(networkError()).mockImplementation(answer);
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(150));
    await vi.advanceTimersByTimeAsync(20000);
    await run;

    const keys = stageFiles.mock.calls.map(([args]) => args.idempotencyKey);
    expect(keys).toHaveLength(3);
    expect(keys[0]).toEqual(expect.any(String));
    expect(keys[1]).toBe(keys[0]);
    expect(keys[2]).toEqual(expect.any(String));
    expect(keys[2]).not.toBe(keys[0]);
    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(importer.emitted("import-error")).toBeUndefined();
  });

  it("sends where each file sat in the drop, the same again on a resend", async () => {
    stageFiles
      .mockRejectedValueOnce(networkError())
      .mockImplementation(uploadAnsweringAfter(10));
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const onUnreadable = vi.fn();
    const dropped = await extractSupportedImportFilesFromDataTransfer(
      folderDrop({ "/a": ["x.png", "x.txt"], "/b": ["x.png"] }),
      { onUnreadable },
    );
    // Every folder listed cleanly: the paths below come from a whole walk.
    expect(onUnreadable).not.toHaveBeenCalled();
    const importer = mountImporter();

    const run = importer.vm.startImport([...dropped, ...pictures(1)]);
    await vi.advanceTimersByTimeAsync(20000);
    await run;

    const sent = stageFiles.mock.calls.map(([args]) => args.relpaths);
    // A picked file has no place in a drop, so it goes as a loose file.
    const paths = ["/a/x.png", "/a/x.txt", "/b/x.png", ""];
    expect(sent).toEqual([paths, paths]);
    const keys = stageFiles.mock.calls.map(([args]) => args.idempotencyKey);
    expect(keys[1]).toBe(keys[0]);
    expect(commitStaging).toHaveBeenCalledTimes(1);
  });

  it("asks again while the server is still taking the same batch", async () => {
    stageFiles
      .mockRejectedValueOnce(
        refusal(409, "batch_in_progress", "This batch is still arriving."),
      )
      .mockImplementation(uploadAnsweringAfter(10));
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(20000);
    await run;

    const keys = stageFiles.mock.calls.map(([args]) => args.idempotencyKey);
    expect(keys).toHaveLength(2);
    expect(keys[0]).toEqual(expect.any(String));
    expect(keys[1]).toBe(keys[0]);
    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(importer.emitted("import-error")).toBeUndefined();
  });

  it.each([
    [413, "staging_file_limit", "Too many files for one import."],
    [507, "insufficient_storage", "Not enough free disk space to stage these."],
    [422, "idempotency_key_mismatch", "That batch was sent with other files."],
  ])(
    "does not send a batch the server refused (%i) again",
    async (status, code, message) => {
      stageFiles.mockRejectedValue(refusal(status, code, message));
      vi.spyOn(Math, "random").mockReturnValue(0.5);
      const importer = mountImporter();
      const store = useTasksStore();

      const run = importer.vm.startImport(pictures(2));
      // Time for two resends (1 s and 2 s waits), within the failed row's 8 s.
      await vi.advanceTimersByTimeAsync(5000);

      expect(stageFiles).toHaveBeenCalledTimes(1);
      expect(cancelStaging).toHaveBeenCalledTimes(1);
      expect(commitStaging).not.toHaveBeenCalled();
      await run;
      expect(importer.emitted("import-error")).toEqual([[{ message }]]);
      expect(onlyRun(store)).toMatchObject({
        status: "failed",
        message: `Import failed: ${message}`,
      });
    },
  );

  it("follows the import when the session turns out to be committed already", async () => {
    stageFiles.mockRejectedValue(
      refusal(409, "staging_closed", "This import is already running.", {
        stage: "importing",
        task_id: "task-1",
      }),
    );
    cancelStaging.mockRejectedValue(
      refusal(409, "import_committed", "This import is already running.", {
        stage: "importing",
        task_id: "task-1",
      }),
    );
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(30000);
    await run;

    expect(stageFiles).toHaveBeenCalledTimes(1);
    expect(fetchStagingStatus).toHaveBeenCalledWith(
      expect.objectContaining({ stagingId: "staging-1" }),
    );
    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")[0][0]).toMatchObject({
      importedCount: 2,
    });
  });

  it("stops a cancelled upload at once, even between attempts", async () => {
    stageFiles.mockRejectedValueOnce(networkError());
    vi.spyOn(Math, "random").mockReturnValue(0.999);
    const importer = mountImporter();
    const store = useTasksStore();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(0);
    expect(stageFiles).toHaveBeenCalledTimes(1);
    store.abortImportRun(Object.keys(store.importRuns)[0]);
    await vi.advanceTimersByTimeAsync(0);

    expect(importer.emitted("import-cancelled") ?? []).toHaveLength(1);
    expect(stageFiles).toHaveBeenCalledTimes(1);
    expect(cancelStaging).toHaveBeenCalledTimes(1);
    await run;
    expect(store.importRuns).toEqual({});
  });
});

describe("ImageImporter commit", () => {
  it("checks a commit whose reply was lost instead of cancelling it", async () => {
    commitStaging.mockRejectedValueOnce(networkError());
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ processed: 1 }))
      .mockResolvedValue(COMPLETED);
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(30000);
    await run;

    expect(cancelStaging).not.toHaveBeenCalled();
    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")[0][0]).toMatchObject({
      importedCount: 2,
    });
  });

  it("commits again when the server says the commit never started", async () => {
    commitStaging
      .mockRejectedValueOnce(
        refusal(503, "library_switching", "The library is switching."),
      )
      .mockResolvedValue({ taskId: "task-1", stagedCount: 2, replayed: false });
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ stage: "staging", phase: null }))
      .mockResolvedValue(COMPLETED);
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(30000);
    await run;

    expect(commitStaging).toHaveBeenCalledTimes(2);
    expect(cancelStaging).not.toHaveBeenCalled();
    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")).toHaveLength(1);
  });

  it("discards the staging when the server refuses the commit", async () => {
    const message = "Not enough free disk space to import these.";
    commitStaging.mockRejectedValue(
      refusal(507, "insufficient_storage", message),
    );
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(30000);
    await run;

    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(cancelStaging).toHaveBeenCalledTimes(1);
    expect(fetchStagingStatus).not.toHaveBeenCalled();
    expect(importer.emitted("import-error")).toEqual([[{ message }]]);
  });

  it("stops asking about an unanswered commit once the tasks store stops polling", async () => {
    commitStaging.mockRejectedValue(networkError());
    fetchStagingStatus.mockRejectedValue(networkError());
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();
    const store = useTasksStore();
    const follow = vi.spyOn(store, "followImportRun");

    importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(5000);
    // The server is unreachable: the importer is still asking.
    expect(fetchStagingStatus).toHaveBeenCalled();

    // Signing out unmounts the app, and with it the importer, and stops the
    // store's polling.
    importer.unmount();
    wrapper = null;
    store.stopPolling();
    const asked = fetchStagingStatus.mock.calls.length;
    await vi.advanceTimersByTimeAsync(65000);

    expect(fetchStagingStatus).toHaveBeenCalledTimes(asked);
    expect(follow).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem("pixlstash:importRuns")).toBeNull();
    expect(store.importRuns).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not follow a commit answered after the tasks store stopped polling", async () => {
    commitStaging.mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(
            () =>
              resolve({ taskId: "task-1", stagedCount: 2, replayed: false }),
            5000,
          ),
        ),
    );
    const importer = mountImporter();
    const store = useTasksStore();
    const follow = vi.spyOn(store, "followImportRun");

    importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(1000);
    expect(commitStaging).toHaveBeenCalledTimes(1);
    importer.unmount();
    wrapper = null;
    store.stopPolling();
    await vi.advanceTimersByTimeAsync(65000);

    expect(follow).not.toHaveBeenCalled();
    expect(fetchStagingStatus).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem("pixlstash:importRuns")).toBeNull();
    expect(store.importRuns).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not commit again once the tasks store stops polling during the wait between commits", async () => {
    commitStaging.mockRejectedValue(networkError());
    fetchStagingStatus.mockResolvedValue(
      serverStatus({ stage: "staging", phase: null }),
    );
    // The wait before the second commit is then 1 s.
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();
    const store = useTasksStore();
    const follow = vi.spyOn(store, "followImportRun");

    importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(100);
    // The commit's reply was lost and the server says it never started: the
    // importer is waiting to commit again.
    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);

    importer.unmount();
    wrapper = null;
    store.stopPolling();
    await vi.advanceTimersByTimeAsync(65000);

    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);
    expect(follow).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem("pixlstash:importRuns")).toBeNull();
    expect(store.importRuns).toEqual({});
    expect(vi.getTimerCount()).toBe(0);
  });

  it("hands an unanswered commit to the store after the importer unmounts", async () => {
    commitStaging.mockRejectedValue(networkError());
    fetchStagingStatus
      .mockRejectedValueOnce(networkError())
      .mockResolvedValueOnce(serverStatus())
      .mockResolvedValue(COMPLETED);
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();
    const store = useTasksStore();

    importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(100);
    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);

    // A view switch: the importer goes, the store keeps polling.
    importer.unmount();
    wrapper = null;
    await vi.advanceTimersByTimeAsync(2000);

    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(cancelStaging).not.toHaveBeenCalled();
    expect(onlyRun(store)).toMatchObject({
      status: "completed",
      message: "Imported 2 images",
    });
  });
});

describe("ImageImporter after the commit", () => {
  it("keeps following an import that takes longer than ten minutes", async () => {
    let polls = 0;
    fetchStagingStatus.mockImplementation(async () => {
      polls += 1;
      return polls <= 700
        ? serverStatus({ total: 1000, staged: 1000, processed: polls })
        : serverStatus({
            stage: "completed",
            phase: null,
            total: 1000,
            processed: 1000,
            importedCount: 1000,
          });
    });
    const importer = mountImporter();

    const run = importer.vm.startImport(pictures(2));
    await vi.runAllTimersAsync();
    await run;

    expect(polls).toBe(701);
    expect(commitStaging).toHaveBeenCalledTimes(1);
    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")[0][0]).toMatchObject({
      importedCount: 1000,
    });
  });

  it("reports an import the server lost track of as unknown, not failed", async () => {
    fetchStagingStatus
      .mockRejectedValueOnce(httpError(503))
      .mockRejectedValue(httpError(404));
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const importer = mountImporter();
    const store = useTasksStore();

    const run = importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(5000);
    await run;

    expect(importer.emitted("import-error")).toBeUndefined();
    expect(importer.emitted("import-finished")[0][0]).toMatchObject({
      phase: "unknown",
      importedCount: null,
    });
    expect(onlyRun(store).status).toBe("unknown");
  });

  it("leaves the import to the store: kept for a reload, finished after the importer is gone", async () => {
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ processed: 1 }))
      .mockResolvedValueOnce(serverStatus({ processed: 1 }))
      .mockResolvedValue(COMPLETED);
    const importer = mountImporter();
    const store = useTasksStore();

    importer.vm.startImport(pictures(2));
    await vi.advanceTimersByTimeAsync(100);
    const saved = JSON.parse(
      window.sessionStorage.getItem("pixlstash:importRuns") || "[]",
    );
    expect(saved).toEqual([
      expect.objectContaining({
        stagingId: "staging-1",
        backendUrl: "/api/v1",
      }),
    ]);

    importer.unmount();
    wrapper = null;
    await vi.advanceTimersByTimeAsync(3000);

    expect(onlyRun(store)).toMatchObject({
      status: "completed",
      message: "Imported 2 images",
    });
    expect(window.sessionStorage.getItem("pixlstash:importRuns")).toBeNull();
  });
});
