import { afterEach, describe, it, expect, beforeEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";

// The store imports the singleton apiClient (+ isReadOnly); mock so no HTTP runs.
vi.mock("../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  apiClient: { get: vi.fn() },
  isReadOnly: { value: false },
  onSessionReset: () => () => {},
}));
vi.mock("../api/workers", () => ({
  getWorkerProgress: vi.fn(async () => ({ workers: {} })),
}));
vi.mock("../api/pictureImport", () => ({ fetchStagingStatus: vi.fn() }));

import { fetchStagingStatus } from "../api/pictureImport";
import { useImportArrivalsStore } from "./useImportArrivalsStore";
import { useTasksStore } from "./useTasksStore";

beforeEach(() => {
  setActivePinia(createPinia());
});

// These cover the import-abort subsystem that ImageImporter now wires live: a
// registered handler must fire on abortImportRun, and must NOT fire once the run
// is unregistered (the terminal / unmount cleanup path).
describe("useTasksStore import abort wiring", () => {
  it("invokes the registered handler when abortImportRun is called", () => {
    const store = useTasksStore();
    const handler = vi.fn();
    store.registerImportAbort("import-1", handler);

    store.abortImportRun("import-1");

    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("does not fire after unregisterImportAbort (cleanup path)", () => {
    const store = useTasksStore();
    const handler = vi.fn();
    store.registerImportAbort("import-2", handler);
    store.unregisterImportAbort("import-2");

    store.abortImportRun("import-2");

    expect(handler).not.toHaveBeenCalled();
  });

  it("abortImportRun is a no-op for an unknown run id", () => {
    const store = useTasksStore();
    expect(() => store.abortImportRun("does-not-exist")).not.toThrow();
  });

  it("ignores a non-function handler", () => {
    const store = useTasksStore();
    store.registerImportAbort("import-3", null);
    expect(() => store.abortImportRun("import-3")).not.toThrow();
  });

  it("carries the honest `abortable` flag through setImportRun", () => {
    const store = useTasksStore();
    // Committed server-side rows are not client-abortable → no cancel affordance.
    store.setImportRun("import-4", { current: 0, total: 10 });
    expect(store.importRuns["import-4"].abortable).toBe(false);
    // A run flagged abortable surfaces the cancel affordance in the Tasks tab.
    store.setImportRun("import-5", { current: 0, total: 10, abortable: true });
    expect(store.importRuns["import-5"].abortable).toBe(true);
  });
});

// What fetchStagingStatus resolves with, for one moment of a server import.
function serverStatus(overrides = {}) {
  return {
    stage: "importing",
    phase: "processing",
    staged: 4,
    total: 4,
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
function networkError() {
  return Object.assign(new Error("Network Error"), {
    isAxiosError: true,
    code: "ERR_NETWORK",
  });
}
function httpError(status) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    isAxiosError: true,
    response: { status, data: { detail: "Staging session not found" } },
  });
}

const RUNS_KEY = "pixlstash:importRuns";
const RUN = {
  runId: "import-a",
  stagingId: "staging-a",
  backendUrl: "/api/v1",
  total: 4,
};
const LOST_TRACK =
  "Lost track of this import (the server restarted or the session expired). Pictures already imported are kept.";

function persistedRuns() {
  return JSON.parse(window.sessionStorage.getItem(RUNS_KEY) || "[]");
}

// A committed import is the store's to follow, not the importer's: the store
// outlives the grid that uploaded it, and a reload picks it back up.
describe("following a committed import", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fetchStagingStatus.mockReset();
    window.sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("follows one session once, however many importers ask", async () => {
    fetchStagingStatus.mockResolvedValue(serverStatus({ processed: 1 }));
    const store = useTasksStore();

    const first = store.followImportRun(RUN);
    const second = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchStagingStatus).toHaveBeenCalledTimes(2);

    store.stopPolling();
    const [a, b] = await Promise.all([first, second]);
    expect(a).toEqual({ state: "aborted" });
    // One follower: both asks are answered by the same outcome.
    expect(b).toBe(a);
  });

  it("says what the server is doing, and reaches 100% only when it is done", async () => {
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ phase: "queued" }))
      .mockResolvedValueOnce(serverStatus({ processed: 2 }))
      // Every file checked, nothing inserted yet: not done.
      .mockResolvedValueOnce(serverStatus({ processed: 4 }))
      .mockResolvedValueOnce(
        serverStatus({ phase: "persisting", processed: 4 }),
      )
      .mockResolvedValueOnce(
        serverStatus({ phase: "finalizing", processed: 4 }),
      )
      .mockResolvedValue(
        serverStatus({
          stage: "completed",
          phase: null,
          processed: 4,
          importedCount: 3,
          duplicateCount: 1,
        }),
      );
    const store = useTasksStore();
    const rows = [];

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);
    rows.push({ ...store.importRuns["import-a"] });
    for (let i = 0; i < 4; i++) {
      await vi.advanceTimersByTimeAsync(1000);
      rows.push({ ...store.importRuns["import-a"] });
    }

    expect(rows.map(({ message, percent }) => [message, percent])).toEqual([
      ["Waiting for the importer…", 0],
      ["Checking pictures (2 of 4)…", 50],
      ["Checking pictures (4 of 4)…", 95],
      ["Adding pictures to the library…", 95],
      ["Finishing up…", 95],
    ]);
    expect(rows.every((row) => row.status === "running")).toBe(true);
    // A queued import has done nothing yet, so no "0 / 4" count.
    expect(rows[0].total).toBe(0);
    expect(rows[1]).toMatchObject({ current: 2, total: 4 });

    await vi.advanceTimersByTimeAsync(1000);
    await expect(done).resolves.toMatchObject({ state: "completed" });
    expect(store.importRuns["import-a"]).toMatchObject({
      status: "completed",
      percent: 100,
      current: 4,
      total: 4,
      message: "Imported 3 images",
    });
    await vi.advanceTimersByTimeAsync(2600);
    expect(store.importRuns["import-a"]).toBeUndefined();
  });

  it("reads as before against a server that names no phase", async () => {
    fetchStagingStatus.mockResolvedValue(
      serverStatus({ phase: null, processed: 1 }),
    );
    const store = useTasksStore();

    store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);

    expect(store.importRuns["import-a"]).toMatchObject({
      message: "Checking for duplicates…",
      percent: 25,
    });
    store.stopPolling();
  });

  it("reconnects through a lost connection instead of failing", async () => {
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ processed: 1 }))
      .mockRejectedValueOnce(networkError())
      .mockRejectedValueOnce(httpError(503))
      .mockResolvedValue(
        serverStatus({
          stage: "completed",
          phase: null,
          processed: 4,
          importedCount: 4,
        }),
      );
    // The middle of each jitter range: retries at +0.5 s, then +1 s.
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const store = useTasksStore();
    const rows = [];

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.importRuns["import-a"]).toMatchObject({
      status: "running",
      message: "Reconnecting to the server…",
      current: 1,
      total: 4,
    });
    for (let i = 0; i < 30; i++) {
      const row = store.importRuns["import-a"];
      if (row) rows.push({ status: row.status, message: row.message });
      await vi.advanceTimersByTimeAsync(500);
    }

    await expect(done).resolves.toMatchObject({ state: "completed" });
    expect(rows.map((row) => row.status)).not.toContain("failed");
    expect(rows.at(-1)).toEqual({
      status: "completed",
      message: "Imported 4 images",
    });
  });

  it("hands each picture's outcome to the arriving tiles once", async () => {
    fetchStagingStatus
      .mockResolvedValueOnce(
        serverStatus({
          processed: 1,
          outcomes: [{ key: "k1", outcome: "new", picture_id: 7 }],
        }),
      )
      .mockResolvedValueOnce(
        serverStatus({
          processed: 2,
          outcomes: [{ key: "k2", outcome: "duplicate", picture_id: 3 }],
        }),
      )
      .mockResolvedValue(
        serverStatus({ stage: "completed", phase: null, processed: 2 }),
      );
    const arrivals = useImportArrivalsStore();
    arrivals.begin("import-a", [
      new File(["x"], "a.png"),
      new File(["y"], "b.png"),
    ]);
    const store = useTasksStore();

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(2000);
    await done;

    expect(
      fetchStagingStatus.mock.calls.map(([args]) => args.outcomesFrom),
    ).toEqual([0, 1, 2]);
    expect(arrivals.runs[0].counts).toMatchObject({ added: 1, duplicate: 1 });
  });

  it("ends as unknown, never failed, when the server no longer has the session", async () => {
    fetchStagingStatus
      .mockResolvedValueOnce(serverStatus({ processed: 2 }))
      .mockRejectedValue(httpError(404));
    const arrivals = useImportArrivalsStore();
    arrivals.begin("import-a", [new File(["x"], "a.png")]);
    const store = useTasksStore();

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(1000);

    await expect(done).resolves.toMatchObject({ state: "unknown" });
    expect(store.importRuns["import-a"]).toMatchObject({
      status: "unknown",
      message: LOST_TRACK,
      current: 2,
      total: 4,
    });
    expect(arrivals.runs).toEqual([]);
    expect(persistedRuns()).toEqual([]);
    await vi.advanceTimersByTimeAsync(8000);
    expect(store.importRuns["import-a"]).toBeUndefined();
  });

  it("says it is still working after 10 minutes without progress", async () => {
    fetchStagingStatus.mockResolvedValue(serverStatus({ processed: 1 }));
    const store = useTasksStore();

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(9 * 60 * 1000);
    expect(store.importRuns["import-a"].message).toBe(
      "Checking pictures (1 of 4)…",
    );
    await vi.advanceTimersByTimeAsync(2 * 60 * 1000);

    expect(store.importRuns["import-a"]).toMatchObject({
      status: "running",
      message: "Still working on the server; no progress for 10 min",
      current: 1,
    });
    store.stopPolling();
    await expect(done).resolves.toEqual({ state: "aborted" });
  });

  it("keeps the server's reason when the import fails", async () => {
    fetchStagingStatus.mockResolvedValue(
      serverStatus({ stage: "failed", phase: null, error: "Disk full" }),
    );
    const arrivals = useImportArrivalsStore();
    arrivals.begin("import-a", [new File(["x"], "a.png")]);
    const store = useTasksStore();

    await expect(store.followImportRun(RUN)).resolves.toMatchObject({
      state: "failed",
    });
    expect(store.importRuns["import-a"]).toMatchObject({
      status: "failed",
      label: "Import failed",
      message: "Import failed: Disk full",
    });
    expect(arrivals.runs[0]).toMatchObject({
      stage: "error",
      error: "Disk full",
    });
    await vi.advanceTimersByTimeAsync(8000);
    expect(store.importRuns["import-a"]).toBeUndefined();
  });

  it("leaves no row behind an import cancelled on the server", async () => {
    fetchStagingStatus.mockResolvedValue(
      serverStatus({ stage: "cancelled", phase: null }),
    );
    const store = useTasksStore();

    await expect(store.followImportRun(RUN)).resolves.toMatchObject({
      state: "cancelled",
    });
    expect(store.importRuns["import-a"]).toBeUndefined();
  });

  it("does not reopen an import that has ended", async () => {
    fetchStagingStatus.mockResolvedValueOnce(
      serverStatus({
        stage: "completed",
        phase: null,
        processed: 4,
        importedCount: 4,
      }),
    );
    const store = useTasksStore();
    const first = await store.followImportRun(RUN);
    // A late answer from a stale poll must not turn it back into running.
    fetchStagingStatus.mockResolvedValue(serverStatus({ processed: 1 }));

    const again = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);

    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);
    expect(store.importRuns["import-a"].status).toBe("completed");
    expect(await again).toBe(first);
  });

  it("remembers the run for a reload until it ends", async () => {
    fetchStagingStatus.mockResolvedValueOnce(serverStatus({ processed: 1 }));
    const store = useTasksStore();

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);
    expect(persistedRuns()).toEqual([RUN]);

    fetchStagingStatus.mockResolvedValue(
      serverStatus({
        stage: "completed",
        phase: null,
        processed: 4,
        importedCount: 4,
      }),
    );
    await vi.advanceTimersByTimeAsync(1000);
    await done;
    expect(persistedRuns()).toEqual([]);
  });

  it("picks the import back up after a reload", async () => {
    window.sessionStorage.setItem(RUNS_KEY, JSON.stringify([RUN]));
    fetchStagingStatus.mockResolvedValue(serverStatus({ processed: 3 }));
    const store = useTasksStore();

    store.startPolling();
    await vi.advanceTimersByTimeAsync(0);

    expect(fetchStagingStatus).toHaveBeenCalledWith(
      expect.objectContaining({
        backendUrl: "/api/v1",
        stagingId: "staging-a",
      }),
    );
    expect(store.importRuns["import-a"]).toMatchObject({
      status: "running",
      current: 3,
      total: 4,
    });
    store.stopPolling();
  });

  it("stops following when polling stops, and keeps the run for next time", async () => {
    fetchStagingStatus.mockResolvedValue(serverStatus({ processed: 1 }));
    const store = useTasksStore();

    const done = store.followImportRun(RUN);
    await vi.advanceTimersByTimeAsync(0);
    store.stopPolling();
    await vi.advanceTimersByTimeAsync(60000);

    expect(fetchStagingStatus).toHaveBeenCalledTimes(1);
    expect(persistedRuns()).toEqual([RUN]);
    await expect(done).resolves.toEqual({ state: "aborted" });
  });

  it("still follows when the browser refuses storage", async () => {
    const refuse = () => {
      throw new DOMException("The operation is insecure.", "SecurityError");
    };
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(refuse);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(refuse);
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(refuse);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    fetchStagingStatus.mockResolvedValue(
      serverStatus({
        stage: "completed",
        phase: null,
        processed: 4,
        importedCount: 4,
      }),
    );
    const store = useTasksStore();

    store.startPolling();
    await expect(store.followImportRun(RUN)).resolves.toMatchObject({
      state: "completed",
    });
    expect(warn).toHaveBeenCalled();
    store.stopPolling();
  });
});
