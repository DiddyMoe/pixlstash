// useTrainingRunsStore: what a session reset leaves behind for the next pick,
// and how many reads of the runs and the jobs the readers cost.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../api/modelImports", () => ({ listRuns: vi.fn(), trashRun: vi.fn() }));
vi.mock("../api/training", () => ({
  cancelTrainingJob: vi.fn(),
  createTrainingJob: vi.fn(),
  deleteTrainingJob: vi.fn(),
  getTrainingJob: vi.fn(),
  listTrainingJobs: vi.fn(),
}));

import { listRuns, trashRun } from "../api/modelImports";
import {
  cancelTrainingJob,
  createTrainingJob,
  deleteTrainingJob,
  getTrainingJob,
  listTrainingJobs,
} from "../api/training";
import { notifySessionReset } from "../utils/apiClient";
import { useModelFoldersStore } from "./useModelFoldersStore";
import { useTrainingRunsStore } from "./useTrainingRunsStore";

function walter() {
  return {
    name: "walter_v2",
    checkpoints: [
      { step: 500, filename: "walter_v2_000000500.safetensors" },
      { step: 1000, filename: "walter_v2.safetensors" },
    ],
  };
}

/** One finished run, read and picked, as the grid leaves it. */
async function pickWalter(store) {
  store.runs = [walter()];
  store.jobs = [];
  store.jobsLoaded = true;
  store.chosen = ["run:walter_v2"];
  await nextTick();
}

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("a session reset and the per-checkpoint picks", () => {
  // The single run's steps fill once per run name, so a reload does not re-tick
  // a checkpoint the reader excluded. A reset must not leave that name behind:
  // the next pick of the same run, in whatever tick it lands, starts full.
  it.each([
    ["after the reset is seen", true],
    ["in the same tick as the reset", false],
  ])("ticks every checkpoint of a run picked again %s", async (_, settle) => {
    const store = useTrainingRunsStore();
    await pickWalter(store);
    expect(store.chosenSteps).toEqual([500, 1000]);
    store.chosenSteps = [1000];

    notifySessionReset("test");
    if (settle) await nextTick();
    await pickWalter(store);

    expect(store.chosenSteps).toEqual([500, 1000]);
  });
});

describe("the cards of jobs with no folder", () => {
  const job = (uuid, status, run_name = null, started_at = null) => ({
    uuid,
    name: uuid,
    status,
    run_name,
    started_at,
    created_at: "2026-10-01T00:00:00Z",
  });
  const STARTED = "2026-10-01T00:01:00Z";

  // A run that failed or was cancelled before ai-toolkit made a folder has
  // nothing on disk to draw, so its job is its only card (#2293); one that
  // finished, or ended after it had a folder, is drawn from the folder or not
  // at all.
  it("keeps a card for a run that ended before it had a folder", () => {
    const store = useTrainingRunsStore();
    store.runs = [walter()];
    store.jobs = [
      job("queued", "queued"),
      job("dataset-failed", "error"),
      job("cancelled", "stopped"),
      // Finished: imported, so it has its finish time (#2291).
      { ...job("finished", "completed"), completed_at: STARTED },
      job("imported-away", "error", "gone_v1", STARTED),
      // A started run under an output root since moved names no run.
      job("root-moved", "stopped", null, STARTED),
      job("walter", "error", "walter_v2", STARTED),
    ];

    const keys = store.cards.map((card) => card.key);
    expect(keys).toEqual([
      "job:queued",
      "job:dataset-failed",
      "job:cancelled",
      "run:walter_v2",
    ]);
    const failed = store.cards.find((card) => card.key === "job:dataset-failed");
    expect(failed.live).toBe(false);
    expect(failed.run).toBe(null);
  });
});

describe("a card's name", () => {
  it("is its job's name, and the folder's for a run trained elsewhere", () => {
    const store = useTrainingRunsStore();
    store.runs = [walter(), { name: "elsewhere_v1", checkpoints: [] }];
    store.jobs = [
      {
        uuid: "abcd1234",
        name: "Walter Z-Image Turbo #2",
        status: "completed",
        run_name: "walter_v2",
        started_at: "2026-10-01T00:01:00Z",
        completed_at: "2026-10-01T01:00:00Z",
        created_at: "2026-10-01T00:00:00Z",
      },
    ];

    const names = Object.fromEntries(store.cards.map((c) => [c.key, c.name]));
    expect(names).toEqual({
      "run:walter_v2": "Walter Z-Image Turbo #2",
      "run:elsewhere_v1": "elsewhere_v1",
    });
  });
});

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Let every settled read write, and the watchers it woke run. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Register `id` as the ai-toolkit output root. */
async function withSource(id = 1) {
  useModelFoldersStore().folders = [{ id, path: "/home/me/runs", kind: "source" }];
  await nextTick();
}

const names = (store) => store.runs.map((r) => r.name);

/**
 * Note when `promise` settles, and what `look` saw at that moment.
 *
 * Pinia hands every caller of an action a promise of its own, so two readers
 * sharing one read is told by WHEN each settles, not by promise identity.
 */
function watchSettle(promise, look = () => undefined) {
  const seen = { settled: false, saw: undefined };
  promise.then(
    () => Object.assign(seen, { settled: true, saw: look() }),
    () => Object.assign(seen, { settled: true, saw: look() }),
  );
  return seen;
}

describe("reading the runs and the jobs", () => {
  // A tab switch fires both focus and visibilitychange, the sidebar and the
  // grid both read on mount, and the live poll reads every 5 s: readers that
  // only want "the list as it is now" share the read already in flight. A
  // reader that comes after a change (a stop, a retry, a delete, an import)
  // must get a read that STARTED after it, so it never joins.
  beforeEach(() => {
    vi.clearAllMocks();
    listRuns.mockReset();
    listTrainingJobs.mockReset();
  });

  it("asks once for the runs however many readers join", async () => {
    const store = useTrainingRunsStore();
    await withSource(1);
    const read = deferred();
    listRuns.mockReturnValue(read.promise);

    const readers = [1, 2, 3].map(() =>
      watchSettle(store.loadRuns({ join: true }), () => names(store)),
    );

    expect(listRuns).toHaveBeenCalledTimes(1);
    await flush();
    expect(readers.some((r) => r.settled)).toBe(false);
    read.resolve([walter()]);
    await flush();
    // Every reader is answered by that one read, with its answer in place.
    expect(readers.map((r) => r.saw)).toEqual([
      ["walter_v2"],
      ["walter_v2"],
      ["walter_v2"],
    ]);
    expect(store.loading).toBe(false);
  });

  it("starts a read of its own for a reader that does not join, and keeps its answer", async () => {
    const store = useTrainingRunsStore();
    await withSource(1);
    const older = deferred();
    const newer = deferred();
    listRuns
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise);

    store.loadRuns({ join: true });
    const fresh = store.loadRuns();
    expect(listRuns).toHaveBeenCalledTimes(2);
    // A reader arriving now joins the newer read, not the one it superseded:
    // it is answered while the older read is still out.
    const late = watchSettle(store.loadRuns({ join: true }), () => names(store));
    expect(listRuns).toHaveBeenCalledTimes(2);

    newer.resolve([walter()]);
    await fresh;
    await flush();
    expect(late).toEqual({ settled: true, saw: ["walter_v2"] });
    older.resolve([{ name: "stale_v1", checkpoints: [] }]);
    await flush();
    expect(names(store)).toEqual(["walter_v2"]);
  });

  it("does not join a read of a folder that is no longer the output root", async () => {
    const store = useTrainingRunsStore();
    await withSource(1);
    const old = deferred();
    const now = deferred();
    listRuns.mockReturnValueOnce(old.promise).mockReturnValue(now.promise);

    store.loadRuns({ join: true });
    useModelFoldersStore().folders = [
      { id: 4, path: "/home/me/other-runs", kind: "source" },
    ];
    // The same tick: the store's root watcher has not run yet.
    const next = store.loadRuns({ join: true });

    expect(listRuns).toHaveBeenCalledTimes(2);
    expect(listRuns.mock.calls[1]).toEqual([4]);
    now.resolve([walter()]);
    await next;
    old.resolve([{ name: "old_root_v1", checkpoints: [] }]);
    await flush();
    expect(names(store)).toEqual(["walter_v2"]);
  });

  it("gives a failed read to every reader that joined it, and asks again next time", async () => {
    const store = useTrainingRunsStore();
    await withSource(1);
    const failing = deferred();
    listRuns.mockReturnValueOnce(failing.promise);

    // A failed read is reported in `error`, not by rejecting: every reader
    // that joined it must settle with that failure already in place.
    const readers = [1, 2].map(() =>
      watchSettle(store.loadRuns({ join: true }), () => store.error),
    );
    expect(listRuns).toHaveBeenCalledTimes(1);
    await flush();
    expect(readers.some((r) => r.settled)).toBe(false);
    failing.reject({ response: { data: { detail: "fake-unreadable" } } });
    await flush();
    expect(readers.map((r) => r.saw)).toEqual(["fake-unreadable", "fake-unreadable"]);
    expect(store.loading).toBe(false);

    listRuns.mockResolvedValueOnce([walter()]);
    await store.loadRuns({ join: true });
    expect(listRuns).toHaveBeenCalledTimes(2);
    expect(store.error).toBe("");
    expect(names(store)).toEqual(["walter_v2"]);
  });

  it("asks once for the jobs however many readers join", async () => {
    const store = useTrainingRunsStore();
    const read = deferred();
    listTrainingJobs.mockReturnValue(read.promise);

    const uuids = () => store.jobs.map((j) => j.uuid);
    const readers = [1, 2, 3].map(() =>
      watchSettle(store.loadJobs({ join: true }), uuids),
    );

    expect(listTrainingJobs).toHaveBeenCalledTimes(1);
    await flush();
    expect(readers.some((r) => r.settled)).toBe(false);
    read.resolve([{ uuid: "fake-job", name: "walter", status: "queued" }]);
    await flush();
    expect(readers.map((r) => r.saw)).toEqual([
      ["fake-job"],
      ["fake-job"],
      ["fake-job"],
    ]);
    expect(store.jobsLoaded).toBe(true);
  });

  it("starts a jobs read of its own for a reader that does not join, and keeps its answer", async () => {
    const store = useTrainingRunsStore();
    const older = deferred();
    const newer = deferred();
    listTrainingJobs
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise);

    store.loadJobs({ join: true });
    const fresh = store.loadJobs();
    expect(listTrainingJobs).toHaveBeenCalledTimes(2);
    const late = watchSettle(store.loadJobs({ join: true }), () =>
      store.jobs.map((j) => j.uuid),
    );
    expect(listTrainingJobs).toHaveBeenCalledTimes(2);

    newer.resolve([{ uuid: "fake-new", name: "new", status: "queued" }]);
    await fresh;
    await flush();
    expect(late).toEqual({ settled: true, saw: ["fake-new"] });
    older.resolve([{ uuid: "fake-old", name: "old", status: "queued" }]);
    await flush();
    expect(store.jobs.map((j) => j.uuid)).toEqual(["fake-new"]);
  });

  it("answers each jobs reader with what its read found, superseded or not", async () => {
    // A store outside this one (naming a new run, the Tasks quit) reads the
    // jobs here and needs the list its own read found, or null if it failed.
    const store = useTrainingRunsStore();
    const older = deferred();
    const newer = deferred();
    listTrainingJobs
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
      .mockRejectedValueOnce(new Error("fake-down"));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const first = store.loadJobs();
    const second = store.loadJobs();
    const joined = store.loadJobs({ join: true });
    newer.resolve([{ uuid: "fake-new", name: "new", status: "queued" }]);
    older.resolve([{ uuid: "fake-old", name: "old", status: "queued" }]);
    expect((await first).map((j) => j.uuid)).toEqual(["fake-old"]);
    expect((await second).map((j) => j.uuid)).toEqual(["fake-new"]);
    expect(await joined).toBe(await second);
    expect(store.jobs.map((j) => j.uuid)).toEqual(["fake-new"]);

    expect(await store.loadJobs()).toBeNull();
    expect(store.jobsFailed).toBe(true);
    warn.mockRestore();
  });

  it("gives a failed jobs read to every reader that joined it, and asks again next time", async () => {
    const store = useTrainingRunsStore();
    const failing = deferred();
    listTrainingJobs.mockReturnValueOnce(failing.promise);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const readers = [1, 2].map(() =>
      watchSettle(store.loadJobs({ join: true }), () => store.jobsFailed),
    );
    expect(listTrainingJobs).toHaveBeenCalledTimes(1);
    await flush();
    expect(readers.some((r) => r.settled)).toBe(false);
    failing.reject(new Error("fake-down"));
    await flush();
    expect(readers.map((r) => r.saw)).toEqual([true, true]);
    warn.mockRestore();

    listTrainingJobs.mockResolvedValueOnce([]);
    await store.loadJobs({ join: true });
    expect(listTrainingJobs).toHaveBeenCalledTimes(2);
    expect(store.jobsFailed).toBe(false);
  });

  it("does not join a read from before a session reset", async () => {
    // A reset drops whatever the old session's reads answer, so a reader
    // joining one would wait for an answer that is never written.
    const store = useTrainingRunsStore();
    await withSource(1);
    const oldRuns = deferred();
    const oldJobs = deferred();
    listRuns.mockReturnValueOnce(oldRuns.promise).mockResolvedValueOnce([walter()]);
    listTrainingJobs.mockReturnValueOnce(oldJobs.promise).mockResolvedValueOnce([]);

    store.reload({ join: true });
    notifySessionReset("test");
    // The new session's registry, in the same tick, so the root watcher sees
    // no change and starts no read of its own.
    useModelFoldersStore().folders = [{ id: 1, path: "/home/me/runs", kind: "source" }];
    const reader = watchSettle(store.reload({ join: true }));
    await flush();

    expect(listRuns).toHaveBeenCalledTimes(2);
    expect(listTrainingJobs).toHaveBeenCalledTimes(2);
    expect(reader.settled).toBe(true);
    expect(names(store)).toEqual(["walter_v2"]);
    expect(store.jobsLoaded).toBe(true);
    oldRuns.resolve([{ name: "old_session_v1", checkpoints: [] }]);
    oldJobs.resolve([{ uuid: "fake-old", name: "old", status: "queued" }]);
    await flush();
    expect(names(store)).toEqual(["walter_v2"]);
    expect(store.jobs).toEqual([]);
  });

  // Each change below is made while a shared read is in flight, as the live
  // poll's is every 5 s. That read left before the change; the store must
  // read again after it, and the older answer landing last must not win.
  it("reads the jobs again after stopping one", async () => {
    const store = useTrainingRunsStore();
    const events = [];
    const before = deferred();
    const after = deferred();
    listTrainingJobs
      .mockImplementationOnce(() => (events.push("list"), before.promise))
      .mockImplementationOnce(() => (events.push("list"), after.promise));
    cancelTrainingJob.mockImplementation(async () => {
      events.push("cancel");
    });
    const job = { id: 7, uuid: "fake-job", name: "walter", status: "completed" };

    store.loadJobs({ join: true });
    const stopping = store.stopJob(job, "walter");
    await flush();
    after.resolve([{ ...job, status: "stopped" }]);
    await flush();
    before.resolve([job]);
    await flush();

    expect(await stopping).toBe(true);
    expect(events).toEqual(["list", "cancel", "list"]);
    expect(store.jobs.map((j) => j.status)).toEqual(["stopped"]);
  });

  it("reads the jobs again after queuing a failed run once more", async () => {
    const store = useTrainingRunsStore();
    const events = [];
    const before = deferred();
    const after = deferred();
    const failedJob = { id: 7, uuid: "fake-failed", name: "walter", status: "error" };
    const queued = { id: 8, uuid: "fake-retry", name: "walter", status: "queued" };
    listTrainingJobs
      .mockImplementationOnce(() => (events.push("list"), before.promise))
      .mockImplementationOnce(() => (events.push("list"), after.promise));
    getTrainingJob.mockResolvedValue({ name: "walter", resolved_profile: {} });
    createTrainingJob.mockImplementation(async () => {
      events.push("create");
      return queued;
    });
    deleteTrainingJob.mockImplementation(async () => {
      events.push("delete");
    });

    store.loadJobs({ join: true });
    const retrying = store.retryJob(failedJob);
    await flush();
    after.resolve([queued]);
    await flush();
    before.resolve([failedJob]);
    await flush();

    expect(await retrying).toBe(queued);
    expect(events).toEqual(["list", "create", "delete", "list"]);
    expect(store.jobs.map((j) => j.uuid)).toEqual(["fake-retry"]);
    // The new card is picked, which it can only be once the read has it.
    expect(store.chosen).toEqual(["job:fake-retry"]);
  });

  it("reads the runs again after clearing one", async () => {
    const store = useTrainingRunsStore();
    await withSource(1);
    const events = [];
    const before = deferred();
    const after = deferred();
    listRuns
      .mockImplementationOnce(() => (events.push("list"), before.promise))
      .mockImplementationOnce(() => (events.push("list"), after.promise));
    trashRun.mockImplementation(async () => {
      events.push("trash");
    });
    listTrainingJobs.mockResolvedValue([]);

    store.loadRuns({ join: true });
    const clearing = store.deleteRuns([
      { key: "run:walter_v2", name: "walter_v2", run: walter(), job: null },
    ]);
    await flush();
    after.resolve([]);
    await flush();
    before.resolve([walter()]);
    await flush();
    await clearing;

    expect(events).toEqual(["list", "trash", "list"]);
    expect(names(store)).toEqual([]);
  });
});

describe("a checkpoint picked while the runs are read", () => {
  // The re-read keeps a single run's ticked checkpoints by keying their fill
  // on the run's name; it must not put back what was ticked when it STARTED.
  it.each([
    ["unticked", [500, 1000], [1000]],
    ["ticked", [1000], [1000, 500]],
  ])("stays %s when the read lands", async (_, before, picked) => {
    listRuns.mockReset();
    const store = useTrainingRunsStore();
    await withSource(1);
    await pickWalter(store);
    store.chosenSteps = before;
    await nextTick();
    const read = deferred();
    listRuns.mockReturnValueOnce(read.promise);

    const reading = store.loadRuns();
    store.chosenSteps = picked;
    read.resolve([walter()]);
    await reading;
    await nextTick();

    expect(store.chosenSteps).toEqual(picked);
  });
});
