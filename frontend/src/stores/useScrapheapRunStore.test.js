import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { toRaw } from "vue";

vi.mock("../api/pictures", () => ({
  startScrapheapRun: vi.fn(),
  getScrapheapRun: vi.fn(),
  getLatestScrapheapRun: vi.fn(),
  cancelScrapheapRun: vi.fn(),
}));

vi.mock("../api/operations", () => ({
  getUndoState: vi.fn(),
}));

// The run store hands its receipt rule to the shared operation store, and
// raises the restore run's one receipt through it; the mock observes both
// without the real store's network reads.
vi.mock("./useOperationStore", () => {
  const setNarrationFilter = vi.fn();
  const buildReceipt = vi.fn((operation, mode) => ({
    key: 1,
    mode,
    operationId: operation.id,
    batchId: operation.batch_id,
    summary: "fake-slice-summary",
    targetCount: Number(operation.target_count) || 0,
    mergedCount: 3,
  }));
  const showReceipt = vi.fn();
  return {
    useOperationStore: () => ({ setNarrationFilter, buildReceipt, showReceipt }),
  };
});

import {
  cancelScrapheapRun,
  getLatestScrapheapRun,
  getScrapheapRun,
  startScrapheapRun,
} from "../api/pictures";
import { getUndoState } from "../api/operations";
import { notifySessionReset } from "../utils/apiClient";
import { useOperationStore } from "./useOperationStore";
import { useScrapheapRunStore } from "./useScrapheapRunStore";

const NO_ANSWER = { code: "ECONNABORTED", message: "timeout of 60000ms exceeded" };

function runFixture(over = {}) {
  return {
    id: "run-1",
    kind: "purge",
    status: "running",
    active: true,
    include_protected: false,
    total: 50000,
    processed: 0,
    cancel_requested: false,
    deleted_count: 0,
    skipped_count: 0,
    skipped_locked_count: 0,
    skipped_locked: [],
    skipped_restored: 0,
    ghosts_kept: 0,
    ghosts_cascaded: 0,
    restored_count: 0,
    snapshots_with_deleted: null,
    segment_batch_ids: [],
    error_code: null,
    error: null,
    created_at: "2026-11-24T10:00:00Z",
    started_at: "2026-11-24T10:00:00Z",
    updated_at: "2026-11-24T10:00:00Z",
    finished_at: null,
    ...over,
  };
}

function ended(over = {}) {
  return runFixture({
    status: "complete",
    active: false,
    finished_at: "2026-11-24T10:01:00Z",
    ...over,
  });
}

/** A rejection with the shape axios gives the run routes' refusals. */
function refusal(status, detail) {
  return Object.assign(new Error(`fake-${status}`), {
    response: { status, data: { detail } },
  });
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** The rule the store handed to the operation store. */
function installedFilter() {
  const calls = useOperationStore().setNarrationFilter.mock.calls;
  expect(calls.length).toBe(1);
  expect(calls[0][0]).toBe("scrapheap-run");
  return calls[0][1];
}

let warnSpy;

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  for (const fn of [
    startScrapheapRun,
    getScrapheapRun,
    getLatestScrapheapRun,
    cancelScrapheapRun,
    getUndoState,
  ]) {
    fn.mockReset();
  }
  const operations = useOperationStore();
  operations.setNarrationFilter.mockClear();
  operations.buildReceipt.mockClear();
  operations.showReceipt.mockClear();
});

afterEach(() => {
  // A run left active keeps a poll timer; the next test must not inherit it.
  useScrapheapRunStore().forget();
  vi.useRealTimers();
  warnSpy.mockRestore();
});

/** Start a run the server answers at once, and return the store. */
async function followRun(over = {}, retryAfterMs = 1000, context = null) {
  const store = useScrapheapRunStore();
  startScrapheapRun.mockResolvedValue({ run: runFixture(over), retryAfterMs });
  await store.start({ kind: over.kind ?? "purge", confirmToken: "example-token", context });
  return store;
}

describe("useScrapheapRunStore - start and follow", () => {
  it("starts a run and shows the start pending until the answer lands", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    startScrapheapRun.mockReturnValue(answer.promise);

    const pending = store.start({
      kind: "purge",
      includeProtected: true,
      confirmToken: "example-token",
    });
    expect(startScrapheapRun).toHaveBeenCalledWith({
      kind: "purge",
      pictureIds: undefined,
      includeProtected: true,
      confirmToken: "example-token",
    });
    expect(store.action).toBe("start");

    const run = runFixture();
    answer.resolve({ run, retryAfterMs: 1000 });
    expect(await pending).toEqual({ ok: true, run });
    expect(store.action).toBe(null);
    // The store holds the very run the server answered with.
    expect(toRaw(store.run)).toBe(run);
    expect(store.active).toBe(true);
  });

  it("refuses a second request while one is on the wire, without sending it", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockReturnValue(deferred().promise);
    store.start({ kind: "restore" });

    expect(await store.start({ kind: "restore" })).toEqual({
      ok: false,
      error: { kind: "busy", status: null, detail: "", code: null, run: null },
    });
    expect(startScrapheapRun).toHaveBeenCalledTimes(1);
  });

  it("polls at the server's interval, held to the bounds", async () => {
    const store = await followRun({}, 5000);
    getScrapheapRun.mockResolvedValue({
      run: runFixture({ processed: 500 }),
      retryAfterMs: 50,
    });

    await vi.advanceTimersByTimeAsync(4999);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledWith("run-1");
    expect(store.run.processed).toBe(500);

    // A 50 ms hint is a mistake, not a reason to spin: the floor is 500 ms.
    getScrapheapRun.mockClear();
    getScrapheapRun.mockResolvedValue({
      run: runFixture({ processed: 1000 }),
      retryAfterMs: 10 * 60 * 1000,
    });
    await vi.advanceTimersByTimeAsync(499);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);

    // Ten minutes is held to the 30 s ceiling.
    getScrapheapRun.mockClear();
    getScrapheapRun.mockResolvedValue({ run: runFixture(), retryAfterMs: null });
    await vi.advanceTimersByTimeAsync(29999);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);

    // No hint at all is the 2 s default.
    getScrapheapRun.mockClear();
    await vi.advanceTimersByTimeAsync(1999);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);
  });

  it("publishes the end of a run it started, with the start's context, and stops polling", async () => {
    const context = { gesture: "empty" };
    const store = await followRun({}, 1000, context);
    const last = ended({ deleted_count: 50000, snapshots_with_deleted: [] });
    getScrapheapRun.mockResolvedValue({ run: last, retryAfterMs: null });

    await vi.advanceTimersByTimeAsync(1000);
    expect(store.active).toBe(false);
    expect(store.outcome).toEqual({ run: last, context, startedHere: true });
    expect(toRaw(store.outcome).context).toBe(context);

    getScrapheapRun.mockClear();
    await vi.advanceTimersByTimeAsync(60000);
    expect(getScrapheapRun).not.toHaveBeenCalled();
  });

  it("clears an outcome only for the run it names", async () => {
    const store = await followRun();
    getScrapheapRun.mockResolvedValue({ run: ended(), retryAfterMs: null });
    await vi.advanceTimersByTimeAsync(1000);

    store.acknowledgeOutcome("run-0");
    expect(store.outcome?.run.id).toBe("run-1");
    store.acknowledgeOutcome("run-1");
    expect(store.outcome).toBe(null);
  });

  it("stops following a run the server no longer has, without an outcome", async () => {
    const store = await followRun();
    getScrapheapRun.mockRejectedValue(
      refusal(404, { code: "scrapheap_run_not_found", message: "" }),
    );

    await vi.advanceTimersByTimeAsync(1000);
    expect(store.run).toBe(null);
    expect(store.unreachable).toBe(false);
    expect(store.outcome).toBe(null);
    getScrapheapRun.mockClear();
    await vi.advanceTimersByTimeAsync(60000);
    expect(getScrapheapRun).not.toHaveBeenCalled();
  });

  it("backs off while the run cannot be read, and recovers", async () => {
    const store = await followRun();
    getScrapheapRun.mockRejectedValue(NO_ANSWER);

    await vi.advanceTimersByTimeAsync(1000);
    expect(store.unreachable).toBe(true);
    expect(store.run.id).toBe("run-1");
    // 2 s doubled per failure: 4 s after the first.
    getScrapheapRun.mockClear();
    await vi.advanceTimersByTimeAsync(3999);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);
    // Then 8 s after the second.
    getScrapheapRun.mockClear();
    getScrapheapRun.mockResolvedValue({
      run: runFixture({ processed: 2500 }),
      retryAfterMs: 1000,
    });
    await vi.advanceTimersByTimeAsync(7999);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(store.unreachable).toBe(false);
    expect(store.run.processed).toBe(2500);
  });

  it("drops a poll answer that left before a Stop and lands after it", async () => {
    const store = await followRun();
    const stale = deferred();
    getScrapheapRun.mockReturnValue(stale.promise);
    await vi.advanceTimersByTimeAsync(1000);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);

    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 1000 });
    expect(await store.cancel()).toEqual({ ok: true, run: stopping });
    expect(cancelScrapheapRun).toHaveBeenCalledWith("run-1");

    stale.resolve({ run: runFixture({ processed: 500 }), retryAfterMs: 1000 });
    await vi.advanceTimersByTimeAsync(0);
    expect(toRaw(store.run)).toBe(stopping);
  });

  it("ignores a poll that left before a Stop and failed after it", async () => {
    const store = await followRun();
    const stale = deferred();
    getScrapheapRun.mockReturnValue(stale.promise);
    await vi.advanceTimersByTimeAsync(1000);

    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 1000 });
    await store.cancel();

    stale.reject(NO_ANSWER);
    await vi.advanceTimersByTimeAsync(0);
    // The Stop's answer is the newer word: the run is reachable, and the next
    // read keeps the interval that answer named rather than backing off.
    expect(store.unreachable).toBe(false);
    getScrapheapRun.mockReset();
    getScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 1000 });
    await vi.advanceTimersByTimeAsync(1000);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);
  });

  it("does nothing on Stop when no run is going", async () => {
    const store = useScrapheapRunStore();
    expect(await store.cancel()).toEqual({ ok: true, run: null });
    expect(cancelScrapheapRun).not.toHaveBeenCalled();
  });

  it("publishes a run its Stop ended at once, before any slice ran", async () => {
    const store = await followRun({ status: "pending", started_at: null });
    const stopped = ended({ status: "cancelled", started_at: null });
    cancelScrapheapRun.mockResolvedValue({ run: stopped, retryAfterMs: null });

    await store.cancel();
    expect(store.outcome).toEqual({ run: stopped, context: null, startedHere: true });
  });
});

describe("useScrapheapRunStore - refusals", () => {
  it("follows the run a refusal names, as somebody else's", async () => {
    const store = useScrapheapRunStore();
    const other = runFixture({ id: "run-9", kind: "restore" });
    startScrapheapRun.mockRejectedValue(
      refusal(409, {
        code: "scrapheap_run_active",
        message: "A Scrapheap run is already going.",
        run: other,
      }),
    );

    const result = await store.start({ kind: "purge", confirmToken: "example-token" });
    expect(result.ok).toBe(false);
    expect(result.error.code).toBe("scrapheap_run_active");
    expect(result.error.run).toBe(other);
    expect(toRaw(store.run)).toBe(other);
    expect(store.action).toBe(null);

    // Its end is an outcome, but not this tab's gesture's.
    getScrapheapRun.mockResolvedValue({
      run: ended({ id: "run-9", kind: "restore" }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(2000);
    expect(store.outcome.startedHere).toBe(false);
    expect(store.outcome.context).toBe(null);
  });

  it("reports a refused token as it was served, with no run", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockRejectedValue(
      refusal(409, "The confirmation has already been used."),
    );

    const result = await store.start({ kind: "purge", confirmToken: "example-token" });
    expect(result).toEqual({
      ok: false,
      error: {
        kind: "conflict",
        status: 409,
        detail: "The confirmation has already been used.",
        code: null,
        run: null,
      },
    });
    expect(store.run).toBe(null);
    expect(getLatestScrapheapRun).not.toHaveBeenCalled();
  });

  it("looks for the run an unanswered start may have queued", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockRejectedValue(NO_ANSWER);
    const queued = runFixture({ status: "pending" });
    getLatestScrapheapRun.mockResolvedValue(queued);

    const result = await store.start({ kind: "purge", confirmToken: "example-token" });
    expect(result.ok).toBe(false);
    expect(result.error.kind).toBe("timeout");
    expect(toRaw(result.error.run)).toBe(queued);
    expect(toRaw(store.run)).toBe(queued);
    expect(store.active).toBe(true);
  });

  it("reports an unanswered start with no run when none is going", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockRejectedValue(NO_ANSWER);
    // The newest run is an old one that already ended: not this start's.
    getLatestScrapheapRun.mockResolvedValue(ended({ id: "run-0" }));

    const result = await store.start({ kind: "restore" });
    expect(result.ok).toBe(false);
    expect(result.error.kind).toBe("timeout");
    expect(result.error.run).toBe(null);
    expect(store.active).toBe(false);
    expect(store.outcome).toBe(null);
  });
});

describe("useScrapheapRunStore - resume", () => {
  it("reads the newest run once per session unless forced", async () => {
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockResolvedValue(ended({ id: "run-0" }));

    // A second ask while the first is on the wire shares its read.
    const [first, second] = await Promise.all([store.resume(), store.resume()]);
    expect(first.id).toBe("run-0");
    expect(second.id).toBe("run-0");
    await store.resume();
    expect(getLatestScrapheapRun).toHaveBeenCalledTimes(1);

    await store.resume({ force: true });
    expect(getLatestScrapheapRun).toHaveBeenCalledTimes(2);
    // A run found already ended changed nothing under the user.
    expect(store.outcome).toBe(null);
  });

  it("follows an active run it finds, and publishes its end as somebody else's", async () => {
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockResolvedValue(runFixture({ id: "run-4" }));
    await store.resume();
    expect(store.active).toBe(true);

    // While it is active, even a forced resume leaves it to its poll.
    await store.resume({ force: true });
    expect(getLatestScrapheapRun).toHaveBeenCalledTimes(1);

    getScrapheapRun.mockResolvedValue({
      run: ended({ id: "run-4" }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(2000);
    expect(store.outcome).toEqual({
      run: expect.objectContaining({ id: "run-4" }),
      context: null,
      startedHere: false,
    });
  });

  it("logs a failed look and shows nothing", async () => {
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockRejectedValue(NO_ANSWER);
    expect(await store.resume()).toBe(null);
    expect(warnSpy).toHaveBeenCalled();
  });
});

describe("useScrapheapRunStore - narration", () => {
  it("keeps a restore run's History steps off the pill, and nothing else", async () => {
    await followRun({ kind: "restore" });
    const holdsBack = installedFilter();

    // Not named by a poll yet: any restore while a restore run is active.
    expect(
      holdsBack({ batch_id: "cli-unnamed", op_type: "pictures.scrapheap.restore" }),
    ).toBe(true);
    expect(holdsBack({ batch_id: "cli-tag", op_type: "pictures.tag" })).toBe(false);

    getScrapheapRun.mockResolvedValue({
      run: ended({ kind: "restore", segment_batch_ids: ["cli-segment-1"] }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    // After the run, its own step is still held back (the log can be re-read
    // after the last poll); a later single restore narrates as usual.
    expect(
      holdsBack({ batch_id: "cli-segment-1", op_type: "pictures.scrapheap.restore" }),
    ).toBe(true);
    expect(
      holdsBack({ batch_id: "cli-later", op_type: "pictures.scrapheap.restore" }),
    ).toBe(false);
  });

  it("does not hold back a restore while a purge run is going", async () => {
    await followRun({ kind: "purge" });
    const holdsBack = installedFilter();
    expect(
      holdsBack({ batch_id: "cli-one", op_type: "pictures.scrapheap.restore" }),
    ).toBe(false);
  });

  it("installs its rule only once a run is followed", async () => {
    useScrapheapRunStore();
    expect(useOperationStore().setNarrationFilter).not.toHaveBeenCalled();
  });
});

describe("useScrapheapRunStore - the restore receipt", () => {
  it("raises one receipt naming the whole run while its step is the newest", async () => {
    const store = useScrapheapRunStore();
    const step = {
      id: 41,
      batch_id: "cli-segment-1",
      op_type: "pictures.scrapheap.restore",
      target_count: 200,
    };
    getUndoState.mockResolvedValue({ can_undo: true, next_undo: step });

    const run = ended({
      kind: "restore",
      restored_count: 1200,
      segment_batch_ids: ["cli-segment-1"],
    });
    expect(await store.raiseRestoreReceipt(run)).toBe(true);
    const operations = useOperationStore();
    expect(operations.buildReceipt).toHaveBeenCalledWith(step, "did");
    expect(operations.showReceipt).toHaveBeenCalledWith(
      expect.objectContaining({
        operationId: 41,
        batchId: "cli-segment-1",
        summary: "Restored 1,200 pictures from the Scrapheap",
        targetCount: 1200,
        mergedCount: 0,
      }),
    );
  });

  it("raises none when another step has become the newest", async () => {
    const store = useScrapheapRunStore();
    getUndoState.mockResolvedValue({
      next_undo: { id: 42, batch_id: "cli-tag", op_type: "pictures.tag" },
    });
    const run = ended({
      kind: "restore",
      restored_count: 3,
      segment_batch_ids: ["cli-segment-1"],
    });
    expect(await store.raiseRestoreReceipt(run)).toBe(false);
    expect(useOperationStore().showReceipt).not.toHaveBeenCalled();
  });

  it("raises none for a run with several steps, a purge, or nothing restored", async () => {
    const store = useScrapheapRunStore();
    expect(
      await store.raiseRestoreReceipt(
        ended({
          kind: "restore",
          restored_count: 9000,
          segment_batch_ids: ["cli-segment-1", "cli-segment-2"],
        }),
      ),
    ).toBe(false);
    expect(
      await store.raiseRestoreReceipt(
        ended({ kind: "purge", deleted_count: 5, segment_batch_ids: ["cli-x"] }),
      ),
    ).toBe(false);
    expect(
      await store.raiseRestoreReceipt(
        ended({ kind: "restore", restored_count: 0, segment_batch_ids: ["cli-x"] }),
      ),
    ).toBe(false);
    expect(getUndoState).not.toHaveBeenCalled();
  });

  it("raises none, and logs, when the undo state cannot be read", async () => {
    const store = useScrapheapRunStore();
    getUndoState.mockRejectedValue(NO_ANSWER);
    const run = ended({
      kind: "restore",
      restored_count: 1,
      segment_batch_ids: ["cli-segment-1"],
    });
    expect(await store.raiseRestoreReceipt(run)).toBe(false);
    expect(warnSpy).toHaveBeenCalled();
    expect(useOperationStore().showReceipt).not.toHaveBeenCalled();
  });

  it("raises none when the session was reset while it looked", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    getUndoState.mockReturnValue(answer.promise);
    const pending = store.raiseRestoreReceipt(
      ended({
        kind: "restore",
        restored_count: 1,
        segment_batch_ids: ["cli-segment-1"],
      }),
    );

    notifySessionReset();
    answer.resolve({ next_undo: { id: 41, batch_id: "cli-segment-1" } });
    expect(await pending).toBe(false);
    expect(useOperationStore().showReceipt).not.toHaveBeenCalled();
  });
});

describe("useScrapheapRunStore - session reset", () => {
  it("forgets the run, its outcome and its steps, and drops answers from before", async () => {
    const store = await followRun({
      kind: "restore",
      segment_batch_ids: ["cli-segment-1"],
    });
    const holdsBack = installedFilter();
    const late = deferred();
    getScrapheapRun.mockReturnValue(late.promise);
    await vi.advanceTimersByTimeAsync(1000);

    notifySessionReset();
    expect(store.run).toBe(null);
    expect(store.outcome).toBe(null);
    expect(
      holdsBack({ batch_id: "cli-segment-1", op_type: "pictures.scrapheap.restore" }),
    ).toBe(false);

    late.resolve({ run: ended({ kind: "restore" }), retryAfterMs: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(store.run).toBe(null);
    expect(store.outcome).toBe(null);
  });

  it("answers a start that was on the wire across the reset as cancelled", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    startScrapheapRun.mockReturnValue(answer.promise);
    const pending = store.start({ kind: "restore" });

    notifySessionReset();
    answer.resolve({ run: runFixture({ kind: "restore" }), retryAfterMs: 1000 });
    expect((await pending).error.kind).toBe("cancelled");
    expect(store.run).toBe(null);
    await vi.advanceTimersByTimeAsync(60000);
    expect(getScrapheapRun).not.toHaveBeenCalled();
  });

  it("looks for the newest run again for the next session", async () => {
    // The other direction: a store that forgot and then never looked again
    // would hide a run the next credential is entitled to follow.
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockResolvedValue(ended({ id: "run-0" }));
    await store.resume();
    await store.resume();
    expect(getLatestScrapheapRun).toHaveBeenCalledTimes(1);

    notifySessionReset();
    getLatestScrapheapRun.mockResolvedValue(runFixture({ id: "run-5" }));
    await store.resume();

    expect(getLatestScrapheapRun).toHaveBeenCalledTimes(2);
    expect(store.run.id).toBe("run-5");
    expect(store.active).toBe(true);
  });
});
