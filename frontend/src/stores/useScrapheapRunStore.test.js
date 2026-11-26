import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { toRaw } from "vue";

vi.mock("../api/pictures", () => ({
  startScrapheapRun: vi.fn(),
  getScrapheapRun: vi.fn(),
  getLatestScrapheapRun: vi.fn(),
  cancelScrapheapRun: vi.fn(),
  undoScrapheapRun: vi.fn(),
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
  undoScrapheapRun,
} from "../api/pictures";
import { getUndoState } from "../api/operations";
import { notifySessionReset } from "../utils/apiClient";
import { useOperationStore } from "./useOperationStore";
import { useNoticeStore } from "./useNoticeStore";
import { useUserPrefsStore } from "./useUserPrefsStore";
import { SCRAPHEAP_GESTURES } from "../utils/scrapheapRun";
import {
  STARTED_RUNS_STORAGE_KEY,
  useScrapheapRunStore,
} from "./useScrapheapRunStore";

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
  // The runs a tab started outlive its store, as they do a reload.
  sessionStorage.clear();
  vi.useFakeTimers();
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  for (const fn of [
    startScrapheapRun,
    getScrapheapRun,
    getLatestScrapheapRun,
    cancelScrapheapRun,
    undoScrapheapRun,
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
      requestId: expect.stringMatching(/^cli-/),
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

  it("reads a run it just started early, before the server's whole second", async () => {
    // A one-picture purge has usually ended long before `Retry-After: 1`.
    await followRun({}, 1000);
    getScrapheapRun.mockResolvedValue({
      run: runFixture(),
      retryAfterMs: 1000,
    });

    await vi.advanceTimersByTimeAsync(249);
    expect(getScrapheapRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);
  });

  it("polls at the server's interval, held to the bounds", async () => {
    const store = await followRun({}, 5000);
    // The first read goes early (above); the rest keep the server's interval.
    getScrapheapRun.mockResolvedValue({
      run: runFixture(),
      retryAfterMs: 5000,
    });
    await vi.advanceTimersByTimeAsync(250);
    expect(getScrapheapRun).toHaveBeenCalledTimes(1);

    getScrapheapRun.mockClear();
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

    // The first, early read of the run just started fails.
    await vi.advanceTimersByTimeAsync(250);
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

  // Under `Prefer: wait` a short run's start answers with the ended run. That
  // answer is the outcome, announced from the start, and nothing is polled.
  it("publishes a run its start answered already ended, and polls nothing", async () => {
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
    const store = useScrapheapRunStore();
    const done = ended({ deleted_count: 3 });
    startScrapheapRun.mockResolvedValue({ run: done, retryAfterMs: null });

    const result = await store.start({
      kind: "purge",
      confirmToken: "example-token",
      context,
    });
    expect(result).toEqual({ ok: true, run: done });
    expect(store.active).toBe(false);
    expect(store.outcome).toEqual({ run: done, context, startedHere: true });
    expect(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)).toBe(null);
    await vi.advanceTimersByTimeAsync(5000);
    expect(getScrapheapRun).not.toHaveBeenCalled();
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
    // A refusal is an answer: it is not sent again.
    expect(startScrapheapRun).toHaveBeenCalledTimes(1);
    expect(getLatestScrapheapRun).not.toHaveBeenCalled();
  });
});

// A start carries the gesture's request id, so one whose answer was lost
// is sent again as is, and the server answers with the run it already started.
describe("useScrapheapRunStore - an unanswered start", () => {
  const sentIds = (fn) => fn.mock.calls.map((call) => call.at(-1).requestId);

  it("sends it again under the same request id, and follows the run as its own", async () => {
    const store = useScrapheapRunStore();
    const queued = runFixture({ status: "pending" });
    startScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockRejectedValueOnce(NO_ANSWER)
      .mockResolvedValueOnce({ run: queued, retryAfterMs: 1000 });
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };

    const pending = store.start({
      kind: "purge",
      confirmToken: "example-token",
      context,
    });
    await vi.advanceTimersByTimeAsync(999);
    expect(startScrapheapRun).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(3000);
    expect(await pending).toEqual({ ok: true, run: queued });

    const ids = sentIds(startScrapheapRun);
    expect(ids).toHaveLength(3);
    expect(ids[0]).toMatch(/^cli-/);
    expect(new Set(ids).size).toBe(1);
    // The same confirmation rides along: the server's replay check comes
    // before the token, so it is not spent twice.
    expect(startScrapheapRun.mock.calls[2][0].confirmToken).toBe(
      "example-token",
    );
    expect(toRaw(store.run)).toBe(queued);
    expect(getLatestScrapheapRun).not.toHaveBeenCalled();

    getScrapheapRun.mockResolvedValue({
      run: ended({ deleted_count: 9 }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.outcome.startedHere).toBe(true);
    expect(store.outcome.context).toEqual(context);
  });

  it("gives up after three unanswered attempts and says so", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockRejectedValue(NO_ANSWER);

    const pending = store.start({ kind: "restore" });
    await vi.advanceTimersByTimeAsync(4000);
    const result = await pending;
    expect(result.ok).toBe(false);
    expect(result.error.kind).toBe("timeout");
    expect(result.error.run).toBe(null);
    expect(startScrapheapRun).toHaveBeenCalledTimes(3);
    expect(store.action).toBe(null);
    expect(store.run).toBe(null);
    await vi.advanceTimersByTimeAsync(60000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(3);
  });

  it("gives each gesture its own request id", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockResolvedValue({ run: ended(), retryAfterMs: null });
    await store.start({ kind: "restore" });
    await store.start({ kind: "restore" });
    const ids = sentIds(startScrapheapRun);
    expect(ids).toHaveLength(2);
    expect(ids[0]).not.toBe(ids[1]);
  });

  it("does not send a Stop again: the user's next Stop is the retry", async () => {
    const store = await followRun();
    cancelScrapheapRun.mockRejectedValue(NO_ANSWER);
    const result = await store.cancel();
    expect(result.error.kind).toBe("timeout");
    await vi.advanceTimersByTimeAsync(5000);
    expect(cancelScrapheapRun).toHaveBeenCalledTimes(1);
  });

  // The bar's Stop is for the run the bar shows. A start still being sent
  // again is no reason to hold it: the run on the bar is most likely the one
  // that start's lost first attempt began.
  it("stops the run the bar shows while the start is being sent again", async () => {
    const store = useScrapheapRunStore();
    const resent = deferred();
    startScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockReturnValueOnce(resent.promise);
    const pending = store.start({ kind: "purge", confirmToken: "example-token" });
    await vi.advanceTimersByTimeAsync(0);
    // The first attempt's run reaches the bar another way: the socket's
    // forced resume.
    getLatestScrapheapRun.mockResolvedValue(runFixture());
    await store.resume({ force: true });
    expect(store.active).toBe(true);
    expect(store.action).toBe("start");

    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    expect(await store.cancel()).toEqual({ ok: true, run: stopping });
    expect(cancelScrapheapRun).toHaveBeenCalledWith("run-1");
    expect(toRaw(store.run)).toBe(stopping);
    // The start is still out: a second start stays held.
    expect(store.action).toBe("start");

    // The resend leaves after the Stop, under the same request id, and the
    // server answers it with the same run, now stopping.
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);
    expect(new Set(sentIds(startScrapheapRun)).size).toBe(1);
    const later = runFixture({ cancel_requested: true, processed: 1500 });
    resent.resolve({ run: later, retryAfterMs: 1000 });
    expect(await pending).toEqual({ ok: true, run: later });
    expect(toRaw(store.run)).toBe(later);
    expect(store.action).toBe(null);

    getScrapheapRun.mockResolvedValue({
      run: ended({ status: "cancelled", processed: 1500 }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.outcome.startedHere).toBe(true);
  });

  /**
   * A start whose first attempt went unanswered, and the run that attempt
   * began brought to the bar by the socket's forced resume.
   */
  async function resendingWithRunOnBar(
    resent,
    { context = null, onBar = runFixture() } = {},
  ) {
    const store = useScrapheapRunStore();
    startScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockReturnValueOnce(resent.promise);
    const pending = store.start({
      kind: "purge",
      confirmToken: "example-token",
      context,
    });
    await vi.advanceTimersByTimeAsync(0);
    getLatestScrapheapRun.mockResolvedValue(onBar);
    await store.resume({ force: true });
    return { store, pending };
  }

  it("stops the run the bar shows while the start's first attempt is still out", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    startScrapheapRun.mockReturnValueOnce(answer.promise);
    const pending = store.start({ kind: "purge", confirmToken: "example-token" });
    getLatestScrapheapRun.mockResolvedValue(runFixture());
    await store.resume({ force: true });

    const stopping = runFixture({ cancel_requested: true });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    expect(await store.cancel()).toEqual({ ok: true, run: stopping });
    expect(cancelScrapheapRun).toHaveBeenCalledWith("run-1");
    answer.resolve({ run: stopping, retryAfterMs: 5000 });
    expect((await pending).ok).toBe(true);
    expect(store.action).toBe(null);
  });

  it("sends no Stop while the start is out and no run is on the bar", async () => {
    const store = useScrapheapRunStore();
    const resent = deferred();
    startScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockReturnValueOnce(resent.promise);
    const pending = store.start({ kind: "purge", confirmToken: "example-token" });
    await vi.advanceTimersByTimeAsync(0);

    expect(await store.cancel()).toEqual({ ok: true, run: null });
    expect(cancelScrapheapRun).not.toHaveBeenCalled();
    expect(store.action).toBe("start");
    // The start carries on as if no Stop had been pressed.
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);
    const queued = runFixture({ status: "pending" });
    resent.resolve({ run: queued, retryAfterMs: 1000 });
    expect(await pending).toEqual({ ok: true, run: queued });
    expect(toRaw(store.run)).toBe(queued);
  });

  it("keeps the Stop's word over a resend that left before it and is answered after", async () => {
    const resent = deferred();
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
    const { store, pending } = await resendingWithRunOnBar(resent, { context });
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    await store.cancel();
    // The server read the run for the resend before it heard the Stop.
    resent.resolve({ run: runFixture({ processed: 500 }), retryAfterMs: 1000 });
    expect((await pending).ok).toBe(true);
    expect(toRaw(store.run)).toBe(stopping);
    expect(store.action).toBe(null);

    // It is still this tab's run: its end comes with the start's context.
    getScrapheapRun.mockResolvedValue({
      run: ended({ status: "cancelled", processed: 1000 }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(5000);
    expect(store.outcome).toEqual({
      run: expect.objectContaining({ id: "run-1", status: "cancelled" }),
      context,
      startedHere: true,
    });
  });

  // The resend leaves after the Stop, but the server read the run for it
  // before it heard the Stop: its answer is newer by ticket, older in fact.
  it("does not show the run going again for a resend read before the Stop it left after", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent);
    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    await store.cancel();
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    resent.resolve({ run: runFixture({ processed: 900 }), retryAfterMs: 1000 });
    expect((await pending).ok).toBe(true);
    expect(toRaw(store.run)).toBe(stopping);
  });

  it("holds a second Stop while the first is out, and lets the first Stop's answer land after the resend's", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent);
    const stopAnswer = deferred();
    cancelScrapheapRun.mockReturnValue(stopAnswer.promise);
    const stopping = store.cancel();
    expect(store.action).toBe("cancel");
    expect(await store.cancel()).toEqual({
      ok: false,
      error: { kind: "busy", status: null, detail: "", code: null, run: null },
    });
    expect(cancelScrapheapRun).toHaveBeenCalledTimes(1);

    // The resend leaves after the Stop and is answered first, read before the
    // server heard the Stop: the run as it then was, going, is the newest word
    // until the Stop's answer lands, and the Stop stays pending.
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);
    resent.resolve({ run: runFixture({ processed: 900 }), retryAfterMs: 1000 });
    expect((await pending).ok).toBe(true);
    expect(store.run.processed).toBe(900);
    expect(store.action).toBe("cancel");

    const asked = runFixture({ cancel_requested: true, processed: 1000 });
    stopAnswer.resolve({ run: asked, retryAfterMs: 5000 });
    expect(await stopping).toEqual({ ok: true, run: asked });
    expect(toRaw(store.run)).toBe(asked);
    expect(store.action).toBe(null);
  });

  // A queued run ends at once on a Stop, so its end is known before the start
  // says whose run it is. It is announced once as somebody else's, then again
  // as this tab's, with nothing left behind for a reload to announce.
  it.each(["before", "after"])(
    "announces as this tab's a run the Stop ended at once, when the resend that left %s the Stop answers",
    async (when) => {
      const resent = deferred();
      const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
      const { store, pending } = await resendingWithRunOnBar(resent, {
        context,
        onBar: runFixture({ status: "pending", started_at: null }),
      });
      if (when === "before") await vi.advanceTimersByTimeAsync(1000);
      const stopped = ended({ status: "cancelled", started_at: null });
      cancelScrapheapRun.mockResolvedValue({ run: stopped, retryAfterMs: null });
      await store.cancel();
      expect(store.outcome).toEqual({
        run: stopped,
        context: null,
        startedHere: false,
      });
      if (when === "after") await vi.advanceTimersByTimeAsync(1000);
      expect(startScrapheapRun).toHaveBeenCalledTimes(2);

      resent.resolve({
        run: ended({ status: "cancelled", started_at: null }),
        retryAfterMs: null,
      });
      expect((await pending).ok).toBe(true);
      expect(store.run.status).toBe("cancelled");
      expect(store.outcome).toEqual({
        run: expect.objectContaining({ id: "run-1", status: "cancelled" }),
        context,
        startedHere: true,
      });
      expect(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)).toBe(null);
    },
  );

  it("does not bring back a run the Stop ended for a resend read before the Stop, and announces it as this tab's", async () => {
    const resent = deferred();
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
    const { store, pending } = await resendingWithRunOnBar(resent, {
      context,
      onBar: runFixture({ status: "pending", started_at: null }),
    });
    const stopped = ended({ status: "cancelled", started_at: null });
    cancelScrapheapRun.mockResolvedValue({ run: stopped, retryAfterMs: null });
    await store.cancel();
    await vi.advanceTimersByTimeAsync(1000);

    resent.resolve({
      run: runFixture({ status: "pending", started_at: null }),
      retryAfterMs: 1000,
    });
    expect((await pending).ok).toBe(true);
    expect(toRaw(store.run)).toBe(stopped);
    expect(store.active).toBe(false);
    expect(store.outcome).toEqual({ run: stopped, context, startedHere: true });
    expect(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)).toBe(null);
  });

  // Only an answer about the run the bar shows can be older than it.
  it("takes the start's own run when the bar shows another run asked to stop", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent, {
      onBar: runFixture({ id: "run-9", cancel_requested: true }),
    });
    await vi.advanceTimersByTimeAsync(1000);
    const queued = runFixture({ status: "pending" });
    resent.resolve({ run: queued, retryAfterMs: 1000 });
    expect(await pending).toEqual({ ok: true, run: queued });
    expect(toRaw(store.run)).toBe(queued);
  });

  // An answer that says the run has ended is never older than the bar: the
  // run finished before the server heard the Stop still out from here.
  it("takes a resend's answer that the run has ended while a Stop is still out", async () => {
    const resent = deferred();
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
    const { store, pending } = await resendingWithRunOnBar(resent, { context });
    const stopAnswer = deferred();
    cancelScrapheapRun.mockReturnValue(stopAnswer.promise);
    const stopping = store.cancel();
    expect(store.action).toBe("cancel");
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    const finished = ended({ deleted_count: 50000, processed: 50000 });
    resent.resolve({ run: finished, retryAfterMs: null });
    expect(await pending).toEqual({ ok: true, run: finished });
    expect(toRaw(store.run)).toBe(finished);
    expect(store.active).toBe(false);
    expect(store.outcome).toEqual({ run: finished, context, startedHere: true });

    stopAnswer.resolve({ run: finished, retryAfterMs: null });
    expect((await stopping).ok).toBe(true);
    expect(store.active).toBe(false);
    expect(store.action).toBe(null);
  });

  // A refusal names the run as the server read it, which can be before it
  // heard the Stop: it meets the same rules as any other answer.
  it("does not show the run going again for a refusal read before the Stop it left after", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent);
    const stopping = runFixture({ cancel_requested: true, processed: 1000 });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    await store.cancel();
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    resent.reject(
      refusal(409, {
        code: "scrapheap_run_active",
        message: "A Scrapheap run is already going.",
        run: runFixture({ processed: 900 }),
      }),
    );
    expect((await pending).error.code).toBe("scrapheap_run_active");
    expect(toRaw(store.run)).toBe(stopping);
  });

  it("keeps the Stop's answer when a refusal read before the Stop lands first", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent);
    const stopAnswer = deferred();
    cancelScrapheapRun.mockReturnValue(stopAnswer.promise);
    const stopping = store.cancel();
    await vi.advanceTimersByTimeAsync(1000);
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    resent.reject(
      refusal(409, {
        code: "scrapheap_run_active",
        message: "A Scrapheap run is already going.",
        run: runFixture({ processed: 900 }),
      }),
    );
    expect((await pending).ok).toBe(false);
    expect(store.action).toBe("cancel");

    const asked = runFixture({ cancel_requested: true, processed: 1000 });
    stopAnswer.resolve({ run: asked, retryAfterMs: 5000 });
    expect(await stopping).toEqual({ ok: true, run: asked });
    expect(toRaw(store.run)).toBe(asked);
  });

  it("clears a Stop and a start that were out across a session reset", async () => {
    const resent = deferred();
    const { store, pending } = await resendingWithRunOnBar(resent);
    const stopAnswer = deferred();
    cancelScrapheapRun.mockReturnValue(stopAnswer.promise);
    const stopping = store.cancel();

    notifySessionReset();
    expect(store.action).toBe(null);
    stopAnswer.resolve({
      run: runFixture({ cancel_requested: true }),
      retryAfterMs: 5000,
    });
    expect((await stopping).error.kind).toBe("cancelled");
    await vi.advanceTimersByTimeAsync(1000);
    expect((await pending).ok).toBe(false);
    expect(startScrapheapRun).toHaveBeenCalledTimes(1);
    expect(store.run).toBe(null);

    // The next session's Stop is held by nothing from the last one.
    getLatestScrapheapRun.mockResolvedValue(runFixture({ id: "run-5" }));
    await store.resume();
    cancelScrapheapRun.mockResolvedValue({
      run: runFixture({ id: "run-5", cancel_requested: true }),
      retryAfterMs: 5000,
    });
    expect((await store.cancel()).ok).toBe(true);
    expect(cancelScrapheapRun).toHaveBeenLastCalledWith("run-5");
    expect(store.action).toBe(null);
  });

  it("sends nothing more once the session was reset during the wait", async () => {
    const store = useScrapheapRunStore();
    startScrapheapRun.mockRejectedValue(NO_ANSWER);
    const pending = store.start({ kind: "restore" });
    await vi.advanceTimersByTimeAsync(0);
    notifySessionReset();
    await vi.advanceTimersByTimeAsync(5000);
    expect((await pending).ok).toBe(false);
    expect(startScrapheapRun).toHaveBeenCalledTimes(1);
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

  // The socket forces a resume whenever History names a run the bar does not
  // show, and that look can cross the user's own start on the wire. The run
  // the server started later is the newer word, whichever answer lands first.
  it("follows its start's run when a forced look at an older, ended run lands first", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    startScrapheapRun.mockReturnValueOnce(answer.promise);
    const context = { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false };
    const pending = store.start({
      kind: "purge",
      confirmToken: "example-token",
      context,
    });
    getLatestScrapheapRun.mockResolvedValue(
      ended({ id: "run-0", created_at: "2026-11-24T09:00:00Z" }),
    );
    await store.resume({ force: true });
    expect(store.run.id).toBe("run-0");

    const started = runFixture({ created_at: "2026-11-24T10:00:00.250000+00:00" });
    answer.resolve({ run: started, retryAfterMs: 1000 });
    expect(await pending).toEqual({ ok: true, run: started });
    expect(toRaw(store.run)).toBe(started);
    expect(store.active).toBe(true);

    getScrapheapRun.mockResolvedValue({
      run: ended({ deleted_count: 9 }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(store.outcome).toEqual({
      run: expect.objectContaining({ id: "run-1", deleted_count: 9 }),
      context,
      startedHere: true,
    });
  });

  it("keeps its start's run when a forced look at an older run lands after it", async () => {
    const store = useScrapheapRunStore();
    const answer = deferred();
    startScrapheapRun.mockReturnValueOnce(answer.promise);
    const latest = deferred();
    getLatestScrapheapRun.mockReturnValueOnce(latest.promise);
    const pending = store.start({ kind: "purge", confirmToken: "example-token" });
    const looking = store.resume({ force: true });

    const started = runFixture({ created_at: "2026-11-24T10:00:00.250000+00:00" });
    answer.resolve({ run: started, retryAfterMs: 1000 });
    expect((await pending).ok).toBe(true);
    latest.resolve(ended({ id: "run-0", created_at: "2026-11-24T09:00:00Z" }));
    await looking;
    expect(toRaw(store.run)).toBe(started);
    expect(store.active).toBe(true);
  });

  it("logs a failed look and shows nothing", async () => {
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockRejectedValue(NO_ANSWER);
    expect(await store.resume()).toBe(null);
    expect(warnSpy).toHaveBeenCalled();
  });
});

// A reload loses the store but not the tab: the run it started still ends as
// this tab's own, with the gesture that started it, so the grid settles it the
// way it would have without the reload (the receipt, the locked-set notice).
describe("useScrapheapRunStore - a reload", () => {
  /** Drop the store as a reload does; the tab's session storage stays. */
  function reload(store) {
    store.$dispose();
    setActivePinia(createPinia());
    return useScrapheapRunStore();
  }

  it("keeps only the gesture and the scope of a run it started, never the ids", async () => {
    await followRun({ kind: "restore" }, 1000, {
      gesture: SCRAPHEAP_GESTURES.RESTORE_SELECTION,
      ids: [1, 2, 3],
      scoped: false,
    });
    expect(
      JSON.parse(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)),
    ).toEqual({
      "run-1": { gesture: SCRAPHEAP_GESTURES.RESTORE_SELECTION, scoped: false },
    });
  });

  it("announces a run that ends after the reload as its own", async () => {
    const before = await followRun({ kind: "restore" }, 1000, {
      gesture: SCRAPHEAP_GESTURES.RESTORE_ALL,
    });
    const store = reload(before);
    getLatestScrapheapRun.mockResolvedValue(runFixture({ kind: "restore" }));
    await store.resume();
    expect(store.active).toBe(true);

    getScrapheapRun.mockResolvedValue({
      run: ended({ kind: "restore" }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(2000);
    expect(store.outcome).toEqual({
      run: expect.objectContaining({ id: "run-1", active: false }),
      context: { gesture: SCRAPHEAP_GESTURES.RESTORE_ALL, scoped: false },
      startedHere: true,
    });
    expect(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)).toBe(null);
  });

  it("announces a run that ended during the reload once, and only once", async () => {
    const before = await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.EMPTY,
    });
    const store = reload(before);
    getLatestScrapheapRun.mockResolvedValue(ended());
    await store.resume();
    expect(store.outcome).toEqual({
      run: expect.objectContaining({ id: "run-1", active: false }),
      context: { gesture: SCRAPHEAP_GESTURES.EMPTY, scoped: false },
      startedHere: true,
    });

    store.acknowledgeOutcome("run-1");
    await store.resume({ force: true });
    expect(store.outcome).toBe(null);
    // Nor does the next reload announce it again.
    const after = reload(store);
    await after.resume();
    expect(after.outcome).toBe(null);
  });

  it("forgets the runs it started on a session reset", async () => {
    const before = await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.EMPTY,
    });
    notifySessionReset();
    expect(sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY)).toBe(null);

    const store = reload(before);
    getLatestScrapheapRun.mockResolvedValue(ended());
    await store.resume();
    expect(store.outcome).toBe(null);
  });

  it("logs a stored value it cannot read, and announces the run as somebody else's", async () => {
    sessionStorage.setItem(STARTED_RUNS_STORAGE_KEY, "{not json");
    const store = useScrapheapRunStore();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining(STARTED_RUNS_STORAGE_KEY),
      expect.any(SyntaxError),
    );
    getLatestScrapheapRun.mockResolvedValue(ended());
    await store.resume();
    expect(store.outcome).toBe(null);
  });

  // A browser set to block storage throws on the read too, and every store
  // that uses this one (the tasks store among them) would fail to set up.
  it("sets up with no saved runs, and logs it, when the browser refuses storage", async () => {
    const refusal = new DOMException("The operation is insecure.", "SecurityError");
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw refusal;
    });
    try {
      const store = useScrapheapRunStore();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining(STARTED_RUNS_STORAGE_KEY),
        refusal,
      );
      getLatestScrapheapRun.mockResolvedValue(ended());
      await store.resume();
      expect(store.outcome).toBe(null);
    } finally {
      getItem.mockRestore();
    }
  });
});

// The store, not the grid, says how a run ended. The grid is unmounted
// in most of the app's views, and a run outlives the view it started in. No
// component is mounted in these tests.
describe("useScrapheapRunStore - announcing an end, in any view", () => {
  const keys = () => useNoticeStore().notices.map((n) => n.key);

  async function endTheFollowedRun(last) {
    getScrapheapRun.mockResolvedValue({ run: last, retryAfterMs: null });
    // A run found by a resume is read again at the default interval (2 s); one
    // this tab started, at its own hint of 1 s. Both have polled by now.
    await vi.advanceTimersByTimeAsync(2000);
    // announce() may await the undo state for a restore.
    await vi.advanceTimersByTimeAsync(0);
  }

  it("raises a notice for a run started elsewhere", async () => {
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockResolvedValue(runFixture({ id: "run-4" }));
    await store.resume();
    await endTheFollowedRun(ended({ id: "run-4", deleted_count: 50000 }));
    expect(keys()).toEqual(["scrapheap-run"]);
    // A purge it did not start offers no snapshots dialog.
    expect(store.snapshotsWithDeletedOpen).toBe(false);
  });

  it("raises a notice for an Empty scrapheap it started, and offers the snapshots", async () => {
    const store = await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.EMPTY,
    });
    const snapshots = [{ id: 3, name: "Before the clean-up" }];
    await endTheFollowedRun(
      ended({ deleted_count: 50000, snapshots_with_deleted: snapshots }),
    );
    expect(keys()).toEqual(["scrapheap-run"]);
    expect(store.snapshotsWithDeletedOpen).toBe(true);
    expect(store.snapshotsWithDeleted).toEqual(snapshots);
  });

  it("offers no snapshots once the user asked not to be shown them", async () => {
    useUserPrefsStore().hidePurgeSnapshotWarning = true;
    const store = await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.EMPTY,
    });
    await endTheFollowedRun(
      ended({ snapshots_with_deleted: [{ id: 3, name: "Before" }] }),
    );
    expect(store.snapshotsWithDeletedOpen).toBe(false);
  });

  it("raises the locked-set warning instead, for a selection that met locked sets", async () => {
    await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.PURGE_SELECTION,
    });
    await endTheFollowedRun(
      ended({ deleted_count: 3, skipped_locked_count: 2 }),
    );
    expect(keys()).toEqual(["delete-skipped-locked"]);
  });

  it("says in the locked-set warning what was binned since the confirm", async () => {
    await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.PURGE_SELECTION,
    });
    await endTheFollowedRun(
      ended({ deleted_count: 1200, skipped_locked_count: 2, arrived_since: 3 }),
    );
    const [notice] = useNoticeStore().notices;
    expect(notice.key).toBe("delete-skipped-locked");
    expect(notice.text).toBe(
      "1,200 pictures deleted forever; 2 are in locked sets and were kept. " +
        "3 pictures binned since you confirmed are still in the Scrapheap.",
    );
  });

  it("forgets an offered dialog on a session reset", async () => {
    const store = await followRun({}, 1000, {
      gesture: SCRAPHEAP_GESTURES.EMPTY,
    });
    await endTheFollowedRun(
      ended({ snapshots_with_deleted: [{ id: 3, name: "Before" }] }),
    );
    store.forget();
    expect(store.snapshotsWithDeletedOpen).toBe(false);
    expect(store.snapshotsWithDeleted).toEqual([]);
  });
});

describe("useScrapheapRunStore - narration", () => {
  it("keeps a restore run's History steps off the pill, and nothing else", async () => {
    await followRun({ kind: "restore" });
    const holdsBack = installedFilter();

    // Not named by a poll yet: a server-minted restore while a restore run is
    // active (a run's slices are always minted by the server).
    expect(
      holdsBack({ batch_id: "srv-unnamed", op_type: "pictures.scrapheap.restore" }),
    ).toBe(true);
    expect(holdsBack({ batch_id: "cli-tag", op_type: "pictures.tag" })).toBe(false);

    getScrapheapRun.mockResolvedValue({
      run: ended({ kind: "restore", segment_batch_ids: ["srv-segment-1"] }),
      retryAfterMs: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    // After the run, its own step is still held back (the log can be re-read
    // after the last poll); a later single restore narrates as usual.
    expect(
      holdsBack({ batch_id: "srv-segment-1", op_type: "pictures.scrapheap.restore" }),
    ).toBe(true);
    expect(
      holdsBack({ batch_id: "srv-later", op_type: "pictures.scrapheap.restore" }),
    ).toBe(false);
  });

  it("does not hold back a restore while a purge run is going", async () => {
    await followRun({ kind: "purge" });
    const holdsBack = installedFilter();
    expect(
      holdsBack({ batch_id: "srv-one", op_type: "pictures.scrapheap.restore" }),
    ).toBe(false);
  });

  // The lightbox's restore sends the client's own id, so its pill and its
  // Undo are not swallowed by a restore run going at the same time.
  it("narrates a restore the user sent while a restore run is going", async () => {
    await followRun({ kind: "restore" });
    const holdsBack = installedFilter();
    expect(
      holdsBack({
        batch_id: "cli-lightbox",
        op_type: "pictures.scrapheap.restore",
      }),
    ).toBe(false);
    expect(
      holdsBack({
        batch_id: "srv-slice",
        op_type: "pictures.scrapheap.restore",
      }),
    ).toBe(true);
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
      batch_id: "srv-segment-1",
      op_type: "pictures.scrapheap.restore",
      target_count: 200,
    };
    getUndoState.mockResolvedValue({ can_undo: true, next_undo: step });

    const run = ended({
      kind: "restore",
      restored_count: 1200,
      segment_batch_ids: ["srv-segment-1"],
    });
    expect(await store.raiseRestoreReceipt(run)).toBe(true);
    const operations = useOperationStore();
    expect(operations.buildReceipt).toHaveBeenCalledWith(step, "did");
    expect(operations.showReceipt).toHaveBeenCalledWith(
      expect.objectContaining({
        operationId: 41,
        batchId: "srv-segment-1",
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
      segment_batch_ids: ["srv-segment-1"],
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
          segment_batch_ids: ["srv-segment-1", "srv-segment-2"],
        }),
      ),
    ).toBe(false);
    expect(
      await store.raiseRestoreReceipt(
        ended({ kind: "purge", deleted_count: 5, segment_batch_ids: ["srv-x"] }),
      ),
    ).toBe(false);
    expect(
      await store.raiseRestoreReceipt(
        ended({ kind: "restore", restored_count: 0, segment_batch_ids: ["srv-x"] }),
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
      segment_batch_ids: ["srv-segment-1"],
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
        segment_batch_ids: ["srv-segment-1"],
      }),
    );

    notifySessionReset();
    answer.resolve({ next_undo: { id: 41, batch_id: "srv-segment-1" } });
    expect(await pending).toBe(false);
    expect(useOperationStore().showReceipt).not.toHaveBeenCalled();
  });
});

describe("useScrapheapRunStore - session reset", () => {
  it("forgets the run, its outcome and its steps, and drops answers from before", async () => {
    const store = await followRun({
      kind: "restore",
      segment_batch_ids: ["srv-segment-1"],
    });
    const holdsBack = installedFilter();
    const late = deferred();
    getScrapheapRun.mockReturnValue(late.promise);
    await vi.advanceTimersByTimeAsync(1000);

    notifySessionReset();
    expect(store.run).toBe(null);
    expect(store.outcome).toBe(null);
    expect(
      holdsBack({ batch_id: "srv-segment-1", op_type: "pictures.scrapheap.restore" }),
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

  // A request from the last session settles under the next one's request:
  // it must leave that request's pending state alone.
  it("keeps the next session's start held when a start from before the reset settles", async () => {
    const store = useScrapheapRunStore();
    const before = deferred();
    const after = deferred();
    startScrapheapRun
      .mockReturnValueOnce(before.promise)
      .mockReturnValueOnce(after.promise);
    const old = store.start({ kind: "restore" });
    notifySessionReset();
    const current = store.start({ kind: "restore" });
    expect(store.action).toBe("start");

    before.resolve({ run: runFixture({ kind: "restore" }), retryAfterMs: 1000 });
    expect((await old).error.kind).toBe("cancelled");
    expect(store.action).toBe("start");
    expect((await store.start({ kind: "restore" })).error.kind).toBe("busy");
    expect(startScrapheapRun).toHaveBeenCalledTimes(2);

    const queued = runFixture({ id: "run-5", kind: "restore", status: "pending" });
    after.resolve({ run: queued, retryAfterMs: 1000 });
    expect(await current).toEqual({ ok: true, run: queued });
    expect(store.action).toBe(null);
  });

  it("keeps the next session's Stop held when a Stop from before the reset settles", async () => {
    const store = await followRun();
    const before = deferred();
    const after = deferred();
    cancelScrapheapRun
      .mockReturnValueOnce(before.promise)
      .mockReturnValueOnce(after.promise);
    const old = store.cancel();
    notifySessionReset();
    getLatestScrapheapRun.mockResolvedValue(runFixture({ id: "run-5" }));
    await store.resume();
    const current = store.cancel();
    expect(store.action).toBe("cancel");

    before.reject(NO_ANSWER);
    expect((await old).ok).toBe(false);
    expect(store.action).toBe("cancel");
    expect((await store.cancel()).error.kind).toBe("busy");
    expect(cancelScrapheapRun).toHaveBeenCalledTimes(2);

    const stopping = runFixture({ id: "run-5", cancel_requested: true });
    after.resolve({ run: stopping, retryAfterMs: 5000 });
    expect(await current).toEqual({ ok: true, run: stopping });
    expect(store.action).toBe(null);
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

// A restore run can be undone as a run of its own, offered on the
// restore's outcome notice, and an undo History blocked can be finished.
describe("useScrapheapRunStore - undoing a restore", () => {
  const RESTORED = {
    kind: "restore",
    restored_count: 1200,
    segment_batch_ids: ["srv-segment-1", "srv-segment-2"],
  };

  function undoRun(over = {}) {
    return runFixture({
      id: "run-u",
      kind: "undo",
      target_run_id: "run-1",
      total: 1200,
      undone_count: 0,
      ...over,
    });
  }

  async function endTheFollowedRun(last) {
    getScrapheapRun.mockResolvedValue({ run: last, retryAfterMs: null });
    await vi.advanceTimersByTimeAsync(2000);
    await vi.advanceTimersByTimeAsync(0);
  }

  const outcomeNotice = () =>
    useNoticeStore().notices.find((n) => n.key === "scrapheap-run");

  it("starts the undo under a request id and follows it as this tab's", async () => {
    const store = useScrapheapRunStore();
    const going = undoRun();
    undoScrapheapRun.mockResolvedValue({ run: going, retryAfterMs: 1000 });

    expect(await store.undoRestore("run-1")).toEqual({ ok: true, run: going });
    expect(undoScrapheapRun).toHaveBeenCalledWith("run-1", {
      requestId: expect.stringMatching(/^cli-/),
    });
    expect(toRaw(store.run)).toBe(going);

    await endTheFollowedRun(
      undoRun({ status: "complete", active: false, undone_count: 1200 }),
    );
    expect(store.outcome.startedHere).toBe(true);
    expect(store.outcome.context.gesture).toBe(SCRAPHEAP_GESTURES.UNDO_RESTORE);
    const notice = outcomeNotice();
    expect(notice.text).toBe("Put 1,200 pictures back in the Scrapheap.");
    expect(notice.action).toBe(null);
    // An undo is no restore: it raises no restore receipt.
    expect(getUndoState).not.toHaveBeenCalled();
  });

  it("sends an unanswered undo again under the same request id", async () => {
    const store = useScrapheapRunStore();
    undoScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockResolvedValueOnce({ run: undoRun(), retryAfterMs: 1000 });
    const pending = store.undoRestore("run-1");
    await vi.advanceTimersByTimeAsync(1000);
    expect((await pending).ok).toBe(true);
    const [first, second] = undoScrapheapRun.mock.calls;
    expect(second).toEqual(first);
  });

  it("stops the run the bar shows while an undo is being sent again", async () => {
    const store = useScrapheapRunStore();
    const resent = deferred();
    undoScrapheapRun
      .mockRejectedValueOnce(NO_ANSWER)
      .mockReturnValueOnce(resent.promise);
    const pending = store.undoRestore("run-1");
    await vi.advanceTimersByTimeAsync(0);
    getLatestScrapheapRun.mockResolvedValue(undoRun());
    await store.resume({ force: true });
    expect(store.action).toBe("undo");

    const stopping = undoRun({ cancel_requested: true });
    cancelScrapheapRun.mockResolvedValue({ run: stopping, retryAfterMs: 5000 });
    expect(await store.cancel()).toEqual({ ok: true, run: stopping });
    expect(cancelScrapheapRun).toHaveBeenCalledWith("run-u");
    expect(store.action).toBe("undo");

    await vi.advanceTimersByTimeAsync(1000);
    resent.resolve({ run: stopping, retryAfterMs: 5000 });
    expect((await pending).ok).toBe(true);
    expect(store.action).toBe(null);
  });

  it("offers Undo restore on the notice of a restore it started, and undoes that run", async () => {
    await followRun({ kind: "restore" }, 1000, {
      gesture: SCRAPHEAP_GESTURES.RESTORE_ALL,
    });
    await endTheFollowedRun(ended({ ...RESTORED, can_undo: true }));
    const notice = outcomeNotice();
    expect(notice.action.label).toBe("Undo restore");

    undoScrapheapRun.mockResolvedValue({ run: undoRun(), retryAfterMs: 1000 });
    useNoticeStore().invokeAction(notice.id);
    await vi.advanceTimersByTimeAsync(0);
    expect(undoScrapheapRun).toHaveBeenCalledWith("run-1", expect.anything());
  });

  it("offers no Undo for a restore that cannot be undone, or that another tab started", async () => {
    await followRun({ kind: "restore" }, 1000, {
      gesture: SCRAPHEAP_GESTURES.RESTORE_ALL,
    });
    await endTheFollowedRun(ended({ ...RESTORED, can_undo: false }));
    expect(outcomeNotice().action).toBe(null);

    useNoticeStore().clear();
    const store = useScrapheapRunStore();
    getLatestScrapheapRun.mockResolvedValue(
      runFixture({ id: "run-4", kind: "restore" }),
    );
    await store.resume({ force: true });
    await endTheFollowedRun(
      ended({ ...RESTORED, id: "run-4", can_undo: true }),
    );
    expect(outcomeNotice().action).toBe(null);
  });

  it("says what History entry blocked an undo, and offers to finish it", async () => {
    const store = useScrapheapRunStore();
    undoScrapheapRun.mockResolvedValue({ run: undoRun(), retryAfterMs: 1000 });
    await store.undoRestore("run-1");
    const blockedBy =
      "Undo stopped: a later change in History (\u201ctest-label\u201d, entry 7) must be undone first.";
    await endTheFollowedRun(
      undoRun({
        status: "blocked",
        active: false,
        undone_count: 400,
        error_code: "history_moved_on",
        error: blockedBy,
      }),
    );
    const notice = outcomeNotice();
    expect(notice.level).toBe("warning");
    expect(notice.text).toBe(blockedBy);
    expect(notice.action.label).toBe("Finish undo");

    undoScrapheapRun.mockClear();
    undoScrapheapRun.mockResolvedValue({
      run: undoRun({ id: "run-u2" }),
      retryAfterMs: 1000,
    });
    useNoticeStore().invokeAction(notice.id);
    await vi.advanceTimersByTimeAsync(0);
    expect(undoScrapheapRun).toHaveBeenCalledWith("run-1", expect.anything());
  });

  it("says why the notice's undo was refused", async () => {
    await followRun({ kind: "restore" }, 1000, {
      gesture: SCRAPHEAP_GESTURES.RESTORE_ALL,
    });
    await endTheFollowedRun(ended({ ...RESTORED, can_undo: true }));
    undoScrapheapRun.mockRejectedValue(
      refusal(409, {
        code: "nothing_to_undo",
        message: "fake-server-message",
        run: ended({ ...RESTORED, can_undo: false }),
      }),
    );
    useNoticeStore().invokeAction(outcomeNotice().id);
    await vi.advanceTimersByTimeAsync(0);
    const notice = outcomeNotice();
    expect(notice.level).toBe("error");
    expect(notice.text).toBe("That restore has already been undone.");
    expect(undoScrapheapRun).toHaveBeenCalledTimes(1);
  });
});
