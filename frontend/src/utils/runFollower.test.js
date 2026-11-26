// The poll both background runs share (Scrapheap runs, Duplicates auto-stack
// runs). These pin the rules every answer meets, in one place: the interval is
// the server's, held to bounds; an answer that is not a newer word than the
// run held is dropped; a run that is gone is forgotten; failed reads back off
// and recover; a resume reads the newest run once per session; a reset drops
// what was on the wire.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import {
  createRunFollower,
  RUN_POLL_MS,
  RUN_POLL_MIN_MS,
  RUN_POLL_MAX_MS,
  RUN_FIRST_POLL_MS,
} from "./runFollower";

const ACTIVE = { id: "r1", active: true };
const DONE = { id: "r1", active: false };

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeFollower(overrides = {}) {
  const read = vi.fn();
  const readLatest = vi.fn();
  const onAdopt = vi.fn();
  const follower = createRunFollower({
    logPrefix: "[test]",
    runName: "run",
    read,
    readLatest,
    // The order the server started its runs in, as each store reads it off
    // the run: here a plain number.
    order: (run) => run.seq,
    onAdopt,
    ...overrides,
  });
  return { follower, read, readLatest, onAdopt };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("runFollower: the poll interval", () => {
  it("polls at the server's hint, held to the bounds", async () => {
    const { follower, read } = makeFollower();
    read.mockResolvedValue({ run: ACTIVE, retryAfterMs: 10 });

    follower.adopt(ACTIVE, 10, follower.claim());
    await vi.advanceTimersByTimeAsync(RUN_POLL_MIN_MS - 1);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);

    read.mockClear();
    follower.adopt(ACTIVE, 10 * 60 * 1000, follower.claim());
    await vi.advanceTimersByTimeAsync(RUN_POLL_MAX_MS - 1);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("reads a run the user just started early, once", async () => {
    const { follower, read } = makeFollower();
    read.mockResolvedValue({ run: ACTIVE, retryAfterMs: 1000 });

    follower.adopt(ACTIVE, 1000, follower.claim({ fromGesture: true }));
    await vi.advanceTimersByTimeAsync(RUN_FIRST_POLL_MS - 1);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);
    // The poll's own answer keeps the server's interval.
    await vi.advanceTimersByTimeAsync(999);
    expect(read).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(2);
  });

  it("keeps the server's interval after a Stop's answer, and after a resume", async () => {
    const { follower, read } = makeFollower();
    read.mockResolvedValue({ run: ACTIVE, retryAfterMs: 1000 });

    // A run found by a resume, not started by this gesture.
    follower.adopt(ACTIVE, 1000, follower.claim());
    await vi.advanceTimersByTimeAsync(999);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);

    // A Stop is a gesture on the run already followed.
    read.mockClear();
    const stopping = { ...ACTIVE, cancel_requested: true };
    follower.adopt(stopping, 1000, follower.claim({ fromGesture: true }));
    await vi.advanceTimersByTimeAsync(999);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("uses the default when the answer names no interval", async () => {
    const { follower, read } = makeFollower();
    read.mockResolvedValue({ run: DONE, retryAfterMs: null });

    follower.adopt(ACTIVE, null, follower.claim());
    await vi.advanceTimersByTimeAsync(RUN_POLL_MS - 1);
    expect(read).toHaveBeenCalledTimes(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledWith("r1");
  });

  it("stops polling once the run has ended, and calls that the end", async () => {
    const { follower, read, onAdopt } = makeFollower();
    read.mockResolvedValue({ run: DONE, retryAfterMs: 1000 });

    follower.adopt(ACTIVE, 1000, follower.claim());
    await vi.advanceTimersByTimeAsync(1000);
    expect(follower.run.value).toEqual(DONE);
    expect(onAdopt).toHaveBeenLastCalledWith(DONE, {
      previous: ACTIVE,
      ended: true,
      fromGesture: false,
    });

    await vi.advanceTimersByTimeAsync(RUN_POLL_MAX_MS * 2);
    expect(read).toHaveBeenCalledTimes(1);
  });
});

describe("runFollower: the guards every answer meets", () => {
  it("drops an answer older than one already applied", async () => {
    const { follower, read } = makeFollower();
    const slow = deferred();
    read.mockReturnValueOnce(slow.promise);

    follower.adopt(ACTIVE, 500, follower.claim());
    await vi.advanceTimersByTimeAsync(500);
    expect(read).toHaveBeenCalledTimes(1);

    // A Stop's answer leaves after the poll and lands first.
    const stopped = { ...ACTIVE, cancel_requested: true };
    follower.adopt(stopped, 500, follower.claim({ fromGesture: true }));
    slow.resolve({ run: ACTIVE, retryAfterMs: 500 });
    await vi.advanceTimersByTimeAsync(0);

    expect(follower.run.value).toEqual(stopped);
  });

  it("does not count a run found already ended as one that ended", () => {
    const { follower, onAdopt } = makeFollower();
    follower.adopt(DONE, null, follower.claim());
    expect(onAdopt).toHaveBeenCalledWith(DONE, {
      previous: null,
      ended: false,
      fromGesture: false,
    });
  });

  it("counts a gesture's answer that has already ended as the end", () => {
    const { follower, onAdopt } = makeFollower();
    follower.adopt(DONE, null, follower.claim({ fromGesture: true }));
    expect(onAdopt.mock.calls[0][1].ended).toBe(true);
  });

  it("writes nothing from an answer that left before a reset", async () => {
    const { follower, read, onAdopt } = makeFollower();
    const slow = deferred();
    read.mockReturnValueOnce(slow.promise);

    follower.adopt(ACTIVE, 500, follower.claim());
    await vi.advanceTimersByTimeAsync(500);
    onAdopt.mockClear();
    follower.forget();
    slow.resolve({ run: DONE, retryAfterMs: null });
    await vi.advanceTimersByTimeAsync(0);

    expect(follower.run.value).toBeNull();
    expect(onAdopt).not.toHaveBeenCalled();
  });
});

// Answers race: a start, a Stop, a poll and a resume each take a ticket as
// they leave, but the server can read the run for one before another that
// left earlier. Which run is newer is the server's word (the order it started
// them in), and so is how far a run has got (going, asked to stop, ended);
// the ticket decides only between answers that agree on both.
describe("runFollower: which answer is the newer word", () => {
  const OLDER = { id: "r0", seq: 0, active: false };
  const NEWER = { id: "r1", seq: 1, active: true };

  it("takes an answer about a newer run, though a look at an older one left after it", () => {
    const { follower, onAdopt } = makeFollower();
    const start = follower.claim({ fromGesture: true });
    // A forced resume leaves after the start and reads the newest run before
    // the server has made the start's.
    const look = follower.claim();
    expect(follower.adopt(OLDER, null, look)).toBe(true);

    expect(follower.adopt(NEWER, 1000, start)).toBe(true);
    expect(follower.run.value).toEqual(NEWER);
    expect(onAdopt).toHaveBeenLastCalledWith(NEWER, {
      previous: OLDER,
      ended: false,
      fromGesture: true,
    });
  });

  it("drops an answer about an older run, though it left after the newer one's", () => {
    const { follower, onAdopt } = makeFollower();
    const start = follower.claim({ fromGesture: true });
    const look = follower.claim();
    expect(follower.adopt(NEWER, 1000, start)).toBe(true);

    expect(follower.adopt(OLDER, null, look)).toBe(false);
    expect(follower.run.value).toEqual(NEWER);
    expect(onAdopt).toHaveBeenCalledTimes(1);
  });

  it("drops a poll of an older run that left after a start of a newer one", async () => {
    const { follower, read } = makeFollower();
    const going = { ...OLDER, active: true };
    follower.adopt(going, 500, follower.claim());
    const start = follower.claim({ fromGesture: true });
    const slow = deferred();
    read.mockReturnValueOnce(slow.promise);
    await vi.advanceTimersByTimeAsync(500);
    expect(read).toHaveBeenCalledWith("r0");

    follower.adopt(NEWER, 1000, start);
    slow.resolve({ run: { ...going, processed: 5 }, retryAfterMs: 500 });
    await vi.advanceTimersByTimeAsync(0);
    expect(follower.run.value).toEqual(NEWER);
  });

  it("leaves two runs the server cannot tell apart to the ticket, and says when a run carries no order", () => {
    const { follower } = makeFollower();
    const first = follower.claim();
    const second = follower.claim();
    const twin = { id: "r2", seq: 1, active: true };
    expect(follower.adopt(twin, 1000, second)).toBe(true);
    expect(follower.adopt(NEWER, 1000, first)).toBe(false);
    expect(console.warn).not.toHaveBeenCalled();

    const unordered = { id: "r3", active: true };
    expect(follower.adopt(unordered, 1000, follower.claim())).toBe(true);
    expect(console.warn).toHaveBeenCalledWith(
      expect.stringContaining("cannot order run r3 against run r2"),
    );
  });

  it("takes a Stop's answer over a read of the same run that left after it", async () => {
    const { follower, read } = makeFollower();
    follower.adopt(ACTIVE, 500, follower.claim());
    const stop = follower.claim({ fromGesture: true });
    const slow = deferred();
    read.mockReturnValueOnce(slow.promise);
    await vi.advanceTimersByTimeAsync(500);

    // The poll left after the Stop, was read before the server heard it, and
    // lands first.
    slow.resolve({ run: { ...ACTIVE, processed: 5 }, retryAfterMs: 500 });
    await vi.advanceTimersByTimeAsync(0);
    const stopping = { ...ACTIVE, cancel_requested: true };
    expect(follower.adopt(stopping, 500, stop)).toBe(true);
    expect(follower.run.value).toEqual(stopping);
  });

  it("orders answers by when they left against the answer it holds, not one it has replaced", () => {
    const { follower } = makeFollower();
    follower.adopt(ACTIVE, 500, follower.claim());
    const stop = follower.claim({ fromGesture: true });
    const sawStop = follower.claim();
    const missedStop = follower.claim();
    // Both reads left after the Stop. The later one was read before the server
    // heard it and lands first; the Stop's answer replaces it; the earlier
    // read, which saw the Stop, left after the Stop and is the newer word.
    expect(follower.adopt({ ...ACTIVE, processed: 3 }, 500, missedStop)).toBe(
      true,
    );
    const stopping = { ...ACTIVE, cancel_requested: true, processed: 4 };
    expect(follower.adopt(stopping, 500, stop)).toBe(true);
    const further = { ...stopping, processed: 7 };
    expect(follower.adopt(further, 500, sawStop)).toBe(true);
    expect(follower.run.value).toEqual(further);
  });

  it("never takes a run back from asked to stop, or from ended", () => {
    const { follower, onAdopt } = makeFollower();
    const stopping = { ...ACTIVE, cancel_requested: true };
    follower.adopt(stopping, 500, follower.claim({ fromGesture: true }));

    expect(follower.adopt(ACTIVE, 500, follower.claim())).toBe(false);
    expect(follower.run.value).toEqual(stopping);
    expect(follower.adopt(DONE, null, follower.claim())).toBe(true);
    expect(follower.adopt(stopping, 500, follower.claim())).toBe(false);
    expect(follower.adopt(ACTIVE, 500, follower.claim())).toBe(false);
    expect(follower.run.value).toEqual(DONE);
    expect(onAdopt.mock.calls.map(([, info]) => info.ended)).toEqual([
      false,
      true,
    ]);
  });
});

describe("runFollower: failed reads", () => {
  it("forgets a run the server no longer has", async () => {
    const { follower, read } = makeFollower();
    read.mockRejectedValueOnce({ response: { status: 404 } });

    follower.adopt(ACTIVE, 500, follower.claim());
    await vi.advanceTimersByTimeAsync(500);

    expect(follower.run.value).toBeNull();
    expect(follower.unreachable.value).toBe(false);
    await vi.advanceTimersByTimeAsync(RUN_POLL_MAX_MS * 2);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("backs off while the run cannot be read, and recovers", async () => {
    const { follower, read } = makeFollower();
    read
      .mockRejectedValueOnce({ response: { status: 502 } })
      .mockRejectedValueOnce({ code: "ERR_NETWORK" })
      .mockResolvedValueOnce({ run: ACTIVE, retryAfterMs: 1000 });

    follower.adopt(ACTIVE, 500, follower.claim());
    await vi.advanceTimersByTimeAsync(500);
    expect(follower.unreachable.value).toBe(true);

    // First failure: twice the default. Second: four times.
    await vi.advanceTimersByTimeAsync(RUN_POLL_MS * 2 - 1);
    expect(read).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(RUN_POLL_MS * 4 - 1);
    expect(read).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(read).toHaveBeenCalledTimes(3);

    expect(follower.unreachable.value).toBe(false);
  });
});

describe("runFollower: resume", () => {
  it("reads the newest run once per session, sharing a read on the wire", async () => {
    const { follower, readLatest } = makeFollower();
    const latest = deferred();
    readLatest.mockReturnValueOnce(latest.promise);

    const first = follower.resume();
    const second = follower.resume();
    expect(second).toBe(first);
    latest.resolve(DONE);
    await expect(first).resolves.toEqual(DONE);

    await follower.resume();
    expect(readLatest).toHaveBeenCalledTimes(1);
  });

  it("reads again when forced, and after a reset", async () => {
    const { follower, readLatest } = makeFollower();
    readLatest.mockResolvedValue(DONE);

    await follower.resume();
    await follower.resume({ force: true });
    expect(readLatest).toHaveBeenCalledTimes(2);

    follower.forget();
    await follower.resume();
    expect(readLatest).toHaveBeenCalledTimes(3);
  });

  it("leaves an active run to its poll", async () => {
    const { follower, readLatest } = makeFollower();
    follower.adopt(ACTIVE, 1000, follower.claim());
    await follower.resume({ force: true });
    expect(readLatest).not.toHaveBeenCalled();
  });

  it("answers null for a read that a reset overtook", async () => {
    const { follower, readLatest } = makeFollower();
    const latest = deferred();
    readLatest.mockReturnValueOnce(latest.promise);

    const pending = follower.resume();
    follower.forget();
    // The next session already follows a run of its own when the old read lands.
    const next = { id: "r2", active: false };
    follower.adopt(next, null, follower.claim());
    latest.resolve(ACTIVE);

    await expect(pending).resolves.toBeNull();
    expect(follower.run.value).toEqual(next);
  });
});
