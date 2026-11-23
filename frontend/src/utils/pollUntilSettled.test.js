import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  abortableSleep,
  isTransientError,
  pollUntilSettled,
} from "./pollUntilSettled";

// What axios rejects with: `isAxiosError`, and a `response` only when the
// server answered.
function networkError() {
  return Object.assign(new Error("Network Error"), {
    isAxiosError: true,
    code: "ERR_NETWORK",
  });
}
function httpError(status) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    isAxiosError: true,
    response: { status, data: {} },
  });
}

const isTerminal = (status) =>
  ["completed", "failed", "cancelled"].includes(status.stage);
const progressOf = (status) => `${status.stage}|${status.processed}`;

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("pollUntilSettled", () => {
  it("keeps following an import that is still moving past 600 answers", async () => {
    let calls = 0;
    const fetchStatus = vi.fn(async () => {
      calls += 1;
      return calls <= 700
        ? { stage: "importing", processed: calls }
        : { stage: "completed", processed: 700 };
    });

    const done = pollUntilSettled({ fetchStatus, isTerminal, progressOf });
    await vi.runAllTimersAsync();
    const result = await done;

    expect(result.state).toBe("settled");
    expect(result.status.stage).toBe("completed");
    expect(fetchStatus).toHaveBeenCalledTimes(701);
  });

  it("asks again after a second while the answer keeps moving", async () => {
    const times = [];
    let calls = 0;
    const fetchStatus = vi.fn(async () => {
      times.push(Date.now());
      calls += 1;
      return calls <= 5
        ? { stage: "importing", processed: calls }
        : { stage: "completed", processed: 5 };
    });

    const done = pollUntilSettled({ fetchStatus, isTerminal, progressOf });
    await vi.runAllTimersAsync();
    await done;

    const gaps = times.slice(1).map((t, i) => t - times[i]);
    expect(gaps).toEqual([1000, 1000, 1000, 1000, 1000]);
  });

  it("slows down while nothing moves, never past 10 s, and never gives up", async () => {
    const times = [];
    const idle = [];
    const fetchStatus = vi.fn(async () => {
      times.push(Date.now());
      return { stage: "importing", processed: 3 };
    });
    const controller = new AbortController();
    let settled = false;

    const done = pollUntilSettled({
      fetchStatus,
      isTerminal,
      progressOf,
      onStatus: (_status, { idleMs }) => idle.push(idleMs),
      signal: controller.signal,
    }).then((result) => {
      settled = true;
      return result;
    });
    // 4.5 minutes of a job that does not move.
    for (let second = 0; second < 270; second++) {
      await vi.advanceTimersByTimeAsync(1000);
    }

    // One answer and 30 unchanged ones after it.
    expect(fetchStatus.mock.calls.length).toBeGreaterThanOrEqual(31);
    const gaps = times.slice(1).map((t, i) => t - times[i]);
    for (let i = 1; i < gaps.length; i++) {
      expect(gaps[i]).toBeGreaterThanOrEqual(gaps[i - 1]);
    }
    // The first answer is news; each unchanged one after it waits longer.
    expect(gaps.slice(0, 3)).toEqual([1000, 1500, 2250]);
    expect(Math.max(...gaps)).toBe(10000);
    expect(gaps.at(-1)).toBe(10000);
    // Still running: no attempt count ends it.
    expect(settled).toBe(false);
    // How long since the last change, for the "still working" message.
    expect(idle[0]).toBe(0);
    expect(idle.at(-1)).toBe(times.at(-1) - times[0]);

    controller.abort();
    await expect(done).resolves.toEqual({ state: "aborted" });
  });

  it("stops at once on abort mid-sleep, leaving no timer behind", async () => {
    const fetchStatus = vi.fn(async () => ({
      stage: "importing",
      processed: 1,
    }));
    const controller = new AbortController();
    let result = null;

    pollUntilSettled({
      fetchStatus,
      isTerminal,
      progressOf,
      signal: controller.signal,
    }).then((r) => {
      result = r;
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    controller.abort();
    // Microtasks only: a sleep that ignored the signal would still be waiting.
    for (let i = 0; i < 10; i++) await Promise.resolve();

    expect(result).toEqual({ state: "aborted" });
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(60000);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });

  it("rides out a server that cannot be reached for a while", async () => {
    const failures = [
      networkError(),
      httpError(503),
      Object.assign(new Error("timeout of 60000ms exceeded"), {
        isAxiosError: true,
        code: "ECONNABORTED",
      }),
      httpError(429),
      httpError(502),
    ];
    const fetchStatus = vi.fn(async () => {
      if (failures.length) throw failures.shift();
      return { stage: "completed", processed: 2 };
    });
    const onTransientError = vi.fn();

    const done = pollUntilSettled({
      fetchStatus,
      isTerminal,
      progressOf,
      onTransientError,
    });
    await vi.runAllTimersAsync();
    const result = await done;

    expect(result.state).toBe("settled");
    expect(result.status.stage).toBe("completed");
    expect(onTransientError).toHaveBeenCalledTimes(5);
    expect(
      onTransientError.mock.calls.map(([, info]) => info.failures),
    ).toEqual([1, 2, 3, 4, 5]);
  });

  it("waits longer after each failure in a row, up to 10 s", async () => {
    const times = [];
    let calls = 0;
    const fetchStatus = vi.fn(async () => {
      times.push(Date.now());
      calls += 1;
      if (calls <= 8) throw networkError();
      return { stage: "completed" };
    });
    // The top of each jitter range, so the bound itself is what is measured.
    vi.spyOn(Math, "random").mockReturnValue(0.999999);

    const done = pollUntilSettled({ fetchStatus, isTerminal, progressOf });
    await vi.runAllTimersAsync();
    await done;
    Math.random.mockRestore();

    const gaps = times.slice(1).map((t, i) => t - times[i]);
    expect(gaps.slice(0, 4)).toEqual([1000, 2000, 4000, 8000]);
    expect(Math.max(...gaps)).toBe(10000);
  });

  it("ends in unknown, not failed, when the server no longer has the session", async () => {
    const fetchStatus = vi.fn(async () => {
      throw httpError(404);
    });

    const done = pollUntilSettled({ fetchStatus, isTerminal, progressOf });
    await vi.runAllTimersAsync();
    const result = await done;

    expect(result.state).toBe("unknown");
    expect(result.error.response.status).toBe(404);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });

  it("lets an error that is not a failed request through", async () => {
    const fetchStatus = vi.fn(async () => {
      throw new TypeError("status.outcomes is not iterable");
    });

    const done = pollUntilSettled({ fetchStatus, isTerminal, progressOf });
    const outcome = expect(done).rejects.toThrow(TypeError);
    await vi.runAllTimersAsync();
    await outcome;
  });
});

describe("isTransientError", () => {
  it("names what a retry can outlast", () => {
    expect(isTransientError(networkError())).toBe(true);
    expect(isTransientError(httpError(500))).toBe(true);
    expect(isTransientError(httpError(503))).toBe(true);
    expect(isTransientError(httpError(408))).toBe(true);
    expect(isTransientError(httpError(429))).toBe(true);
    expect(isTransientError(httpError(404))).toBe(false);
    expect(isTransientError(httpError(413))).toBe(false);
    expect(isTransientError(new TypeError("bug"))).toBe(false);
  });
});

describe("abortableSleep", () => {
  it("resolves true when the time is up", async () => {
    const slept = abortableSleep(500);
    await vi.advanceTimersByTimeAsync(500);
    await expect(slept).resolves.toBe(true);
  });

  it("resolves false at once on abort and clears its timer", async () => {
    const controller = new AbortController();
    const slept = abortableSleep(5000, controller.signal);
    controller.abort();
    await expect(slept).resolves.toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not sleep at all on a signal already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(abortableSleep(5000, controller.signal)).resolves.toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
});
