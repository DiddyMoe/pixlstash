// Follow a server job by polling its status until the job itself says it is
// over.
//
// A counted loop ("give up after 600 polls") ends on the client's patience,
// not on anything the server said, so a long job that was still moving got
// reported as timed out. Here only three things end the wait:
//   - a terminal answer (`isTerminal(status)`)             → { state: "settled", status }
//   - a definitive HTTP answer (404, 401, 403...)           → { state: "unknown", error }
//     The job's fate is not known, which is not the same as failed.
//   - the caller's `signal` (the follower was disposed)    → { state: "aborted" }
// A request that may succeed if asked again (no reply at all, a timeout, 5xx,
// 408, 429) is retried for as long as it takes, with a growing, jittered wait.
// Anything else thrown (a bug in `fetchStatus`) is not a request failure and
// propagates.
//
// Pacing: 1 s after an answer that moved (`progressOf` changed), then x1.5 per
// unchanged answer up to 10 s. Every wait is abortable, so a disposed follower
// leaves no timer behind.

const MIN_INTERVAL_MS = 1000;
const MAX_INTERVAL_MS = 10000;
const INTERVAL_GROWTH = 1.5;

/**
 * Sleep that ends early on `signal`.
 *
 * @param {number} ms
 * @param {AbortSignal} [signal]
 * @returns {Promise<boolean>} true when the time ran out, false on abort.
 */
export function abortableSleep(ms, signal) {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(false);
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      resolve(false);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve(true);
    }, Math.round(ms));
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Whether a failed request may succeed if sent again: no reply at all
 * (network down, timeout), a server error, 408 or 429. A refusal (4xx) is the
 * server's answer and asking again gets the same one.
 *
 * @param {*} error - what an apiClient call rejected with.
 * @returns {boolean}
 */
export function isTransientError(error) {
  if (!error?.isAxiosError) return false;
  const status = error.response?.status;
  if (status === undefined) return true;
  return status >= 500 || status === 408 || status === 429;
}

/**
 * @param {Object} args
 * @param {(signal?: AbortSignal) => Promise<*>} args.fetchStatus
 * @param {(status: *) => boolean} args.isTerminal
 * @param {(status: *) => *} [args.progressOf] A value that changes when the
 *   job moves (compared with `!==`). Default: never moves, so the wait grows.
 * @param {(status: *, info: {idleMs: number}) => void} [args.onStatus]
 *   Each non-terminal answer, with how long it has been since one moved.
 * @param {(error: *, info: {failures: number}) => void} [args.onTransientError]
 *   Each retried failure, with how many have happened in a row.
 * @param {AbortSignal} [args.signal]
 * @returns {Promise<{state: "settled", status: *} | {state: "unknown", error: *} | {state: "aborted"}>}
 */
export async function pollUntilSettled({
  fetchStatus,
  isTerminal,
  progressOf = () => null,
  onStatus,
  onTransientError,
  signal,
}) {
  let interval = MIN_INTERVAL_MS;
  let seen = false;
  let lastProgress;
  let lastMovedAt = Date.now();
  let failures = 0;

  for (;;) {
    if (signal?.aborted) return { state: "aborted" };
    let status;
    try {
      status = await fetchStatus(signal);
    } catch (error) {
      if (signal?.aborted) return { state: "aborted" };
      if (!isTransientError(error)) {
        if (error?.response) return { state: "unknown", error };
        throw error;
      }
      failures += 1;
      onTransientError?.(error, { failures });
      // Full jitter (AWS Builders' Library): spreads the retries of many
      // clients that lost the server at the same moment.
      const ceiling = Math.min(
        MAX_INTERVAL_MS,
        MIN_INTERVAL_MS * 2 ** (failures - 1),
      );
      if (!(await abortableSleep(Math.random() * ceiling, signal))) {
        return { state: "aborted" };
      }
      continue;
    }
    if (signal?.aborted) return { state: "aborted" };
    failures = 0;
    if (isTerminal(status)) return { state: "settled", status };

    const progress = progressOf(status);
    const now = Date.now();
    if (!seen || progress !== lastProgress) {
      seen = true;
      lastProgress = progress;
      lastMovedAt = now;
      interval = MIN_INTERVAL_MS;
    } else {
      interval = Math.min(MAX_INTERVAL_MS, interval * INTERVAL_GROWTH);
    }
    onStatus?.(status, { idleMs: now - lastMovedAt });
    if (!(await abortableSleep(interval, signal))) return { state: "aborted" };
  }
}
