// Following a background run on the server by polling it.
//
// The Scrapheap's runs (`stores/useScrapheapRunStore.js`) and the Duplicates
// auto-stack runs (`stores/useDedupStore.js`) are both asynchronous
// request-replies: a start answers at once with the run, and the client reads
// the run again at the interval the server names until it ends. This is the
// part of that the two share: the poll, its bounds and backoff, the guards
// every answer meets, the once-per-session resume and forgetting on a session
// reset. Each store keeps its own requests, and decides through `onAdopt`
// what a run's arrival or end means to it.
//
// Not `pollUntilSettled.js`: that is one awaited loop, and a run here also
// enters out of band, as the answer to a start or a stop the user sent.

import { computed, ref } from "vue";

/**
 * The poll interval when an active run's answer names none, and the bounds a
 * named one is held to. The server's `Retry-After` is the interval; a value
 * outside these bounds is read as a mistake rather than as a reason to spin or
 * to fall silent for minutes. Failed reads back off from the default up to the
 * ceiling.
 */
export const RUN_POLL_MS = 2000;
export const RUN_POLL_MIN_MS = 500;
export const RUN_POLL_MAX_MS = 30000;

/**
 * The first read of a run the user just started. `Retry-After` is whole
 * seconds (RFC 9110 §10.2.3), so the server cannot ask for less than 1 s
 * without asking for 0, and a run of one picture has usually ended long before
 * then. The first read goes early; every later one keeps the server's interval.
 */
export const RUN_FIRST_POLL_MS = 250;

/**
 * How far a run has got: going (0), asked to stop (1), ended (2). The server
 * only moves a run forward through these (`cancel_requested` is never cleared,
 * and an ended run does not start again), so an answer that shows a run at an
 * earlier stage than the one held was read before it.
 *
 * @param {Object} run
 * @returns {number}
 */
function stageOf(run) {
  if (!run.active) return 2;
  return run.cancel_requested ? 1 : 0;
}

/**
 * Create the state and the poll that follow one kind of run. Call it inside a
 * store's setup; the store stops the poll when its scope is disposed.
 *
 * @param {Object} options
 * @param {string} options.logPrefix - the console tag, e.g. `"[scrapheap]"`.
 * @param {string} options.runName - the run in a log line, e.g. `"run"`.
 * @param {Function} options.read - `(id) => Promise<{run, retryAfterMs}>`.
 * @param {Function} options.readLatest - `() => Promise<Object|null>`, the
 *   newest run.
 * @param {Function} options.order - `(run) => number`, where the run stands in
 *   the order the server started its runs in: a later run has a larger number.
 *   Two runs that answer the same number are told apart by ticket.
 * @param {Function} [options.onAdopt] - `(run, {previous, ended,
 *   fromGesture})`, called once a run has been taken. `ended` is true for a
 *   run seen to end: on the poll's word, or on a gesture's answer when the run
 *   had already ended by the time it arrived. A run found already ended by a
 *   resume is taken but has not `ended`.
 * @returns {Object} `run` and `unreachable` (refs), `active` (computed),
 *   `epoch()`, `claim()`, `adopt()`, `schedulePoll()`, `stopPoll()`,
 *   `readLatest()`, `resume()` and `forget()`.
 */
export function createRunFollower({
  logPrefix,
  runName,
  read,
  readLatest: fetchLatest,
  order,
  onAdopt = () => {},
}) {
  /** The run followed: the newest one heard of, or null. */
  const run = ref(null);
  const active = computed(() => Boolean(run.value?.active));
  // True while the run cannot be read. The poll keeps trying, more slowly, and
  // the UI says the progress it shows is the last known.
  const unreachable = ref(false);

  let pollTimer = null;
  let pollFailures = 0;
  // Bumped when the run is forgotten (a session reset): an answer still on the
  // wire from before then writes nothing.
  let epoch = 0;
  // Every read or write of the run takes a ticket as it leaves. The server can
  // read the run for one request before another that left earlier, so the
  // ticket decides only between answers that agree on which run and how far
  // it has got ({@link isNewerWord}). `appliedTicket` is the ticket of the
  // answer held (or of the 404 or reset that cleared it): an answer about a
  // newer run can carry an older ticket than a look at an older one, and
  // later answers about the newer run are ordered against it, not the look.
  let ticket = 0;
  let appliedTicket = 0;
  let resumeAttempted = false;
  let resumeInFlight = null;

  const aRunName = /^[aeiou]/i.test(runName) ? `an ${runName}` : `a ${runName}`;

  /** The current epoch, for a caller to tell whether a reset came between. */
  function currentEpoch() {
    return epoch;
  }

  /**
   * Take a ticket for a request about to leave.
   *
   * @param {Object} [options]
   * @param {number} [options.from] - the epoch the caller started in; the
   *   current one when omitted.
   * @param {boolean} [options.fromGesture=false] - the request is the user's
   *   own start, stop or undo.
   * @returns {{epoch: number, ticket: number, fromGesture: boolean}}
   */
  function claim({ from = epoch, fromGesture = false } = {}) {
    return { epoch: from, ticket: ++ticket, fromGesture };
  }

  function stopPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  /**
   * Read the run again after `delayMs`, held to the poll bounds. One timer at
   * most: scheduling replaces any read already waiting. Only an active run is
   * scheduled, and every path that ends or forgets one stops the timer.
   *
   * @param {number|null} delayMs - the server's hint, or null for the default.
   * @param {Object} [options]
   * @param {boolean} [options.first=false] - the first read of a run the user
   *   just started: no later than {@link RUN_FIRST_POLL_MS}.
   */
  function schedulePoll(delayMs, { first = false } = {}) {
    stopPoll();
    const hinted = Number.isFinite(delayMs) ? delayMs : RUN_POLL_MS;
    const bounded = Math.min(
      RUN_POLL_MAX_MS,
      Math.max(RUN_POLL_MIN_MS, hinted),
    );
    const delay = first ? Math.min(RUN_FIRST_POLL_MS, bounded) : bounded;
    const scheduledIn = epoch;
    pollTimer = setTimeout(() => {
      pollTimer = null;
      if (scheduledIn === epoch) poll();
    }, delay);
  }

  async function poll() {
    const followed = run.value;
    if (!followed?.active) return;
    const sent = claim();
    try {
      const reply = await read(followed.id);
      adopt(reply.run, reply.retryAfterMs, sent);
    } catch (err) {
      // A read that left before the answer held (a Stop's, say) or before a
      // reset says nothing about the run as it is now.
      if (sent.epoch !== epoch || sent.ticket < appliedTicket) return;
      if (run.value?.id !== followed.id) return;
      if (Number(err?.response?.status) === 404) {
        // The run is gone: another library is open, or the server no longer
        // keeps it. There is nothing left to follow.
        console.warn(
          `${logPrefix} ${runName} ${followed.id} no longer exists; no longer following it`,
          err,
        );
        appliedTicket = sent.ticket;
        stopPoll();
        run.value = null;
        unreachable.value = false;
        pollFailures = 0;
        return;
      }
      pollFailures += 1;
      unreachable.value = true;
      const backoff = Math.min(
        RUN_POLL_MAX_MS,
        RUN_POLL_MS * 2 ** pollFailures,
      );
      console.warn(
        `${logPrefix} could not read ${runName} ${followed.id} ` +
          `(${pollFailures} in a row); trying again in ${backoff} ms`,
        err,
      );
      schedulePoll(backoff);
    }
  }

  /**
   * How much later the server started `next` than `held`: 1, -1, or 0 when it
   * cannot tell them apart.
   *
   * @param {Object} held
   * @param {Object} next
   * @returns {number}
   */
  function runStep(held, next) {
    const heldPlace = order(held);
    const nextPlace = order(next);
    if (!Number.isFinite(heldPlace) || !Number.isFinite(nextPlace)) {
      console.warn(
        `${logPrefix} cannot order ${runName} ${next.id} against ` +
          `${runName} ${held.id} (${nextPlace} against ${heldPlace}); ` +
          "the answer that left last is taken",
      );
      return 0;
    }
    return Math.sign(nextPlace - heldPlace);
  }

  /**
   * Whether an answer is a newer word than the run held. The server's order
   * comes first: a run it started later is newer whichever answer left first,
   * and of the same run, an answer that shows it further on (asked to stop,
   * ended) is newer than one read before. Answers that agree on both are told
   * apart by the order they left in.
   *
   * @param {Object} next - the run answered.
   * @param {number} t - the answer's ticket.
   * @returns {boolean}
   */
  function isNewerWord(next, t) {
    const held = run.value;
    if (held) {
      const step =
        held.id === next.id
          ? Math.sign(stageOf(next) - stageOf(held))
          : runStep(held, next);
      if (step !== 0) return step > 0;
    }
    return t >= appliedTicket;
  }

  /**
   * Take a run the server answered with as the one followed.
   *
   * The single way a run enters, so every answer meets the same guards: one
   * from before a session reset is dropped, and so is one that is not a newer
   * word than the run held ({@link isNewerWord}).
   *
   * @param {Object} next
   * @param {number|null} retryAfter - the server's poll hint, in ms.
   * @param {Object} sent - the {@link claim} taken when the request left.
   * @returns {boolean} whether the run was taken.
   */
  function adopt(
    next,
    retryAfter,
    { epoch: from, ticket: t, fromGesture = false },
  ) {
    if (!next || from !== epoch || !isNewerWord(next, t)) return false;
    appliedTicket = t;
    const previous = run.value;
    const sameRun = previous?.id === next.id;
    run.value = next;
    unreachable.value = false;
    pollFailures = 0;
    // A run the user's own start just began is read early once; a Stop's
    // answer, a poll's or a resume's keeps the server's interval.
    if (next.active) {
      schedulePoll(retryAfter, { first: fromGesture && !sameRun });
    } else {
      stopPoll();
    }
    const ended = !next.active && (sameRun ? previous.active : fromGesture);
    onAdopt(next, { previous, ended, fromGesture });
    return true;
  }

  /**
   * Read the newest run and follow it. Logs and answers null on failure: the
   * caller is either an unanswered start, which reports its own failure, or a
   * resume, which has nothing to show without it.
   *
   * @param {number} [from] - the epoch the caller started in.
   * @returns {Promise<Object|null>} the run now followed, or null.
   */
  async function readLatest(from = epoch) {
    const sent = claim({ from });
    try {
      const latest = await fetchLatest();
      if (latest) adopt(latest, null, sent);
    } catch (err) {
      console.warn(
        `${logPrefix} could not look for ${aRunName} to follow`,
        err,
      );
    }
    return from === epoch ? run.value : null;
  }

  /**
   * Find the newest run and follow it, so a run started before a reload, in
   * another tab or on an earlier visit shows here.
   *
   * Once per session unless forced. A run already being followed while active
   * is left to its poll, and a read already on the wire is shared.
   *
   * @param {Object} [options]
   * @param {boolean} [options.force=false]
   * @returns {Promise<Object|null>} the run now followed, or null.
   */
  function resume({ force = false } = {}) {
    if (resumeInFlight) return resumeInFlight;
    if (active.value || (!force && resumeAttempted)) {
      return Promise.resolve(run.value);
    }
    resumeAttempted = true;
    const pending = readLatest(epoch).finally(() => {
      if (resumeInFlight === pending) resumeInFlight = null;
    });
    resumeInFlight = pending;
    return pending;
  }

  /**
   * Stop following and drop what was learned, so nothing read under one
   * session reaches the next. The run itself carries on, on the server.
   */
  function forget() {
    epoch += 1;
    stopPoll();
    appliedTicket = ticket;
    run.value = null;
    unreachable.value = false;
    pollFailures = 0;
    resumeAttempted = false;
    resumeInFlight = null;
  }

  return {
    run,
    active,
    unreachable,
    epoch: currentEpoch,
    claim,
    adopt,
    schedulePoll,
    stopPoll,
    readLatest,
    resume,
    forget,
  };
}
