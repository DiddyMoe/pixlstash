// The Scrapheap's background runs: emptying it, and restoring all of it.
//
// Both used to be one request that did the whole heap before answering, which
// a heap of tens of thousands of pictures outlasts: the request timed out while
// the server carried on. They are now asynchronous request-replies, the same
// shape as the Duplicates auto-stack run (`useDedupStore`): the start answers at
// once with the run, the server works through the heap in slices, and this
// store follows the run by polling at the interval the server names until it
// ends. The run does not depend on this tab; a reload, another tab or a later
// visit finds it again through the newest run.
//
// The store follows; the grid acts. When a followed run ends the store
// publishes `outcome`, and the grid (`components/views/ImageGrid.vue`) does what
// the old synchronous answer used to trigger: re-read the heap, keep the locked
// pictures selected, offer the snapshots that still hold what was destroyed.

import { defineStore } from "pinia";
import { ref, computed, onScopeDispose } from "vue";

import {
  cancelScrapheapRun,
  getLatestScrapheapRun,
  getScrapheapRun,
  startScrapheapRun,
} from "../api/pictures";
import { getUndoState } from "../api/operations";
import { onSessionReset } from "../utils/apiClient";
import { requestFailure } from "../utils/apiError";
import { picturesPhrase } from "../utils/scrapheapRun";
import { useOperationStore } from "./useOperationStore";

/**
 * The poll interval when an active run's answer names none, and the bounds a
 * named one is held to. The server's `Retry-After` is the interval; a value
 * outside these bounds is read as a mistake rather than as a reason to spin or
 * to fall silent for minutes. Failed reads back off from the default up to the
 * ceiling.
 */
export const SCRAPHEAP_RUN_POLL_MS = 2000;
export const SCRAPHEAP_RUN_POLL_MIN_MS = 500;
export const SCRAPHEAP_RUN_POLL_MAX_MS = 30000;

/** The operation a restore run's slices record: the only kind a run writes. */
const OP_TYPE_RESTORE = "pictures.scrapheap.restore";

/** The owner name this store's rule is kept under in the operation store. */
const NARRATION_OWNER = "scrapheap-run";

/**
 * Why a run request failed, in the terms the grid words it.
 *
 * {@link requestFailure}'s classification, plus the run a refusal is about: a
 * second start is refused with `{code: "scrapheap_run_active", message, run}`,
 * and `run` is the run already going.
 *
 * @param {*} err - the rejection from a `../api/pictures` run call.
 * @returns {{kind: string, status: (number|null), detail: string,
 *   code: (string|null), run: (Object|null)}}
 */
function runFailure(err) {
  const served = err?.response?.data?.detail;
  const run =
    served &&
    typeof served === "object" &&
    served.run &&
    typeof served.run === "object"
      ? served.run
      : null;
  return { ...requestFailure(err), run };
}

export const useScrapheapRunStore = defineStore("scrapheapRun", () => {
  /** The run this store follows: the newest one it has heard of, or null. */
  const run = ref(null);
  const active = computed(() => Boolean(run.value?.active));
  // True while the run cannot be read. The poll keeps trying, more slowly, and
  // the bar says the progress it shows is the last known.
  const unreachable = ref(false);
  // The request on the wire for the run ("start" or "cancel"), so the control
  // that sent it can show it pending; null when there is none.
  const action = ref(null);
  /**
   * A followed run that has just been seen to end, for the grid to act on:
   * `{run, context, startedHere}`. `startedHere` is true for a run this tab's
   * own start created, and `context` is what that start was given. A run found
   * already ended is not an outcome; nothing changed under the user.
   */
  const outcome = ref(null);

  let pollTimer = null;
  let pollFailures = 0;
  // Bumped when the store forgets the run (a session reset): an answer still
  // on the wire from before then writes nothing.
  let epoch = 0;
  // Every read or write of the run takes a ticket as it leaves, and an answer
  // older than the newest one applied is dropped: a poll sent before a Stop
  // must not land after it and show the run as never asked to stop.
  let ticket = 0;
  let appliedTicket = 0;
  let resumeAttempted = false;
  let resumeInFlight = null;
  // Every History step a followed restore run has recorded. Kept after the run
  // ends, because the operation log can be re-read after the last poll.
  const segmentBatches = new Set();
  // The runs this tab started, by id, with the context their start was given.
  const startedRuns = new Map();
  let narrationFilterInstalled = false;

  /** A refusal decided here, without a request, shaped like a served one. */
  function localRefusal(kind, code = null) {
    return {
      ok: false,
      error: { kind, status: null, detail: "", code, run: null },
    };
  }

  /**
   * The receipt filter: true for an operation a restore run recorded.
   *
   * A restore run records its slices as History steps under the origin of the
   * tab that started it, so without this that tab narrates each slice as its
   * own gesture: a pill every few seconds, each offering to undo 500 arbitrary
   * pictures. The run raises one receipt when it ends instead
   * ({@link raiseRestoreReceipt}). A step a poll has named is known by its
   * batch id. One it has not named yet is any restore while a restore run is
   * active, which also keeps a single restore sent from the lightbox in that
   * window off the pill (it stays in History).
   *
   * @param {Object} operation - an operation-log row.
   * @returns {boolean}
   */
  function holdsBackNarration(operation) {
    const batchId = operation?.batch_id;
    if (batchId && segmentBatches.has(batchId)) return true;
    const followed = run.value;
    return Boolean(
      followed?.active &&
      followed.kind === "restore" &&
      operation?.op_type === OP_TYPE_RESTORE,
    );
  }

  /**
   * Hand the receipt filter to the operation store, once, when this store
   * first follows a run. Lazily, so a session that never runs one leaves the
   * operation store exactly as it was.
   */
  function installNarrationFilter() {
    if (narrationFilterInstalled) return;
    useOperationStore().setNarrationFilter(NARRATION_OWNER, holdsBackNarration);
    narrationFilterInstalled = true;
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
   */
  function schedulePoll(delayMs) {
    stopPoll();
    const hinted = Number.isFinite(delayMs) ? delayMs : SCRAPHEAP_RUN_POLL_MS;
    const delay = Math.min(
      SCRAPHEAP_RUN_POLL_MAX_MS,
      Math.max(SCRAPHEAP_RUN_POLL_MIN_MS, hinted),
    );
    pollTimer = setTimeout(() => {
      pollTimer = null;
      poll();
    }, delay);
  }

  async function poll() {
    const followed = run.value;
    const claim = { epoch, ticket: ++ticket };
    try {
      const reply = await getScrapheapRun(followed.id);
      adopt(reply.run, reply.retryAfterMs, claim);
    } catch (err) {
      // A read that left before a newer answer landed (a Stop, a reset) says
      // nothing about the run as it is now.
      if (claim.epoch !== epoch || claim.ticket < appliedTicket) return;
      if (run.value?.id !== followed.id) return;
      if (Number(err?.response?.status) === 404) {
        // The run is gone: the server restarted (runs live in its memory), or
        // another library is open. There is nothing left to follow.
        console.warn(
          `[scrapheap] run ${followed.id} no longer exists; no longer following it`,
          err,
        );
        appliedTicket = claim.ticket;
        stopPoll();
        run.value = null;
        unreachable.value = false;
        pollFailures = 0;
        return;
      }
      pollFailures += 1;
      unreachable.value = true;
      const backoff = Math.min(
        SCRAPHEAP_RUN_POLL_MAX_MS,
        SCRAPHEAP_RUN_POLL_MS * 2 ** pollFailures,
      );
      console.warn(
        `[scrapheap] could not read run ${followed.id} ` +
          `(${pollFailures} in a row); trying again in ${backoff} ms`,
        err,
      );
      schedulePoll(backoff);
    }
  }

  /**
   * Take a run the server answered with as the one this store follows.
   *
   * The single way a run enters the store, so every answer meets the same
   * guards: one from before a session reset, or older than an answer already
   * applied, is dropped. A run that ended becomes the `outcome`: on the poll's
   * word, or on a gesture's answer when the run had already ended by the time
   * it arrived (a Stop before the first slice). A run found by {@link resume}
   * already ended is followed but is not an outcome.
   *
   * @param {Object} next
   * @param {number|null} retryAfter - the server's poll hint, in ms.
   * @param {Object} claim - taken when the request left.
   * @param {number} claim.epoch
   * @param {number} claim.ticket
   * @param {boolean} [claim.fromGesture=false] - the answer to this tab's own
   *   start or stop.
   * @returns {boolean} whether the run was taken.
   */
  function adopt(next, retryAfter, { epoch: from, ticket: sent, fromGesture = false }) {
    if (!next || from !== epoch || sent < appliedTicket) return false;
    appliedTicket = sent;
    installNarrationFilter();
    for (const batchId of next.segment_batch_ids ?? []) {
      segmentBatches.add(batchId);
    }
    const previous = run.value;
    const sameRun = previous?.id === next.id;
    run.value = next;
    unreachable.value = false;
    pollFailures = 0;
    if (next.active) schedulePoll(retryAfter);
    else stopPoll();
    const ended = !next.active && (sameRun ? previous.active : fromGesture);
    if (ended) {
      outcome.value = {
        run: next,
        context: startedRuns.get(next.id) ?? null,
        startedHere: startedRuns.has(next.id),
      };
      startedRuns.delete(next.id);
    }
    return true;
  }

  /**
   * Send a start or a stop. Not retried: a start has no request id the server
   * could recognise a second time, and a purge's confirmation is spent on the
   * first attempt, so a second send would be refused anyway.
   *
   * @param {string} name - `start` or `cancel`.
   * @param {Function} send - `() => Promise<{run, retryAfterMs}>`.
   * @param {Object|null} [context] - for a start, kept with the run it starts.
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   */
  async function submit(name, send, context = null) {
    if (action.value) return localRefusal("busy");
    action.value = name;
    const from = epoch;
    const claim = { epoch: from, ticket: ++ticket, fromGesture: true };
    try {
      const reply = await send();
      if (from !== epoch) return localRefusal("cancelled");
      if (name === "start" && reply.run?.id != null) {
        startedRuns.set(reply.run.id, context);
      }
      adopt(reply.run, reply.retryAfterMs, claim);
      return { ok: true, run: reply.run };
    } catch (err) {
      const failure = runFailure(err);
      if (from !== epoch) return { ok: false, error: failure };
      console.warn(
        `[scrapheap] the run ${name} did not go through ` +
          `(${failure.kind}${failure.status ? ` ${failure.status}` : ""}` +
          `${failure.code ? `, ${failure.code}` : ""})`,
        err,
      );
      if (failure.run) {
        // The refusal names the run already going: follow it, so the bar shows
        // what the refusal was about. Not as the gesture's answer: that run is
        // somebody else's, and its end is not this gesture's outcome.
        adopt(failure.run, null, { epoch: from, ticket: claim.ticket });
      } else if (
        name === "start" &&
        (failure.kind === "timeout" || failure.kind === "network")
      ) {
        // No answer is not "nothing happened": the start may have reached the
        // server and queued a run. Look, so a run it did start shows its
        // progress. The caller still hears that the start went unanswered.
        const found = await readLatest(from);
        if (found?.active) return { ok: false, error: { ...failure, run: found } };
      }
      return { ok: false, error: failure };
    } finally {
      if (from === epoch) action.value = null;
    }
  }

  /**
   * Read the newest run and follow it. Logs and answers null on failure: the
   * caller is either an unanswered start, which reports its own failure, or a
   * resume, which has nothing to show without it.
   *
   * @param {number} from - the epoch the caller started in.
   * @returns {Promise<Object|null>} the run now followed, or null.
   */
  async function readLatest(from) {
    const claim = { epoch: from, ticket: ++ticket };
    try {
      const latest = await getLatestScrapheapRun();
      if (latest) adopt(latest, null, claim);
    } catch (err) {
      console.warn("[scrapheap] could not look for a run to follow", err);
    }
    return from === epoch ? run.value : null;
  }

  /**
   * Start emptying the Scrapheap, or restoring it, as a background run.
   *
   * @param {Object} options
   * @param {"purge"|"restore"} options.kind
   * @param {Array<number|string>} [options.pictureIds] - the selection; the
   *   whole heap when omitted.
   * @param {boolean} [options.includeProtected=false] - a purge only: also
   *   destroy reference-folder originals.
   * @param {string} [options.confirmToken] - a purge only: the delete
   *   preview's single-use confirmation for exactly this selection.
   * @param {Object} [options.context] - anything the grid needs back when the
   *   run ends (which gesture started it, what the confirm showed). Returned as
   *   `outcome.context`.
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   *   `error.code` is `scrapheap_run_active` when another run is going (that
   *   run is now followed, and is `error.run`) and `scrapheap_runs_unavailable`
   *   when the server is not running background work. `error.run` is also set
   *   when an unanswered start turned out to have queued a run.
   */
  function start({
    kind,
    pictureIds,
    includeProtected = false,
    confirmToken,
    context = null,
  } = {}) {
    return submit(
      "start",
      () =>
        startScrapheapRun({ kind, pictureIds, includeProtected, confirmToken }),
      context,
    );
  }

  /**
   * Ask the followed run to stop. It stops at its next slice; what it already
   * did stays done.
   *
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   */
  function cancel() {
    const followed = run.value;
    if (!followed?.active) return Promise.resolve({ ok: true, run: followed });
    return submit("cancel", () => cancelScrapheapRun(followed.id));
  }

  /**
   * Find the newest run and follow it, so a run started before a reload, in
   * another tab or on an earlier visit shows its progress here, and a restore
   * run's History steps stay off this tab's receipt pill.
   *
   * Once per session unless forced. The app asks on every sidebar refresh,
   * which is how a reload picks a run up wherever it lands; the Scrapheap view
   * forces a fresh read when it opens, since another tab may have started one
   * since. A run already being followed while active is left to its poll.
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
   * The grid has acted on an outcome. Only that run's: an outcome that has
   * since been replaced by a newer one is left for its own acknowledgement.
   *
   * @param {string} runId
   */
  function acknowledgeOutcome(runId) {
    if (outcome.value?.run?.id === runId) outcome.value = null;
  }

  /**
   * Raise one receipt for a restore run this tab started, in place of the
   * per-slice ones {@link holdsBackNarration} kept off the pill.
   *
   * Only when the run restored something, recorded exactly one History step,
   * and that step is still the newest undoable one: the pill's Undo undoes the
   * newest step, so a receipt for a run with several steps, or one another
   * action has since buried, would offer an Undo that reverses something else.
   * A purge restores nothing and records no step, so it never gets one.
   *
   * @param {Object} ended - the run, as its last poll read it.
   * @returns {Promise<boolean>} whether a receipt was raised; when not, the
   *   caller says what the run did some other way.
   */
  async function raiseRestoreReceipt(ended) {
    const segments = ended?.segment_batch_ids ?? [];
    const restored = Number(ended?.restored_count) || 0;
    if (segments.length !== 1 || restored < 1) return false;
    const from = epoch;
    let step;
    try {
      step = (await getUndoState())?.next_undo ?? null;
    } catch (err) {
      console.warn(
        `[scrapheap] could not read the undo state after run ${ended.id}; ` +
          "no receipt for it",
        err,
      );
      return false;
    }
    if (from !== epoch || step?.batch_id !== segments[0]) return false;
    const operationStore = useOperationStore();
    const entry = operationStore.buildReceipt(step, "did");
    // The step's own row names one slice of the run; the receipt names the run.
    operationStore.showReceipt({
      ...entry,
      summary: `Restored ${picturesPhrase(restored)} from the Scrapheap`,
      targetCount: restored,
      mergedCount: 0,
    });
    return true;
  }

  /**
   * Stop following any run and drop everything learned about it, so nothing
   * read under one session reaches the next. The run itself carries on, on
   * the server.
   */
  function forget() {
    epoch += 1;
    stopPoll();
    appliedTicket = ticket;
    run.value = null;
    unreachable.value = false;
    action.value = null;
    outcome.value = null;
    pollFailures = 0;
    resumeAttempted = false;
    resumeInFlight = null;
    segmentBatches.clear();
    startedRuns.clear();
  }

  onScopeDispose(stopPoll);
  const unsubscribeSessionReset = onSessionReset(forget);
  onScopeDispose(() => unsubscribeSessionReset());

  return {
    run,
    active,
    unreachable,
    action,
    outcome,
    start,
    cancel,
    resume,
    acknowledgeOutcome,
    raiseRestoreReceipt,
    forget,
  };
});
