// The Scrapheap's background runs: emptying it, restoring all of it, and
// undoing a restore.
//
// Both used to be one request that did the whole heap before answering, which
// a heap of tens of thousands of pictures outlasts: the request timed out while
// the server carried on. They are now asynchronous request-replies, the same
// shape as the Duplicates auto-stack run (`useDedupStore`): the start answers at
// once with the run, the server works through the heap in slices, and this
// store follows the run by polling at the interval the server names until it
// ends (`utils/runFollower.js`, shared with the auto-stack run). The run does not depend on this tab; a reload, another tab or a later
// visit finds it again through the newest run.
//
// The store follows and announces; the grid acts. When a followed run ends the
// store says how, in whatever view the user is in (the grid may not be mounted
// at all): the outcome notice, the restore's receipt, the locked-set warning,
// and the snapshots that still hold what was destroyed (App.vue mounts that
// dialog). It also publishes `outcome`, and the grid
// (`components/views/ImageGrid.vue`) does the grid-local part the old
// synchronous answer used to trigger: re-read the heap, keep the locked
// pictures selected.

import { defineStore } from "pinia";
import { ref, onScopeDispose } from "vue";

import {
  cancelScrapheapRun,
  getLatestScrapheapRun,
  getScrapheapRun,
  startScrapheapRun,
  undoScrapheapRun,
} from "../api/pictures";
import { getUndoState } from "../api/operations";
import { newOperationBatchId, onSessionReset } from "../utils/apiClient";
import { requestFailure } from "../utils/apiError";
import { parseServerTimestamp } from "../utils/retention";
import {
  buildLockedDeleteMessage,
  raiseLockedDeleteNotice,
} from "../utils/lockedDelete";
import { createRunFollower } from "../utils/runFollower";
import {
  SCRAPHEAP_GESTURES,
  arrivedSentences,
  picturesPhrase,
  scrapheapRunFailureSentence,
  scrapheapRunOutcomeNotice,
} from "../utils/scrapheapRun";
import { useNoticeStore } from "./useNoticeStore";
import { useOperationStore } from "./useOperationStore";
import { useUserPrefsStore } from "./useUserPrefsStore";

/** The operation a restore run's slices record: the only kind a run writes. */
const OP_TYPE_RESTORE = "pictures.scrapheap.restore";

/**
 * The prefix of a batch id the server minted (`SERVER_BATCH_ID_PREFIX` in
 * operation_log_service.py). A client's own are `cli-` (utils/apiClient.js).
 */
const SERVER_BATCH_ID_PREFIX = "srv-";

/** The owner name this store's rule is kept under in the operation store. */
const NARRATION_OWNER = "scrapheap-run";

/**
 * A start or an undo whose answer never arrived is sent again under the same
 * request id, which the server answers with the run the first
 * attempt started, if it did, rather than a second one; a purge's
 * confirmation is not spent by the replay. These are the waits before the
 * second and third attempt, as the Duplicates auto-stack run waits
 * (`useDedupStore`); after them the caller hears that no answer came.
 */
const SUBMIT_RETRY_MS = Object.freeze([1000, 3000]);

/** The requests that carry a request id, and so are safe to send again. */
const RESENDABLE = new Set(["start", "undo"]);

/** The notice key every run's outcome shares, so a newer one replaces it. */
const OUTCOME_NOTICE_KEY = "scrapheap-run";

/**
 * Where this tab keeps the runs it started, so a reload still announces the
 * end of one as its own: `{[runId]: {gesture, scoped}}`, or null for a start
 * given no context. Session storage, because it is per tab and survives a
 * reload (https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage);
 * local storage would let a second tab claim the run. Never the picture ids: a
 * selection can be tens of thousands, and it does not survive a reload anyway.
 */
export const STARTED_RUNS_STORAGE_KEY = "pixlstash.scrapheapRun.started";

/**
 * The runs an earlier page of this tab started, as they were stored.
 *
 * @returns {Map<string, Object|null>}
 */
function readStartedRuns() {
  // A browser that refuses storage (a blocked-storage setting, a sandboxed
  // frame) throws on the read as well as the write: the store still sets up,
  // with no saved runs, as `persistStartedRuns` carries on past a refused write.
  let stored;
  try {
    stored = window.sessionStorage.getItem(STARTED_RUNS_STORAGE_KEY);
  } catch (err) {
    console.warn(
      `[scrapheap] could not read ${STARTED_RUNS_STORAGE_KEY}; ` +
        "a run that ends now is announced as one started elsewhere",
      err,
    );
    return new Map();
  }
  if (!stored) return new Map();
  try {
    const parsed = JSON.parse(stored);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return new Map(Object.entries(parsed));
    }
    console.warn(
      `[scrapheap] ignoring the runs this tab started: ${STARTED_RUNS_STORAGE_KEY} holds no object`,
      stored,
    );
  } catch (err) {
    console.warn(
      `[scrapheap] ignoring the runs this tab started: ${STARTED_RUNS_STORAGE_KEY} is not JSON; ` +
        "a run that ends now is announced as one started elsewhere",
      err,
    );
  }
  return new Map();
}

/** Resolve after `ms`, for a retry's pause. */
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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
  // The run this store follows (the newest one it has heard of, or null), its
  // poll, and the guards every answer meets.
  const follower = createRunFollower({
    logPrefix: "[scrapheap]",
    runName: "run",
    read: getScrapheapRun,
    readLatest: getLatestScrapheapRun,
    // A run's id is a random hex string, so the server's order is its
    // `created_at`: stamped under the registry's lock, and what it reads the
    // newest run by.
    order: (r) => parseServerTimestamp(r.created_at),
    onAdopt: takeRun,
  });
  const { run, active, unreachable } = follower;
  // The request on the wire for the run ("start", "undo" or "cancel"), so the
  // control that sent it can show it pending; null when there is none. A stop
  // may leave while a start or an undo is still out (the bar's Stop is for the
  // run the bar shows), and then this is "cancel" until the stop is answered.
  const action = ref(null);
  // What `action` is made of: the start or the undo out, and whether a stop
  // is. Kept apart because each request clears only its own when it settles.
  let gestureOut = null;
  let stopOut = false;
  /**
   * A followed run that has just been seen to end, for the grid to act on:
   * `{run, context, startedHere}`. `startedHere` is true for a run this tab's
   * own start created, and `context` is what that start was given. A run found
   * already ended is not an outcome; nothing changed under the user.
   */
  const outcome = ref(null);
  // After a purge this tab started, the snapshots that still hold the deleted
  // pictures' metadata (the archives are not scrubbed), so the user can delete
  // them if the deletion was for privacy. App.vue shows the dialog.
  const snapshotsWithDeleted = ref([]);
  const snapshotsWithDeletedOpen = ref(false);

  // Every History step a followed restore run has recorded. Kept after the run
  // ends, because the operation log can be re-read after the last poll.
  const segmentBatches = new Set();
  // The runs this tab started, by id, with the context their start was given.
  // A run started before a reload comes back with only what was stored.
  const startedRuns = readStartedRuns();
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
   * batch id. One it has not named yet is a restore under a server-minted
   * (`srv-`) batch id while a restore run is active: a run's slices are always
   * minted by the server, and a gesture's own restore (the lightbox's) always
   * carries the client's `cli-` id, so it keeps its pill.
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
      operation?.op_type === OP_TYPE_RESTORE &&
      String(batchId ?? "").startsWith(SERVER_BATCH_ID_PREFIX),
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

  /**
   * Store the runs this tab started, as `{gesture, scoped}` each. A tab that
   * cannot store them still announces its runs until it is reloaded, so the
   * failure is logged and the run goes on.
   */
  function persistStartedRuns() {
    const stored = {};
    for (const [id, context] of startedRuns) {
      stored[id] = context
        ? { gesture: context.gesture ?? null, scoped: Boolean(context.scoped) }
        : null;
    }
    try {
      if (startedRuns.size === 0) {
        window.sessionStorage.removeItem(STARTED_RUNS_STORAGE_KEY);
      } else {
        window.sessionStorage.setItem(
          STARTED_RUNS_STORAGE_KEY,
          JSON.stringify(stored),
        );
      }
    } catch (err) {
      console.warn(
        `[scrapheap] could not store the ${startedRuns.size} run(s) this tab started; ` +
          "after a reload their end is announced as one started elsewhere",
        err,
      );
    }
  }

  /**
   * A run was taken (see `createRunFollower`'s `adopt`). A run that ended
   * becomes the `outcome`: on the poll's word, or on a gesture's answer when
   * the run had already ended by the time it arrived (a Stop before the first
   * slice). A run found by {@link resume} already ended is followed but is not
   * an outcome, unless this tab started it before a reload and has not yet
   * announced its end.
   *
   * @param {Object} next
   * @param {{ended: boolean}} how
   */
  function takeRun(next, { ended }) {
    installNarrationFilter();
    for (const batchId of next.segment_batch_ids ?? []) {
      segmentBatches.add(batchId);
    }
    const startedHere = startedRuns.has(next.id);
    if (ended || (!next.active && startedHere)) {
      outcome.value = {
        run: next,
        context: startedRuns.get(next.id) ?? null,
        startedHere,
      };
      if (startedHere) {
        startedRuns.delete(next.id);
        persistStartedRuns();
      }
      void announce(outcome.value);
    }
  }

  /**
   * Say how a followed run ended. A run this tab started gets what its
   * gesture's answer used to bring: the snapshots that still hold what a purge
   * destroyed, the locked-set warning for a selection, the restore's receipt,
   * and otherwise a notice. One started elsewhere gets the notice only. Raised
   * here and not by the grid, which is unmounted in most of the app's views:
   * a notice reporting a background operation must survive navigation
   * (notice-surface.md §6).
   *
   * The purge's notices go up before anything is awaited, so the grid, acting
   * on the same `outcome` after this returns, finds them already raised.
   *
   * @param {{run: Object, context: (Object|null), startedHere: boolean}} ended
   */
  async function announce({ run: ended, context, startedHere }) {
    const noticeStore = useNoticeStore();
    const gesture = startedHere ? (context?.gesture ?? null) : null;
    if (gesture && ended.kind === "purge") {
      offerSnapshotsWithDeleted(ended.snapshots_with_deleted);
      const lockedCount = Number(ended.skipped_locked_count) || 0;
      if (
        gesture === SCRAPHEAP_GESTURES.PURGE_SELECTION &&
        ended.status === "complete" &&
        lockedCount > 0
      ) {
        // The locked-set notice carries the lever (how to unlock), and, like
        // every purge's end, what was binned after the confirmation.
        const message = buildLockedDeleteMessage({
          lockedCount,
          deletedCount: Number(ended.deleted_count) || 0,
          forever: true,
        });
        raiseLockedDeleteNotice(noticeStore, {
          ...message,
          body: [message.body, ...arrivedSentences(ended)].join(" "),
        });
        return;
      }
    } else if (
      gesture &&
      ended.kind === "restore" &&
      ended.status === "complete" &&
      (await raiseRestoreReceipt(ended))
    ) {
      return;
    }
    const { level, message } = scrapheapRunOutcomeNotice(ended, gesture);
    noticeStore[level](message, {
      key: OUTCOME_NOTICE_KEY,
      action: outcomeAction(ended, gesture),
    });
  }

  /**
   * The one action an outcome notice offers, or null.
   *
   * A restore this tab started that left something to undo offers to undo it
   * (the receipt pill, raised only for a one-step restore, covers the rest). An
   * undo that History blocked offers to finish it, which works once the user
   * has undone the change its sentence names; pressed before that, it is
   * blocked again at the same place and says so.
   *
   * @param {Object} ended
   * @param {string|null} gesture - this tab's gesture, null for another's run.
   * @returns {{label: string, handler: Function}|null}
   */
  function outcomeAction(ended, gesture) {
    if (!gesture) return null;
    if (ended.kind === "restore" && ended.can_undo) {
      return {
        label: "Undo restore",
        handler: () => void undoFromNotice(ended.id),
      };
    }
    if (
      ended.kind === "undo" &&
      ended.status === "blocked" &&
      ended.error_code === "history_moved_on" &&
      ended.target_run_id
    ) {
      return {
        label: "Finish undo",
        handler: () => void undoFromNotice(ended.target_run_id),
      };
    }
    return null;
  }

  /**
   * An outcome notice's undo. The notice is gone by the time the answer
   * comes, and the grid that words a start's refusal may not be mounted, so a
   * refusal is said here.
   *
   * @param {string} restoreRunId
   */
  async function undoFromNotice(restoreRunId) {
    const result = await undoRestore(restoreRunId);
    if (result.ok) return;
    const message = scrapheapRunFailureSentence(
      "undo",
      result.error,
      SCRAPHEAP_GESTURES.UNDO_RESTORE,
    );
    if (!message) return;
    const level = result.error?.run?.active ? "warning" : "error";
    useNoticeStore()[level](message, { key: OUTCOME_NOTICE_KEY });
  }

  function offerSnapshotsWithDeleted(snapshots) {
    if (useUserPrefsStore().hidePurgeSnapshotWarning) return;
    if (!Array.isArray(snapshots) || !snapshots.length) return;
    snapshotsWithDeleted.value = snapshots;
    snapshotsWithDeletedOpen.value = true;
  }

  /**
   * Send a start, an undo or a stop.
   *
   * A start and an undo carry the gesture's request id (`send` closes over
   * it), so one whose answer never came is sent again as is, up to twice
   * ({@link SUBMIT_RETRY_MS}): the server answers the same id with the run the
   * first attempt started, so the retry's answer is this gesture's, whichever
   * attempt reached it. A refusal is an answer and is not retried, and a stop
   * is not either: the user's next Stop is the retry, and it is harmless.
   *
   * A stop waits only for another stop. One sent while a start or an undo is
   * out, or waiting to be sent again, names the run the bar shows
   * ({@link cancel}) and takes its own ticket. Every answer, a refusal's run
   * included, goes through the follower's one guard, which keeps the newer
   * word whichever answer came back first: a run the server started later, or
   * the same run further on. So a start's answer read before the stop does
   * not show the run going again, and one naming a newer run than a look that
   * left after it is still taken.
   *
   * @param {string} name - `start`, `undo` or `cancel`.
   * @param {Function} send - `() => Promise<{run, retryAfterMs}>`.
   * @param {Object|null} [context] - for a start or an undo, kept with the run
   *   it starts.
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   */
  async function submit(name, send, context = null) {
    const stop = name === "cancel";
    if (stop ? stopOut : action.value) return localRefusal("busy");
    if (stop) stopOut = true;
    else gestureOut = name;
    action.value = stopOut ? "cancel" : gestureOut;
    const from = follower.epoch();
    try {
      for (let attempt = 0; ; attempt += 1) {
        const claim = follower.claim({ from, fromGesture: true });
        try {
          const reply = await send();
          if (from !== follower.epoch()) return localRefusal("cancelled");
          const started = RESENDABLE.has(name) && reply.run?.id != null;
          if (started) {
            startedRuns.set(reply.run.id, context);
            persistStartedRuns();
          }
          const taken = follower.adopt(reply.run, reply.retryAfterMs, claim);
          const held = run.value;
          if (started && !taken && held?.id === reply.run.id && !held.active) {
            // A newer word (a Stop's answer, a poll's) took this run's end
            // before this answer said the run is this gesture's: announce
            // that end again as this tab's own, so its entry is not left
            // behind for a reload to announce.
            console.warn(
              `[scrapheap] the run ${name}'s answer came after run ` +
                `${held.id} had ended; announcing that end as this tab's`,
            );
            follower.adopt(
              held,
              null,
              follower.claim({ from, fromGesture: true }),
            );
          }
          return { ok: true, run: reply.run };
        } catch (err) {
          const failure = runFailure(err);
          if (from !== follower.epoch()) return { ok: false, error: failure };
          const unanswered =
            failure.kind === "timeout" || failure.kind === "network";
          if (
            unanswered &&
            RESENDABLE.has(name) &&
            attempt < SUBMIT_RETRY_MS.length
          ) {
            console.warn(
              `[scrapheap] no answer to the run ${name}; sending the same ` +
                `request again (attempt ${attempt + 2})`,
              err,
            );
            await wait(SUBMIT_RETRY_MS[attempt]);
            if (from !== follower.epoch()) return { ok: false, error: failure };
            continue;
          }
          console.warn(
            `[scrapheap] the run ${name} did not go through ` +
              `(${failure.kind}${failure.status ? ` ${failure.status}` : ""}` +
              `${failure.code ? `, ${failure.code}` : ""})`,
            err,
          );
          if (failure.run) {
            // The refusal names a run (the one already going, or the restore
            // an undo was refused for): follow it, so the bar shows what the
            // refusal was about, unless the bar already holds a newer word
            // (a Stop's answer, a newer run). Not as the gesture's answer: its
            // end is not this gesture's outcome.
            follower.adopt(failure.run, null, {
              epoch: from,
              ticket: claim.ticket,
            });
          }
          return { ok: false, error: failure };
        }
      }
    } finally {
      // After a session reset `forget` has already cleared both.
      if (from === follower.epoch()) {
        if (stop) stopOut = false;
        else gestureOut = null;
        action.value = stopOut ? "cancel" : gestureOut;
      }
    }
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
   *   when the server is not running background work. `error.kind` is
   *   `timeout` or `network` when no attempt was answered.
   */
  function start({
    kind,
    pictureIds,
    includeProtected = false,
    confirmToken,
    context = null,
  } = {}) {
    const requestId = newOperationBatchId();
    return submit(
      "start",
      () =>
        startScrapheapRun({
          kind,
          pictureIds,
          includeProtected,
          confirmToken,
          requestId,
        }),
      context,
    );
  }

  /**
   * Undo a restore run: put what it restored back in the Scrapheap, as a run
   * of its own that this tab follows and announces like a start.
   *
   * @param {string} restoreRunId
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   *   `error.code` as the route refuses (`nothing_to_undo`,
   *   `not_a_restore_run`, `scrapheap_run_active`, ...). A refusal that names
   *   a run has it followed.
   */
  function undoRestore(restoreRunId) {
    const requestId = newOperationBatchId();
    return submit(
      "undo",
      () => undoScrapheapRun(restoreRunId, { requestId }),
      { gesture: SCRAPHEAP_GESTURES.UNDO_RESTORE, scoped: false },
    );
  }

  /**
   * Ask the followed run to stop. It stops at its next slice; what it already
   * did stays done.
   *
   * Always the run the bar shows, even while a start or an undo is still out:
   * with none showing, or one already ended, nothing is sent.
   *
   * @returns {Promise<{ok: true, run: Object}|{ok: false, error: Object}>}
   *   `error.kind` is `busy` only while an earlier stop is still out.
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
    return follower.resume({ force });
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
    const from = follower.epoch();
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
    if (from !== follower.epoch() || step?.batch_id !== segments[0]) {
      return false;
    }
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
    follower.forget();
    gestureOut = null;
    stopOut = false;
    action.value = null;
    outcome.value = null;
    snapshotsWithDeleted.value = [];
    snapshotsWithDeletedOpen.value = false;
    segmentBatches.clear();
    startedRuns.clear();
    persistStartedRuns();
  }

  onScopeDispose(follower.stopPoll);
  const unsubscribeSessionReset = onSessionReset(forget);
  onScopeDispose(() => unsubscribeSessionReset());

  return {
    run,
    active,
    unreachable,
    action,
    outcome,
    snapshotsWithDeleted,
    snapshotsWithDeletedOpen,
    start,
    undoRestore,
    cancel,
    resume,
    acknowledgeOutcome,
    raiseRestoreReceipt,
    forget,
  };
});
