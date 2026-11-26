<script setup>
import { nextTick, onBeforeUnmount, ref, watch } from "vue";
import { API_BASE_URL, isReadOnly } from "../../utils/apiClient";
import { useTasksStore } from "../../stores/useTasksStore";
import { useNoticeStore } from "../../stores/useNoticeStore";
import { useImportArrivalsStore } from "../../stores/useImportArrivalsStore";
import { useLibrariesStore } from "../../stores/useLibrariesStore";
import {
  cancelStaging,
  commitStaging,
  fetchStagingStatus,
  openStagingSession,
  stageFiles,
} from "../../api/pictureImport";
import { errorDetail } from "../../utils/apiError";
import { importRelativePath } from "../../utils/media";
import {
  claimCommitRecords,
  dropCommitRecord,
  saveCommitRecord,
} from "../../utils/importCommitRecord";
import {
  abortableSleep,
  isTransientError,
  pollUntilSettled,
} from "../../utils/pollUntilSettled";

// ── Async streaming-staging import (#459) ────────────────────────────────────
// This component owns the upload and the commit of the two-phase import over
// the streaming-staging session contract. It renders nothing itself: what the
// user sees is the grid's arriving tiles (useImportArrivalsStore →
// ArrivingTiles, #2512) and, for anyone who navigates away, the Tasks-tab row,
// both live from the drop.
//   Phase A "staging / unsafe":  open a staging session, then STREAM the files
//     into it batch by batch, each with its arriving tile's key. A beforeunload
//     guard is armed (leaving now aborts the upload) and the band says so; its
//     Cancel (and the Tasks row's) discards the not-yet-committed session.
//     Each batch carries an Idempotency-Key, the same on every resend, so a
//     batch whose reply was lost is answered from the first copy instead of
//     being staged twice. Only what a resend can outlast is sent again (no
//     reply, a timeout, a 5xx, the server still taking the same batch); a
//     refusal is final, and the staged files are discarded at once.
//   Safe transition:  once every byte is staged, commit hands off to a
//     background PictureImportTask; the guard drops and the tiles turn from
//     "uploading" to "checking". A commit whose reply is lost is not cancelled:
//     the session's status says whether it started.
//   Phase B "importing / safe":  followed BY staging id by the tasks store
//     (useTasksStore.followImportRun), not by this component: the grid that
//     mounts it unmounts on every view switch, and the store also picks the
//     import back up after a reload. Each file's outcome (new / duplicate /
//     scrapheaped / failed) marks its tile as it is decided. The grid
//     refreshes off the backend's CHANGED_PICTURES / PICTURE_IMPORTED WS
//     broadcast (no results payload).
//   Scrapheap matches:  a staged file whose content matches a SOFT-DELETED
//     picture is reported in its own bucket. It is not imported again (that
//     would put a second copy of every scrapheaped picture back on disk); the
//     backend restores the picture instead, as one undoable operation.
// ALL import backend calls are isolated behind ../../api/pictureImport.js.
// The public API (startImport + the four emits) is unchanged so existing call
// sites keep working. Drop-target association (project / set / character) is
// passed through startImport options into openStagingSession and applied
// server-side on commit. There is no per-file results[] array - the grid
// refreshes off the WS broadcast, so import-finished carries counts but an
// empty results list. Media, .zip archives and .txt sidecars all stream.

const props = defineProps({
  backendUrl: { type: String, default: () => API_BASE_URL },
});

const emit = defineEmits([
  "import-started",
  "import-finished",
  "import-cancelled",
  "import-error",
]);

const tasksStore = useTasksStore();
const noticeStore = useNoticeStore();
const arrivals = useImportArrivalsStore();
const librariesStore = useLibrariesStore();

// Whole-flow gate (Phase A + Phase B) used for re-entrancy.
const importActive = ref(false);

const importProgress = ref(0);
const importTotal = ref(0);
const uploadBytesUploaded = ref(0);
const uploadBytesTotal = ref(0);
const importPhase = ref(""); // uploading | processing | done | duplicates | unknown | cancelled | error
const cancelImport = ref(false);
const currentImportController = ref(null);
// Aborted by a cancel, so a wait between two attempts ends at once.
let cancelController = null;

// Unique id for the current import: its Tasks-tab row and its arriving group.
let importRunId = null;
// The current staging session id (Phase A/B backend handle), or null when idle.
let currentStagingId = null;
// True from the commit request on: the bytes are handed over, cancel is over.
let committing = false;

// ── beforeunload guard (the hard backstop for the unsafe upload window) ───────
let guardArmed = false;
function beforeUnloadGuard(e) {
  // Standard pattern: preventDefault + assign returnValue makes the browser show
  // its native "leave site?" prompt. We only arm this WHILE uploading (Phase A).
  e.preventDefault();
  e.returnValue = "";
  return "";
}
function armGuard() {
  if (guardArmed || typeof window === "undefined") return;
  window.addEventListener("beforeunload", beforeUnloadGuard);
  guardArmed = true;
}
function disarmGuard() {
  if (!guardArmed || typeof window === "undefined") return;
  window.removeEventListener("beforeunload", beforeUnloadGuard);
  guardArmed = false;
}

const TERMINAL_IMPORT_PHASES = new Set([
  "done",
  "duplicates",
  "unknown",
  "cancelled",
  "error",
]);

// The longest wait between two sends of one batch, or two commits.
const MAX_RESEND_WAIT_MS = 8000;
// How many times one batch, or one commit, is sent.
const MAX_RETRIES = 3;

/** The code of a refusal the server explained (`detail: {code, message}`). */
function refusalCode(err) {
  return err?.response?.data?.detail?.code ?? null;
}

/**
 * The sentence for a failed import request: the server's reason first, then
 * the transport's message, then `fallback`. The import routes refuse with
 * `detail: {code, message}`, whose sentence errorDetail does not read.
 */
function refusalMessage(err, fallback) {
  const message = err?.response?.data?.detail?.message;
  return (
    errorDetail(err) ||
    (typeof message === "string" ? message.trim() : "") ||
    err?.message ||
    fallback
  );
}

/**
 * Whether a batch the server did not take may be taken if sent again: no
 * reply, a timeout, a server error, or the server still receiving the same
 * batch. A refusal (4xx) is the answer to these files, and 507 is the disk,
 * which a resend does not free.
 */
function canResendBatch(err) {
  if (refusalCode(err) === "batch_in_progress") return true;
  return isTransientError(err) && err?.response?.status !== 507;
}

/** A fresh key for one batch; every resend of that batch carries it. */
function newBatchKey() {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`
  );
}

/** Full jitter: clients that lost the server together do not return together. */
function resendWait(attempt) {
  return Math.random() * Math.min(MAX_RESEND_WAIT_MS, 1000 * 2 ** attempt);
}

function finalizeCancelled() {
  disarmGuard();
  importPhase.value = "cancelled";
  importActive.value = false;
  cancelImport.value = false;
  currentImportController.value = null;
  if (importRunId) {
    tasksStore.clearImportRun(importRunId);
    tasksStore.unregisterImportAbort(importRunId);
    arrivals.dismiss(importRunId);
  }
  // Discard the not-yet-committed staging session (best-effort) so its streamed
  // files are cleaned up server-side.
  if (currentStagingId) {
    const stagingId = currentStagingId;
    currentStagingId = null;
    cancelStaging({ backendUrl: props.backendUrl, stagingId }).catch((err) => {
      console.warn("Failed to cancel staging session", stagingId, err);
    });
  }
  emit("import-cancelled");
}

function finalizeError(message) {
  disarmGuard();
  importPhase.value = "error";
  importActive.value = false;
  if (importRunId) {
    // The arriving band holds the reason until it is dismissed; the task row
    // is the record for anyone who has navigated away from the grid. The store
    // takes the row out later, whether or not this importer is still mounted.
    arrivals.fail(importRunId, message);
    tasksStore.finishImportRun(
      importRunId,
      {
        status: "failed",
        percent: 0,
        current: importProgress.value,
        total: importTotal.value,
        message: `Import failed: ${message}`,
        label: "Import failed",
      },
      8000,
    );
  }
  cancelImport.value = false;
  currentImportController.value = null;
  emit("import-error", { message });
}

/**
 * Discard the staging session of an import that cannot reach its commit, so
 * its files go now instead of when the server reaps the session.
 *
 * @returns {Promise<boolean>} true when the server answers that the import
 *   has started after all: the session is then followed, not failed.
 */
async function abandonStaging() {
  const stagingId = currentStagingId;
  if (!stagingId) return false;
  try {
    await cancelStaging({ backendUrl: props.backendUrl, stagingId });
  } catch (err) {
    if (refusalCode(err) === "import_committed") return true;
    console.warn(
      "Could not discard the staging session; the server removes it when it expires",
      stagingId,
      err,
    );
  }
  currentStagingId = null;
  return false;
}

/** End an import before its commit: discard what was staged, say why. */
async function failBeforeCommit(message) {
  if (await abandonStaging()) {
    await followCommitted(currentStagingId);
    return;
  }
  finalizeError(message);
}

/**
 * Hand the staged files to the server's import.
 *
 * A refusal (4xx, 507) means nothing started. Any other failure (no reply, a
 * timeout, a 5xx, a commit already in progress) leaves open whether the
 * import began, so the session's status decides: still `staging` means it
 * never started, and it is committed again; any later stage means it did,
 * and it is followed. A commit whose reply was lost is never cancelled.
 * Asking the status and waiting to commit again both end on `signal`.
 *
 * @param {string} stagingId
 * @param {number} maxAttempts
 * @param {AbortSignal} signal - the tasks store's, aborted by stopPolling().
 * @returns {Promise<{committed: true, stagedCount?: number}
 *   | {committed: false, error: *} | {committed: false, aborted: true}>}
 */
async function commitImport(stagingId, maxAttempts, signal) {
  for (let attempt = 1; ; attempt++) {
    let failure;
    try {
      const commit = await commitStaging({
        backendUrl: props.backendUrl,
        stagingId,
      });
      return { committed: true, stagedCount: commit.stagedCount };
    } catch (err) {
      if (!err?.isAxiosError) throw err;
      const status = err.response?.status;
      if (status === 507 || (status !== 409 && !isTransientError(err))) {
        return { committed: false, error: err };
      }
      failure = err;
      console.warn(
        `[IMPORT] Commit of ${stagingId} unanswered (attempt ${attempt}); asking the session`,
        err,
      );
    }
    const answer = await pollUntilSettled({
      fetchStatus: (fetchSignal) =>
        fetchStagingStatus({
          backendUrl: props.backendUrl,
          stagingId,
          signal: fetchSignal,
        }),
      isTerminal: (status) => status.stage !== "committing",
      signal,
    });
    if (answer.state === "aborted") return { committed: false, aborted: true };
    // Gone (404) or past staging: the follower says how it ends.
    if (answer.state !== "settled" || answer.status.stage !== "staging") {
      return { committed: true };
    }
    if (attempt >= maxAttempts) return { committed: false, error: failure };
    if (!(await abortableSleep(resendWait(attempt), signal))) {
      return { committed: false, aborted: true };
    }
  }
}

/**
 * The tasks store stopped polling (sign-out, teardown) before the commit was
 * known to have started: nothing follows this import, so its row and tiles go.
 */
function endUnfollowedImport(stagingId) {
  disarmGuard();
  importActive.value = false;
  currentStagingId = null;
  currentImportController.value = null;
  if (importRunId) {
    tasksStore.unregisterImportAbort(importRunId);
    tasksStore.clearImportRun(importRunId);
    arrivals.dismiss(importRunId);
  }
  logImportTrace("Polling stopped before the commit was answered", {
    stagingId,
  });
}

/**
 * Leave a committed import to the tasks store, which follows it to its end
 * even after this importer unmounts, and report how it ended.
 */
async function followCommitted(stagingId, stagedCount) {
  // The session is committed; it can no longer be cancelled from the client.
  committing = true;
  disarmGuard();
  currentStagingId = null;
  const runId = importRunId;
  tasksStore.unregisterImportAbort(runId);
  arrivals.markChecking(runId);
  importTotal.value = stagedCount || importTotal.value;
  importProgress.value = 0;
  // The safe transition: the bytes are on the server and leaving is harmless.
  importPhase.value = "processing";

  const result = await tasksStore.followImportRun({
    runId,
    stagingId,
    backendUrl: props.backendUrl,
    total: importTotal.value,
  });
  currentImportController.value = null;
  cancelImport.value = false;
  importActive.value = false;

  if (result.state === "completed") {
    // There is no per-file results[] - the grid reconciles the new pictures
    // off the backend's CHANGED_PICTURES / PICTURE_IMPORTED WebSocket
    // broadcast, which the grid already consumes.
    const { status } = result;
    const importedCount = status.importedCount ?? 0;
    importPhase.value = importedCount === 0 ? "duplicates" : "done";
    importTotal.value = Math.max(importTotal.value, status.total || 0);
    importProgress.value = importTotal.value;
    emit("import-finished", {
      importedCount,
      total: importTotal.value,
      phase: importPhase.value,
      results: [],
    });
    logImportTrace("Import finished", {
      importedCount,
      duplicateCount: status.duplicateCount ?? 0,
      // The third bucket: content that matched a picture in the Scrapheap,
      // restored instead of imported again.
      scrapheapedCount: status.scrapheapedCount ?? 0,
      scrapheapedPictureIds: status.scrapheapedPictureIds ?? [],
      phase: importPhase.value,
    });
  } else if (result.state === "unknown") {
    // The server no longer knows the import (restarted, expired): what it
    // imported is kept, so the grid still refreshes, and nothing is a failure.
    importPhase.value = "unknown";
    emit("import-finished", {
      importedCount: null,
      total: importTotal.value,
      phase: importPhase.value,
      results: [],
    });
    logImportTrace("Import status unknown", { stagingId });
  } else if (result.state === "failed") {
    importPhase.value = "error";
    const message = result.status.error || "Import failed";
    emit("import-error", { message });
    logImportTrace("Import failed", { message });
  } else if (result.state === "cancelled") {
    // Committed imports don't cancel client-side; an external cancel is a
    // terminal, non-error exit.
    importPhase.value = "cancelled";
    emit("import-cancelled");
  }
  // "aborted": the store stopped following (sign-out, teardown). It has kept
  // the run for a reload, and there is nothing to report.
}

// The single client-side abort path, shared by the arriving band's Cancel and
// the Tasks-tab cancel affordance (via tasksStore.abortImportRun → this handler,
// registered in startImport). It stops the in-flight upload by aborting the
// active request controller, and any wait before a resend; the batch loop's
// cancel checks then unwind cleanly through finalizeCancelled (no error
// toast). This can only stop the PRE-COMMIT upload - once committed, the
// import is a background server task the client cannot stop (see the
// abortable notes in startImport / useTasksStore).
function handleCancelImport() {
  // Once the commit is sent there is nothing a client abort can stop.
  if (!importActive.value || committing) return;
  cancelImport.value = true;
  cancelController?.abort();
  if (currentImportController.value) {
    try {
      currentImportController.value.abort();
    } catch (err) {
      console.warn("Failed to abort current import", err);
    }
  }
}

function logImportTrace(message, details = null) {
  if (details !== null && details !== undefined) {
    console.info(`[IMPORT TRACE] ${message}`, details);
    return;
  }
  console.info(`[IMPORT TRACE] ${message}`);
}

async function startImport(files, options = {}) {
  if (!files || !files.length) return;
  if (importActive.value) {
    // Recover from a stale terminal state left by the previous run.
    if (TERMINAL_IMPORT_PHASES.has(importPhase.value)) {
      importActive.value = false;
    } else {
      // Said, not swallowed: this importer runs one import at a time, and a
      // server-side import can take minutes after the upload is done.
      console.info(
        "Import request ignored because another import is in progress.",
      );
      noticeStore.warning(
        "Another import is still running here. Drop these again when it finishes.",
        { key: "import-busy" },
      );
      return;
    }
  }

  cancelImport.value = false;
  cancelController = new AbortController();
  currentStagingId = null;
  committing = false;
  // Drop any lingering handler from a previous run before minting a new id.
  if (importRunId) tasksStore.unregisterImportAbort(importRunId);
  importRunId = `import-${crypto?.randomUUID?.() ?? Date.now().toString(36)}`;
  const totalBytes = files.reduce((sum, f) => sum + (f.size || 0), 0);
  // The acknowledgement: the arriving tiles appear on the drop itself, from
  // the files the browser already holds, before anything is sent. Keys are
  // streamed beside each file so the server can name every outcome.
  const fileKeys = arrivals.begin(importRunId, files, {
    bytesTotal: totalBytes,
  });

  if (isReadOnly.value) {
    importActive.value = true;
    finalizeError("Importing is not available with a read-only token.");
    return;
  }

  // Make the (otherwise dead) import-abort subsystem live: register the real
  // client-side abort so tasksStore.abortImportRun(runId) actually stops the
  // in-flight upload. Unregistered
  // on every terminal path (finalizeCancelled / finalizeError / the commit)
  // and on unmount.
  tasksStore.registerImportAbort(importRunId, () => handleCancelImport());
  importActive.value = true;
  importProgress.value = 0;
  importTotal.value = files.length;
  uploadBytesUploaded.value = 0;
  uploadBytesTotal.value = totalBytes;
  importPhase.value = "uploading";
  currentImportController.value = null;
  // Arm the unsafe-window guard: leaving/refreshing now aborts the upload.
  armGuard();
  // The Tasks row from the drop, not from the commit: it is the record for
  // anyone who leaves the grid, and while bytes are in flight it can cancel.
  // Reported on every whole percent, so a large batch moves the row while it
  // uploads rather than jumping at its end.
  let reportedPercent = -1;
  const reportUpload = () => {
    const percent = uploadBytesTotal.value
      ? Math.min(100, (uploadBytesUploaded.value / uploadBytesTotal.value) * 100)
      : 0;
    if (Math.floor(percent) === reportedPercent) return;
    reportedPercent = Math.floor(percent);
    tasksStore.setImportRun(importRunId, {
      status: "running",
      percent,
      current: 0,
      total: importTotal.value,
      message: "Uploading - keep this tab open",
      label: "Uploading pictures",
      abortable: true,
    });
  };
  reportUpload();
  emit("import-started", {
    fileCount: files.length,
    projectId: options.projectId ?? null,
  });

  logImportTrace("Import started", {
    fileCount: files.length,
    totalUploadBytes: uploadBytesTotal.value,
    projectId: options.projectId ?? null,
  });

  const BATCH_SIZE = 100;
  const MIN_TIMEOUT_MS = 60000;
  const TIMEOUT_PER_FILE_MS = 4000;
  const TIMEOUT_PER_MB_MS = 100;
  const NO_PROGRESS_ABORT_MS = 15000;
  const overrideTimeout =
    typeof options.timeoutMs === "number" && options.timeoutMs > 0
      ? options.timeoutMs
      : null;

  let uploadedBytesAccum = 0;

  try {
    // ── Phase A: open a staging session, then STREAM every batch into it. The
    // bytes are unsafe until commit - the beforeunload guard is armed. ──
    const session = await openStagingSession({
      backendUrl: props.backendUrl,
      projectId: options.projectId ?? null,
      setId: options.setId ?? null,
      characterId: options.characterId ?? null,
      totalFiles: files.length,
    });
    if (!session.stagingId) {
      finalizeError("Could not open an import staging session.");
      return;
    }
    currentStagingId = session.stagingId;
    // The library the session belongs to, as the server names it: the commit
    // record is kept under it, so a reload resumes the import in that library
    // whether or not the libraries registry had loaded when it was sent.
    const sessionLibraryUuid = session.libraryUuid;

    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      if (cancelImport.value) {
        finalizeCancelled();
        return;
      }

      const batch = files.slice(i, i + BATCH_SIZE);
      const batchKeys = fileKeys.slice(i, i + BATCH_SIZE);
      // Where each file sat in the drop, so a caption tags only the picture in
      // its own folder. Part of the batch the key names: every resend sends
      // the same paths.
      const batchRelpaths = batch.map(importRelativePath);
      const batchBytes = batch.reduce((sum, f) => sum + (f.size || 0), 0);
      const batchTimeoutMs =
        overrideTimeout ??
        Math.max(
          MIN_TIMEOUT_MS,
          batch.length * TIMEOUT_PER_FILE_MS,
          Math.ceil(batchBytes / (1024 * 1024)) * TIMEOUT_PER_MB_MS,
        );
      const batchIndex = Math.floor(i / BATCH_SIZE) + 1;
      // One key per batch, sent on every attempt: a resend after a lost reply
      // is answered from the first copy instead of staging the files twice.
      const batchKey = newBatchKey();

      let ok = false;
      let lastMessage = null;

      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        if (cancelImport.value) {
          finalizeCancelled();
          return;
        }

        const controller = new AbortController();
        currentImportController.value = controller;
        let firstProgressLogged = false;
        let lastProgressAt = Date.now();
        let noProgressAbortFired = false;
        const noProgressTimer = setInterval(() => {
          if (cancelImport.value) return;
          const elapsed = Date.now() - lastProgressAt;
          if (elapsed < NO_PROGRESS_ABORT_MS) return;
          noProgressAbortFired = true;
          console.warn(
            `[IMPORT] Aborting batch ${batchIndex} attempt ${attempt} due to no upload progress for ${elapsed}ms.`,
          );
          try {
            controller.abort("no-upload-progress");
          } catch {
            controller.abort();
          }
        }, 1000);
        const timeout = setTimeout(() => {
          console.warn(
            `[IMPORT] Aborting batch ${batchIndex} after ${batchTimeoutMs}ms timeout (attempt ${attempt}).`,
          );
          controller.abort();
        }, batchTimeoutMs);
        try {
          const stagedBatch = await stageFiles({
            backendUrl: props.backendUrl,
            stagingId: currentStagingId,
            files: batch,
            keys: batchKeys,
            relpaths: batchRelpaths,
            idempotencyKey: batchKey,
            signal: controller.signal,
            timeoutMs: batchTimeoutMs,
            onUploadProgress: (progressEvent) => {
              const loaded = progressEvent.loaded ?? 0;
              lastProgressAt = Date.now();
              if (!firstProgressLogged) firstProgressLogged = true;
              uploadBytesUploaded.value = Math.min(
                uploadBytesTotal.value,
                uploadedBytesAccum + loaded,
              );
              arrivals.setBytesSent(importRunId, uploadBytesUploaded.value);
              reportUpload();
              // Every byte is out: the wait now is the server storing the
              // batch, which the batch timeout bounds. Not a stall.
              if (progressEvent.total > 0 && loaded >= progressEvent.total) {
                clearInterval(noProgressTimer);
              }
            },
          });
          arrivals.markSkipped(
            importRunId,
            batchKeys,
            batch,
            stagedBatch.skipped,
          );
          clearInterval(noProgressTimer);
          clearTimeout(timeout);
          if (controller === currentImportController.value) {
            currentImportController.value = null;
          }
          ok = true;
          break;
        } catch (err) {
          clearInterval(noProgressTimer);
          clearTimeout(timeout);
          if (controller === currentImportController.value) {
            currentImportController.value = null;
          }
          const isAbort =
            err?.name === "AbortError" || err?.code === "ERR_CANCELED";
          const isTimeout = err?.code === "ECONNABORTED";
          if (isAbort && cancelImport.value) {
            finalizeCancelled();
            return;
          }
          let resend;
          if (isAbort || isTimeout) {
            resend = true;
            lastMessage = noProgressAbortFired
              ? "Upload stalled (no progress)"
              : "Upload timed out";
            console.warn(
              `[IMPORT] Batch ${batchIndex} aborted (attempt ${attempt}). reason=${
                noProgressAbortFired
                  ? "no-progress"
                  : isTimeout
                    ? "timeout"
                    : "abort"
              }`,
            );
          } else {
            resend = canResendBatch(err);
            lastMessage = refusalMessage(err, "Upload failed.");
            console.warn(
              `[IMPORT] Batch ${batchIndex} failed (attempt ${attempt}${
                resend ? "" : ", not resent"
              }):`,
              err,
            );
          }
          if (!resend) break;
          if (
            attempt < MAX_RETRIES &&
            !(await abortableSleep(
              resendWait(attempt),
              cancelController.signal,
            ))
          ) {
            finalizeCancelled();
            return;
          }
        }
      }

      if (!ok) {
        await failBeforeCommit(lastMessage ?? "Upload failed.");
        return;
      }

      uploadedBytesAccum += batchBytes;
      uploadBytesUploaded.value = uploadedBytesAccum;
      arrivals.setBytesSent(importRunId, uploadedBytesAccum);
      reportUpload();
      await nextTick();
    }

    if (cancelImport.value) {
      finalizeCancelled();
      return;
    }

    // All bytes are staged - commit the SAFE HANDOFF to the background import.
    // From here Cancel has nothing to stop, so neither the band nor the Tasks
    // row offers it. The leave guard stays armed until the commit is known to
    // have started: closing the tab mid-commit would strand the staged session.
    uploadBytesUploaded.value = uploadBytesTotal.value;
    committing = true;
    arrivals.markChecking(importRunId);
    tasksStore.setImportRun(importRunId, {
      status: "running",
      percent: 0,
      current: 0,
      total: importTotal.value,
      message: "Handing over to the server…",
      label: "Importing pictures",
    });

    const stagingId = currentStagingId;
    // Kept until the commit is answered, so a reload while it is unanswered
    // picks the import up again (resumeInterruptedCommits). From the answer
    // on, the tasks store keeps the import for a reload.
    const libraryUuid = sessionLibraryUuid;
    if (libraryUuid) {
      saveCommitRecord(libraryUuid, {
        runId: importRunId,
        stagingId,
        total: importTotal.value,
      });
    } else {
      console.warn(
        "[IMPORT] The staging session named no library; a reload before the commit is answered loses this import",
        stagingId,
      );
    }
    // The store's signal, not this importer's: a view switch unmounts the
    // importer and the commit is still settled and handed to the store, but
    // signing out stops it.
    const stopSignal = tasksStore.importWorkSignal();
    let commit;
    try {
      commit = await commitImport(stagingId, MAX_RETRIES, stopSignal);
    } finally {
      if (libraryUuid) dropCommitRecord(libraryUuid, stagingId);
    }
    if (commit.aborted || stopSignal.aborted) {
      endUnfollowedImport(stagingId);
      return;
    }
    if (!commit.committed) {
      // Refused before anything started (400 empty / face worker down, 404,
      // 507 disk): the server put the session back to staging. Discard it.
      await failBeforeCommit(
        refusalMessage(commit.error, "Import commit failed."),
      );
      return;
    }
    await followCommitted(stagingId, commit.stagedCount);
  } catch (error) {
    // The server's reason first: opening the session onto a locked set answers
    // 409 with why, and axios's own message says only the status code.
    const message = refusalMessage(error, String(error));
    if (committing) finalizeError(message);
    else await failBeforeCommit(message);
    logImportTrace("Import failed", { message });
  }
}

/**
 * Settle a commit that a reload of the tab interrupted: all of its
 * files were staged, so the import goes ahead as dropped. The commit is sent
 * again, which the server answers with the import it already started, if it
 * did; a commit still being taken is waited on. Then the tasks store follows
 * the import, as it follows any other.
 */
async function resumeCommit(libraryUuid, { runId, stagingId, total }) {
  logImportTrace("Picking up a commit a reload interrupted", { stagingId });
  tasksStore.setImportRun(runId, {
    status: "running",
    percent: 0,
    current: 0,
    total,
    message: "Handing over to the server…",
    label: "Importing pictures",
  });
  const stopSignal = tasksStore.importWorkSignal();
  let commit;
  try {
    commit = await commitImport(stagingId, MAX_RETRIES, stopSignal);
  } finally {
    dropCommitRecord(libraryUuid, stagingId);
  }
  if (commit.aborted || stopSignal.aborted) {
    tasksStore.clearImportRun(runId);
    logImportTrace("Polling stopped before the resumed commit was answered", {
      stagingId,
    });
    return;
  }
  if (commit.committed) {
    tasksStore
      .followImportRun({
        runId,
        stagingId,
        backendUrl: props.backendUrl,
        total,
      })
      .catch((error) =>
        console.error(
          "[IMPORT] Could not follow the import picked up after a reload",
          stagingId,
          error,
        ),
      );
    return;
  }
  tasksStore.clearImportRun(runId);
  console.warn(
    "[IMPORT] The commit a reload interrupted was refused",
    stagingId,
    commit.error,
  );
  if (commit.error?.response?.status === 404) {
    // The server no longer has the session (restarted, or it expired).
    noticeStore.warning(
      "The import a reload interrupted is no longer on the server. Drop those pictures again.",
      { key: "import-resume-failed" },
    );
    return;
  }
  noticeStore.warning(
    `The import a reload interrupted could not start: ${refusalMessage(commit.error, "the server refused it.")}`,
    { key: "import-resume-failed" },
  );
  cancelStaging({ backendUrl: props.backendUrl, stagingId }).catch((err) => {
    console.warn(
      "Could not discard the staging session; the server removes it when it expires",
      stagingId,
      err,
    );
  });
}

/** Pick up the commits of `libraryUuid` a reload interrupted. */
function resumeInterruptedCommits(libraryUuid) {
  if (!libraryUuid || isReadOnly.value) return;
  for (const record of claimCommitRecords(libraryUuid)) {
    resumeCommit(libraryUuid, record).catch((error) =>
      console.error(
        "[IMPORT] Could not pick up the commit a reload interrupted",
        record.stagingId,
        error,
      ),
    );
  }
}

// The library is known once the registry has loaded, after a reload as on a
// sign-in; its interrupted commits are picked up then.
watch(
  () => librariesStore.activeLibrary?.uuid ?? "",
  resumeInterruptedCommits,
  { immediate: true },
);

onBeforeUnmount(() => {
  disarmGuard();
  if (importRunId) tasksStore.unregisterImportAbort(importRunId);
});

defineExpose({ startImport });

// Nothing of its own to draw: the grid's ArrivingTiles and the Tasks-tab row
// show the import (#2512).
defineOptions({ render: () => null });
</script>
