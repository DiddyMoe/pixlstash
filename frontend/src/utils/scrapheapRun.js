/**
 * The words for the Scrapheap's background runs (`stores/useScrapheapRunStore`):
 * how a run this tab started ended, and why a request about a run was refused.
 *
 * The grid (`components/views/ImageGrid.vue`) shows them as notices; while a
 * run goes, the bar above the grid (`widgets/EmptyScrapHeap.vue`) carries its
 * progress. Kept out of the grid so that each sentence can be read, and
 * tested, on its own.
 */

/**
 * The gestures that start a run, as the grid hands them to the store's
 * `start` in `context.gesture` and reads them back from `outcome.context`.
 */
export const SCRAPHEAP_GESTURES = Object.freeze({
  /** The bar's Empty scrapheap, or the sidebar's: the whole heap, destroyed. */
  EMPTY: "empty",
  /** Delete forever on a selection, or on the lightbox's picture. */
  PURGE_SELECTION: "purge-selection",
  /** The bar's Restore all. */
  RESTORE_ALL: "restore-all",
  /** Restore on a selection. */
  RESTORE_SELECTION: "restore-selection",
});

/** What a start that failed was trying to do, as the old notices said it. */
const COULD_NOT = {
  [SCRAPHEAP_GESTURES.EMPTY]: "Couldn't empty the scrapheap.",
  [SCRAPHEAP_GESTURES.PURGE_SELECTION]: "Couldn't delete those pictures.",
  [SCRAPHEAP_GESTURES.RESTORE_ALL]: "Couldn't restore the scrapheap.",
  [SCRAPHEAP_GESTURES.RESTORE_SELECTION]: "Couldn't restore those pictures.",
};

/** A count with grouped digits, never NaN or negative. */
function figure(value) {
  return Math.max(0, Number(value) || 0).toLocaleString();
}

/** A count of pictures in words: "1 picture", "1,200 pictures". */
export function picturesPhrase(value) {
  const count = Math.max(0, Number(value) || 0);
  return `${count.toLocaleString()} ${count === 1 ? "picture" : "pictures"}`;
}

/** The server's sentence, with a full stop when it has none; "" when empty. */
function sentence(text) {
  const trimmed = String(text ?? "").trim();
  return trimmed && !/[.!?]$/.test(trimmed) ? `${trimmed}.` : trimmed;
}

/**
 * Why a request about a run failed, in one sentence, or null when there is
 * nothing to tell: a request the session reset dropped answers nobody.
 *
 * A refusal that named a run has already put that run on the bar (the store
 * follows it), so these sentences point there rather than repeat its figures.
 *
 * @param {"start"|"stop"} action
 * @param {{kind: string, code: (string|null), detail: string,
 *   run: (Object|null)}} error - the run store's refusal.
 * @param {string} [gesture] - a start's gesture, one of
 *   {@link SCRAPHEAP_GESTURES}.
 * @returns {string|null}
 */
export function scrapheapRunFailureSentence(action, error, gesture) {
  const kind = error?.kind;
  if (kind === "cancelled") return null;
  if (kind === "busy") {
    return "The last Scrapheap request is still waiting for an answer.";
  }
  if (error?.code === "scrapheap_run_active") {
    return (
      "Another Scrapheap run is going, and its progress shows above the " +
      "grid. This can start once it ends."
    );
  }
  if (kind === "timeout" || kind === "network") {
    if (action === "stop") {
      return (
        "No answer came from the server, so it is not known whether the run " +
        "is stopping. Press Stop again to retry; a repeated stop is harmless."
      );
    }
    return error?.run?.active
      ? "No answer came from the server, but a Scrapheap run is going, and its progress shows above the grid."
      : "No answer came from the server, so it is not known whether it started. If it did, its progress shows above the grid.";
  }
  if (kind === "forbidden") {
    return "You don't have permission to change the Scrapheap.";
  }
  const lead = action === "stop" ? "Couldn't stop the run." : COULD_NOT[gesture];
  return `${lead} ${sentence(error?.detail) || "Please try again."}`;
}

/** What a purge left in the heap, one sentence per reason. */
function keptSentences(run) {
  const kept = [];
  const originals = Math.max(0, Number(run?.skipped_count) || 0);
  if (originals > 0) {
    kept.push(
      `${figure(originals)} reference-folder ` +
        `${originals === 1 ? "original was" : "originals were"} kept.`,
    );
  }
  const locked = Math.max(0, Number(run?.skipped_locked_count) || 0);
  if (locked > 0) {
    kept.push(
      `${picturesPhrase(locked)} in locked sets ` +
        `${locked === 1 ? "was" : "were"} kept.`,
    );
  }
  return kept;
}

/** What else a restore has to say: what it found gone, and its History. */
function restoreSentences(run) {
  const extra = [];
  const gone = Math.max(0, Number(run?.skipped_restored) || 0);
  if (gone > 0) {
    extra.push(`${picturesPhrase(gone)} had already left it.`);
  }
  const steps = Array.isArray(run?.segment_batch_ids)
    ? run.segment_batch_ids.length
    : 0;
  if (steps > 1) {
    extra.push(
      `History records it as ${figure(steps)} steps, each undone on its own.`,
    );
  }
  return extra;
}

/**
 * How a run this tab started ended, as a notice.
 *
 * A complete restore that raised its receipt says it there instead, and the
 * grid does not ask; a complete Delete forever on a selection that kept locked
 * pictures says it through the locked-set notice, which routes to the set.
 *
 * @param {Object} run - the run as its last poll read it.
 * @param {string} gesture - one of {@link SCRAPHEAP_GESTURES}.
 * @returns {{level: ("success"|"info"|"warning"|"error"), message: string}}
 */
export function scrapheapRunOutcomeNotice(run, gesture) {
  const purge = run?.kind === "purge";
  const done = Math.max(
    0,
    Number(purge ? run?.deleted_count : run?.restored_count) || 0,
  );
  const status = run?.status;
  if (status === "complete") {
    if (purge) {
      const lead =
        gesture === SCRAPHEAP_GESTURES.EMPTY
          ? `Emptied the Scrapheap: ${picturesPhrase(done)} deleted forever.`
          : `Deleted ${picturesPhrase(done)} forever.`;
      return {
        level: "success",
        message: [lead, ...keptSentences(run)].join(" "),
      };
    }
    return {
      level: "success",
      message: [
        `Restored ${picturesPhrase(done)} from the Scrapheap.`,
        ...restoreSentences(run),
      ].join(" "),
    };
  }
  const total = Math.max(0, Number(run?.total) || 0);
  const rest =
    done < total ? "; the rest are still in the Scrapheap." : ".";
  const tally = purge
    ? `Deleted ${figure(done)} of ${picturesPhrase(total)} forever${rest}`
    : `Restored ${figure(done)} of ${picturesPhrase(total)}${rest}`;
  const doing = purge ? "Deleting" : "Restoring";
  if (status === "cancelled") {
    return { level: "info", message: `${doing} stopped. ${tally}` };
  }
  if (status === "interrupted") {
    return {
      level: "warning",
      message: `${doing} was interrupted when the server restarted. ${tally}`,
    };
  }
  return {
    level: "error",
    message: [`${doing} stopped on an error. ${tally}`, sentence(run?.error)]
      .filter(Boolean)
      .join(" "),
  };
}
