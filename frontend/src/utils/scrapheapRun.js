/**
 * The words for the Scrapheap's background runs (`stores/useScrapheapRunStore`):
 * how far a run has got, how a run this tab started ended, and why a request
 * about a run was refused.
 *
 * The run store raises the outcomes as notices, in whatever view the user is
 * in; the grid (`components/views/ImageGrid.vue`) raises the refusals. While a
 * run goes, the bar above the grid (`widgets/EmptyScrapHeap.vue`) and its row
 * in Tasks (`panels/TasksPanel.vue`) carry its progress. Kept out of all of
 * them so that each sentence can be read, and tested, on its own.
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
  /** Undo restore (or Finish undo) on a restore run's outcome notice. */
  UNDO_RESTORE: "undo-restore",
});

/** What a start that failed was trying to do, as the old notices said it. */
const COULD_NOT = {
  [SCRAPHEAP_GESTURES.EMPTY]: "Couldn't empty the scrapheap.",
  [SCRAPHEAP_GESTURES.PURGE_SELECTION]: "Couldn't delete those pictures.",
  [SCRAPHEAP_GESTURES.RESTORE_ALL]: "Couldn't restore the scrapheap.",
  [SCRAPHEAP_GESTURES.RESTORE_SELECTION]: "Couldn't restore those pictures.",
  [SCRAPHEAP_GESTURES.UNDO_RESTORE]: "Couldn't undo the restore.",
};

/**
 * Why an undo was refused, for the codes that say more than "Couldn't undo
 * the restore." and the server's message would.
 * `scrapheap_run_active` is worded with every other run's.
 */
const UNDO_REFUSED = {
  nothing_to_undo: "That restore has already been undone.",
  not_a_restore_run: "Only a restore can be undone.",
  scrapheap_run_not_found:
    "That restore is no longer on record; undo its steps in History instead.",
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

/** What Stop leaves behind, said before it is pressed, by run kind. */
const STOP_KEEPS = {
  purge: "Pictures already deleted forever cannot be restored.",
  restore: "Pictures already restored stay restored.",
  undo: "Pictures already put back stay in the Scrapheap.",
};

/**
 * How far a run has got, as the Scrapheap bar and the run's Tasks row say it.
 *
 * A run's `processed` and `total` count pictures, except an undo's: it works
 * back through the restore's History steps, one step a slice, and its total is
 * their count. Its pictures are `undone_count`, known only as
 * each step is undone. Stop takes effect at the next slice boundary, so it
 * finishes the batch it is on (at most the run view's `slice_max_pictures`),
 * or for an undo the step it is on.
 *
 * @param {Object|null} run - a run view.
 * @returns {{processed: number, total: number, steps: boolean,
 *   determinate: boolean, percent: number, counted: string, stopTip: string}}
 *   `counted` is the progress in words ("300 of 1,200 pictures", "step 2 of
 *   5"); `percent` is a whole number in 0 to 100, and 0 until `determinate`
 *   (going, started, with a total).
 */
export function scrapheapRunProgress(run) {
  const total = Math.max(0, Number(run?.total) || 0);
  const processed = Math.max(0, Number(run?.processed) || 0);
  const steps = run?.kind === "undo";
  const determinate =
    Boolean(run?.active) && run?.status !== "pending" && total > 0;
  const percent = determinate
    ? Math.min(100, Math.round((processed / total) * 100))
    : 0;
  const counted = steps
    ? `step ${figure(processed)} of ${figure(total)}`
    : `${figure(processed)} of ${picturesPhrase(total)}`;
  const bound = Number(run?.slice_max_pictures);
  const after = steps
    ? "the History step it is on"
    : bound > 0
      ? `the current batch of up to ${figure(bound)} pictures`
      : "the current batch";
  const keeps = STOP_KEEPS[run?.kind] ?? STOP_KEEPS.purge;
  return {
    processed,
    total,
    steps,
    determinate,
    percent,
    counted,
    stopTip: `Stops after ${after}. ${keeps}`,
  };
}

/** The server's sentence, with a full stop when it has none; "" when empty. */
function sentence(text) {
  const trimmed = String(text ?? "").trim();
  return trimmed && !/[.!?]$/.test(trimmed) ? `${trimmed}.` : trimmed;
}

/**
 * What interrupted a run, from the code the server closed it with:
 * `library_restored` when a restore brought back a snapshot that held the run
 * as active, otherwise the server stopping (`server_stopped`).
 *
 * @param {Object} run
 * @returns {string}
 */
function interruptedBy(run) {
  return run?.error_code === "library_restored"
    ? "when the library was restored to a snapshot taken before it finished"
    : "when the server restarted";
}

/**
 * Why a request about a run failed, in one sentence, or null when there is
 * nothing to tell: a request the session reset dropped answers nobody.
 *
 * A refusal that named a run has already put that run on the bar (the store
 * follows it), so these sentences point there rather than repeat its figures.
 *
 * @param {"start"|"undo"|"stop"} action
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
    // The store sent the same request three times before giving up; a run
    // any of them started is the one a fourth would get back.
    return "No answer came from the server, so it is not known whether it started. If it did, its progress shows above the grid.";
  }
  if (kind === "forbidden") {
    return "You don't have permission to change the Scrapheap.";
  }
  if (action === "undo" && UNDO_REFUSED[error?.code]) {
    return UNDO_REFUSED[error.code];
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

/**
 * The pictures a purge left because they were binned after its confirmation
 * showed what it would destroy (`arrived_since`), or none.
 */
export function arrivedSentences(run) {
  const arrived = Math.max(0, Number(run?.arrived_since) || 0);
  if (arrived === 0) return [];
  return [
    `${picturesPhrase(arrived)} binned since you confirmed ` +
      `${arrived === 1 ? "is" : "are"} still in the Scrapheap.`,
  ];
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
 * store does not ask; a complete Delete forever on a selection that kept locked
 * pictures says it through the locked-set notice, which routes to the set.
 *
 * @param {Object} run - the run as its last poll read it.
 * @param {string} gesture - one of {@link SCRAPHEAP_GESTURES}.
 * @returns {{level: ("success"|"info"|"warning"|"error"), message: string}}
 */
export function scrapheapRunOutcomeNotice(run, gesture) {
  if (run?.kind === "undo") return undoOutcomeNotice(run);
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
        message: [lead, ...keptSentences(run), ...arrivedSentences(run)].join(
          " ",
        ),
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
  const arrived = purge ? arrivedSentences(run) : [];
  if (status === "cancelled") {
    return {
      level: "info",
      message: [`${doing} stopped. ${tally}`, ...arrived].join(" "),
    };
  }
  if (status === "interrupted") {
    return {
      level: "warning",
      message: [
        `${doing} was interrupted ${interruptedBy(run)}. ${tally}`,
        ...arrived,
      ].join(" "),
    };
  }
  return {
    level: "error",
    message: [
      `${doing} stopped on an error. ${tally}`,
      sentence(run?.error),
      ...arrived,
    ]
      .filter(Boolean)
      .join(" "),
  };
}

/**
 * The sentence an undo that History blocked ends with. `history_moved_on`
 * is the server's own: it names the History entry that sits above the
 * restore, which only the server knows, and says what to do.
 * `undo_refused` is a fixed reason, worded here.
 */
const UNDO_BLOCKED_LEAD = {
  undo_refused:
    "Undo stopped: a picture of this restore is now in a locked set or was deleted forever.",
};

/**
 * How an undo of a restore run ended, as a notice.
 *
 * @param {Object} run - the undo run (`kind: "undo"`) as its last poll read it.
 * @returns {{level: ("success"|"info"|"warning"|"error"), message: string}}
 */
function undoOutcomeNotice(run) {
  const done = Math.max(0, Number(run?.undone_count) || 0);
  const status = run?.status;
  if (status === "complete") {
    return {
      level: "success",
      message: `Put ${picturesPhrase(done)} back in the Scrapheap.`,
    };
  }
  if (status === "blocked" && run?.error_code === "history_moved_on") {
    return { level: "warning", message: sentence(run?.error) };
  }
  // `total` counts the restore's History steps, not pictures.
  const steps = Math.max(0, Number(run?.total) || 0);
  const rest =
    Math.max(0, Number(run?.processed) || 0) < steps
      ? "; the rest stay restored."
      : ".";
  const tally = `Put ${picturesPhrase(done)} back in the Scrapheap${rest}`;
  if (status === "blocked") {
    const lead =
      UNDO_BLOCKED_LEAD[run?.error_code] ??
      ["Undo stopped.", sentence(run?.error)].filter(Boolean).join(" ");
    return { level: "warning", message: `${lead} ${tally}` };
  }
  if (status === "cancelled") {
    return { level: "info", message: `Undo stopped. ${tally}` };
  }
  if (status === "interrupted") {
    return {
      level: "warning",
      message: `Undo was interrupted ${interruptedBy(run)}. ${tally}`,
    };
  }
  return {
    level: "error",
    message: [`Undo stopped on an error. ${tally}`, sentence(run?.error)]
      .filter(Boolean)
      .join(" "),
  };
}
